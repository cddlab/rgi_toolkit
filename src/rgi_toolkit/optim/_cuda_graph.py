"""Fuse straight-line solver regions inside native CUDA IF/WHILE graphs.

Both branches are recorded once, but only the selected branch executes on replay.
Each capture owns its pool: sharing pools across conditional branches can alias
intermediates that remain live in another region.
"""

import ctypes
import gc
import operator
import weakref
from dataclasses import dataclass, field

import torch
from cuda.bindings import driver as cu
from cuda.bindings import runtime as rt
from torch.fx import Graph, GraphModule
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

from rgi_toolkit.optim._cuda_graph_ops import (
    checked,
    condition_kernel,
    copy_tree,
    empty_tree,
)


@dataclass
class Region:
    operations: list = field(default_factory=list)


class Tape(TorchDispatchMode):
    def __init__(self, owner):
        super().__init__()
        self.owner = weakref.proxy(owner)
        self.graph = Graph()
        self.nodes = {}
        self.tensors = {}
        self.inputs = []
        self.created = []

    def argument(self, value):
        if not isinstance(value, torch.Tensor):
            return value
        if not value.is_cuda:
            raise RuntimeError("CUDA solver regions require device-resident constants")
        key = id(value)
        if key not in self.nodes:
            self.nodes[key] = self.graph.placeholder(f"arg_{len(self.inputs)}")
            self.tensors[key] = value
            self.inputs.append(value)
        return self.nodes[key]

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function == torch.ops.aten._local_scalar_dense.default:
            raise RuntimeError(
                "CUDA solver regions cannot read tensor scalars on the host"
            )
        kwargs = kwargs or {}
        decomposition = function in (
            torch.ops.aten._linalg_svd.default,
            torch.ops.aten._linalg_eigh.default,
        )
        if decomposition and (
            args[0].shape[-2:] != (3, 3)
            or args[0].dtype not in (torch.float32, torch.float64)
        ):
            raise RuntimeError(
                "Only real 3x3 decompositions support CUDA solver capture"
            )
        if function == torch.ops.aten._linalg_check_errors.default:
            raise RuntimeError(
                "This custom linear algebra operation requires host error checking"
            )
        if torch.Tag.nondeterministic_seeded in function.tags:
            raise RuntimeError("CUDA solver graphs require deterministic objectives")
        mapped_args = _pytree.tree_map(self.argument, args)
        mapped_kwargs = _pytree.tree_map(self.argument, kwargs)
        target = function
        eager_args, eager_kwargs = args, kwargs
        if decomposition:
            from rgi_toolkit.optim._cuda_linalg import eigh3, svd3

            self.owner.uses_linalg = True
            flag = self.argument(self.owner.library_failure)
            if function == torch.ops.aten._linalg_svd.default:
                if len(args) > 2 and not args[2]:
                    raise RuntimeError("Captured SVD requires singular vectors")
                if kwargs.get("driver") not in (None, "gesvdj"):
                    raise RuntimeError("Captured SVD retains the default gesvdj driver")
                target, mapped_args = svd3._opoverload, (mapped_args[0], flag)
                eager_args = (args[0], self.owner.library_failure)
            else:
                if len(args) > 2 and not args[2]:
                    raise RuntimeError("Captured eigendecomposition requires vectors")
                upper = (args[1] if len(args) > 1 else kwargs.get("UPLO", "L")) == "U"
                target, mapped_args = eigh3._opoverload, (mapped_args[0], flag, upper)
                eager_args = (args[0], self.owner.library_failure, upper)
            mapped_kwargs = {}
            eager_kwargs = {}
        result = target(*eager_args, **eager_kwargs)
        node = self.graph.call_function(target, mapped_args, mapped_kwargs)

        def record(value, selected):
            if isinstance(value, torch.Tensor):
                key = id(value)
                if key not in self.tensors:
                    self.created.append(value)
                self.tensors[key] = value
                self.nodes[key] = selected
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value):
                    record(
                        child,
                        self.graph.call_function(operator.getitem, (selected, index)),
                    )

        record(result, node)
        return result

    def module(self, required):
        # Export only tensors consumed by another region or by the caller.
        destinations = []
        for value in self.created:
            if id(value) not in required:
                continue
            target = self.graph.placeholder(f"destination_{len(destinations)}")
            self.graph.call_function(
                torch.ops.aten.copy_.default, (target, self.nodes[id(value)])
            )
            destinations.append(value)
        self.graph.output(())
        self.graph.eliminate_dead_code()
        self.graph.lint()
        return GraphModule({}, self.graph), tuple(self.inputs + destinations)


class FusedConditionalGraph:
    def __init__(self):
        self.device = torch.cuda.current_device()
        self.kernel = condition_kernel()
        self.root = checked(rt.cudaGraphCreate(0))
        self.root_region = Region()
        self.region = self.root_region
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.captures = []
        self.handles = []
        self.tapes = []
        self.predicates = []
        self.executable = None
        self.tape = None
        self.library_failure = torch.zeros((), dtype=torch.bool, device="cuda")
        self.uses_linalg = False
        self.started = False

    def begin(self):
        assert self.tape is None
        self.tape = Tape(self)
        self.tape.__enter__()
        if not self.started:
            self.library_failure.zero_()
            self.started = True

    def flush(self):
        assert self.tape is not None
        self.tape.__exit__(None, None, None)
        self.region.operations.append(self.tape)
        self.tapes.append(self.tape)
        self.tape = None

    def cond(self, predicate, yes, no, operand):
        self.flush()
        parent = self.region
        first_region, second_region = Region(), Region()
        parent.operations.append((predicate, "if", first_region, second_region))
        self.predicates.append(predicate)
        self.region = first_region
        self.begin()
        first = yes(operand)
        self.flush()
        result = empty_tree(first)
        self.begin()
        copy_tree(result, first)
        self.flush()
        self.region = second_region
        self.begin()
        second = no(operand)
        copy_tree(result, second)
        self.flush()
        self.region = parent
        self.begin()
        return result

    def loop(self, condition, body, state):
        self.flush()
        carried = empty_tree(state)
        self.begin()
        copy_tree(carried, state)
        first_predicate = condition(carried)
        predicate = torch.empty_like(first_predicate)
        predicate.copy_(first_predicate)
        self.flush()
        parent, body_region = self.region, Region()
        parent.operations.append((predicate, "while", body_region))
        self.predicates.append(predicate)
        self.region = body_region
        self.begin()
        result = body(carried)
        copy_tree(carried, result)
        predicate.copy_(condition(carried))
        self.flush()
        self.region = parent
        self.begin()
        return carried

    def capture(self, function, *args):
        captured = torch.cuda.CUDAGraph(keep_graph=True)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            # Library handles and workspaces are stream-local (notably cuSOLVER).
            # Initialize them on the stream that will actually be captured.
            function(*args)
            function(*args)
            torch.cuda.synchronize()
            captured.capture_begin()
            try:
                function(*args)
            finally:
                captured.capture_end()
        self.captures.append(captured)
        return rt.cudaGraph_t(captured.raw_cuda_graph())

    def setter(self, handle, predicate):
        captured = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(self.stream):
            captured.capture_begin()
            try:
                checked(
                    cu.cuLaunchKernel(
                        self.kernel,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        torch.cuda.current_stream().cuda_stream,
                        (
                            (int(handle), predicate.data_ptr()),
                            (ctypes.c_ulonglong, ctypes.c_void_p),
                        ),
                        0,
                    )
                )
            finally:
                captured.capture_end()
        self.captures.append(captured)
        return rt.cudaGraph_t(captured.raw_cuda_graph())

    def emit(self, region, native, required):
        last = []

        def child(graph):
            last[:] = [
                checked(rt.cudaGraphAddChildGraphNode(native, last, len(last), graph))
            ]

        for operation in region.operations:
            if isinstance(operation, Tape):
                module, arguments = operation.module(required)
                if not any(node.op == "call_function" for node in module.graph.nodes):
                    continue
                compiled = torch.compile(module, dynamic=False, fullgraph=True)
                child(self.capture(compiled, *arguments))
                continue
            predicate, kind, *bodies = operation
            handle = checked(rt.cudaGraphConditionalHandleCreate(self.root, 0, 0))
            self.handles.append(handle)
            child(self.setter(handle, predicate))
            parameters = rt.cudaGraphNodeParams()
            parameters.type = rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
            parameters.conditional.handle = handle
            parameters.conditional.type = (
                rt.cudaGraphConditionalNodeType.cudaGraphCondTypeIf
                if kind == "if"
                else rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
            )
            parameters.conditional.size = len(bodies)
            last[:] = [
                checked(rt.cudaGraphAddNode(native, last, len(last), parameters))
            ]
            for body, body_graph in zip(bodies, parameters.conditional.phGraph_out):
                end = self.emit(body, body_graph, required)
                if kind == "while":
                    checked(
                        rt.cudaGraphAddChildGraphNode(
                            body_graph, end, len(end), self.setter(handle, predicate)
                        )
                    )
        return last

    def finish(self, output):
        self.flush()
        self.output = output
        required = {id(x) for tape in self.tapes for x in tape.inputs}
        required.update(id(x) for x in self.predicates)
        required.update(
            id(x) for x in _pytree.tree_leaves(output) if isinstance(x, torch.Tensor)
        )
        # Autodiff can return a broadcast view for a constant gradient. An export
        # buffer must be writable even when its value was originally a view.
        materialized = set()
        for tape in self.tapes:
            for value in tape.created:
                if id(value) not in required or id(value) in materialized:
                    continue
                if any(
                    size > 1 and stride == 0
                    for size, stride in zip(value.shape, value.stride())
                ):
                    with torch.no_grad():
                        value.set_(value.clone(memory_format=torch.contiguous_format))
                    materialized.add(id(value))
        self.materialized_exports = len(materialized)
        torch.cuda.synchronize()
        # The tensors backing every captured region remain alive in this builder.
        # Reclaim unused allocations once; capture_begin/end do not require GC.
        gc.collect()
        torch.cuda.empty_cache()
        self.emit(self.root_region, self.root, required)
        self.executable = checked(rt.cudaGraphInstantiate(self.root, 0))

    def replay(self):
        checked(
            rt.cudaGraphLaunch(self.executable, torch.cuda.current_stream().cuda_stream)
        )
        return self.output

    def close(self):
        """Release native handles after every outstanding replay has completed."""
        if self.tape is not None:
            self.tape.__exit__(None, None, None)
            self.tape = None
        if self.root is None:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            if self.executable is not None:
                checked(rt.cudaGraphExecDestroy(self.executable))
                self.executable = None
            checked(rt.cudaGraphDestroy(self.root))
            self.root = None
            if self.uses_linalg:
                from rgi_toolkit.optim._cuda_linalg import release_stream

                release_stream(self.device, self.stream.cuda_stream)
        self.captures.clear()
        self.tapes.clear()
        self.predicates.clear()
        self.root_region.operations.clear()

    def __del__(self):
        # Interpreter shutdown may already have unloaded the CUDA runtime.
        if getattr(self, "root", None) is not None:
            try:
                self.close()
            except Exception:
                pass
