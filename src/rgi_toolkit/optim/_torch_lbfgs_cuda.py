"""A native-precision CUDA axpy that accepts its coefficient on the device."""

import torch
import triton
import triton.language as tl


@triton.jit
def _axpy(x, other, alpha, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    a = tl.load(x + offsets, mask=mask)
    b = tl.load(other + offsets, mask=mask)
    scale = tl.load(alpha).to(a.dtype)
    tl.store(x + offsets, tl.fma(b, scale, a), mask=mask)


def add(x, other, alpha):
    with torch.cuda.device(x.device):
        _axpy[(triton.cdiv(x.numel(), 256),)](x, other, alpha, x.numel(), 256)
    return x


@triton.jit
def _direction(g, ys, ss, ro, h_diag, alpha, out, n, history, BLOCK: tl.constexpr):
    j = tl.arange(0, BLOCK)
    mask = j < n
    q = -tl.load(g + j, mask=mask, other=0)
    for k in range(history - 1, -1, -1):
        s = tl.load(ss + k * n + j, mask=mask, other=0)
        y = tl.load(ys + k * n + j, mask=mask, other=0)
        a = tl.sum(s * q, 0) * tl.load(ro + k)
        tl.store(alpha + k, a)
        q = tl.fma(-a, y, q)
    r = q * tl.load(h_diag)
    for k in range(history):
        y = tl.load(ys + k * n + j, mask=mask, other=0)
        s = tl.load(ss + k * n + j, mask=mask, other=0)
        b = tl.sum(y * r, 0) * tl.load(ro + k)
        delta = tl.load(alpha + k) - b
        r = tl.fma(delta, s, r)
    tl.store(out + j, r, mask=mask)


def direction(g, old_dirs, old_stps, ro, h_diag):
    if not old_dirs:
        return g.neg().mul(h_diag), []
    ys, ss, rho = torch.stack(old_dirs), torch.stack(old_stps), torch.stack(ro)
    alpha = g.new_empty(len(ro))
    result = torch.empty_like(g)
    with torch.cuda.device(g.device):
        _direction[(1,)](
            g,
            ys,
            ss,
            rho,
            h_diag,
            alpha,
            result,
            g.numel(),
            len(ro),
            triton.next_power_of_2(g.numel()),
        )
    return result, list(alpha.unbind())
