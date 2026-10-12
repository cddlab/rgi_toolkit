from pathlib import Path

import pytest

from models import chai_msa, command


@pytest.mark.parametrize("restraints", [None, {"conformer_restraints_config": {}}])
def test_chai_command_always_uses_prepared_msa(restraints):
    cmd = command(
        {
            "model": "chai-lab_restr",
            "seed": 1,
            "restraints": restraints,
            "arm": "rgi" if restraints else "vanilla",
        },
        {},
    )
    assert cmd[cmd.index("--msa-directory") + 1] == "msa"
    assert ("--restraints-config-path" in cmd) == bool(restraints)


def test_native_chai_msa_conversion_and_stale_query_rejection(tmp_path, monkeypatch):
    pytest.importorskip("chai_lab")
    from types import SimpleNamespace

    import torch
    from chai_lab.data.dataset.msas.load import get_msa_contexts
    from chai_lab.data.parsing.msas.aligned_pqt import (
        expected_basename,
        parse_aligned_pqt_to_msa_context,
    )
    from chai_lab.data.parsing.structure.entity_type import EntityType

    monkeypatch.chdir(tmp_path)
    a3m = Path("input.a3m")
    a3m.write_text("#4\t1\n>query\nAGST\n>homolog\nAGa-T\n")
    job = {"sequence": "AGST", "msa": str(a3m)}
    chai_msa(job)
    pqt = Path("msa") / expected_basename(job["sequence"])
    context = parse_aligned_pqt_to_msa_context(pqt)
    assert context.tokens.shape == (2, 4)
    assert context.deletion_matrix[1, 2] == 1
    chain = SimpleNamespace(
        entity_data=SimpleNamespace(
            pdb_id="test", sequence="AGST", entity_type=EntityType.PROTEIN
        ),
        structure_context=SimpleNamespace(token_residue_index=torch.arange(4)),
    )
    joined, profile = get_msa_contexts([chain], Path("msa"))
    assert joined.depth == profile.depth == 2
    original = pqt.read_bytes()
    a3m.write_text(">wrong_query\nAAAA\n>homolog\nAA-A\n")
    with pytest.raises(ValueError, match="query mismatch"):
        chai_msa(job)
    assert pqt.read_bytes() == original
