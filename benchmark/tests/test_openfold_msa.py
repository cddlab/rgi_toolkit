import shutil
from pathlib import Path

import pytest

from inputs import write_input


@pytest.mark.parametrize("arm", ["vanilla", "ms", "rgi"])
def test_native_openfold_msa_survives_relocation(tmp_path, monkeypatch, arm):
    pytest.importorskip("openfold3")
    from openfold3.core.config.msa_pipeline_configs import (
        MsaSampleProcessorInputInference,
    )
    from openfold3.core.data.io.sequence.msa import MsaSampleParserInference
    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    run = tmp_path / "run"
    data, folder = run / "data", run / "jobs" / "prediction"
    data.mkdir(parents=True)
    folder.mkdir(parents=True)
    name = "qbp-d16-m0.a3m" if arm == "ms" else "qbp.a3m"
    (data / name).write_text(">query\nAGST\n>homolog\nAGa-T\n")
    write_input(
        folder,
        {
            "model": "openfold-3_restr",
            "sequence": "AGST",
            "msa": f"../../data/{name}",
            "restraints": None,
            "seed": 0,
            "arm": arm,
        },
    )
    moved = tmp_path / "moved run"
    shutil.move(run, moved)
    monkeypatch.chdir(moved / "jobs" / "prediction")
    query = InferenceQuerySet.from_json(Path("input.json")).queries["prediction"]
    native_input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        query
    )
    parser = MsaSampleParserInference(MSASettings())
    parsed = parser.parse_msas(parser.create_maps(native_input))
    alignments = next(iter(parsed.rep_id_to_main_msa.values()))
    msa = alignments["uniref90_hits"]
    assert msa.msa.shape == (2, 4)
    assert msa.deletion_matrix[1, 2] == 1
    assert Path("uniref90_hits.a3m").readlink() == Path(f"../../data/{name}")
