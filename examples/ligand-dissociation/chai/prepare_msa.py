"""Convert the two unpaired A3Ms to Chai's aligned-Parquet input format."""

from pathlib import Path

from chai_lab.data.parsing.fasta import read_fasta
from chai_lab.data.parsing.msas.aligned_pqt import (
    a3m_to_aligned_dataframe,
    expected_basename,
)
from chai_lab.data.parsing.msas.data_source import MSADataSource


def main() -> None:
    here = Path(__file__).resolve().parent
    msa_dir = here.parent / "msas/a3m"
    output_dir = here.parent / "msas/chai"
    output_dir.mkdir(parents=True, exist_ok=True)
    proteins = read_fasta(here / "mdm2_p53.fasta")
    for protein, filename in zip(proteins, ["mdm2_p53.a3m", "p53.a3m"], strict=True):
        path = msa_dir / filename
        if read_fasta(path)[0].sequence != protein.sequence:
            raise ValueError(f"The MSA query in {filename} does not match the FASTA.")
        frame = a3m_to_aligned_dataframe(
            path, MSADataSource.BFD_UNICLUST, insert_pairing_key=False
        )
        frame.to_parquet(output_dir / expected_basename(protein.sequence), index=False)


if __name__ == "__main__":
    main()
