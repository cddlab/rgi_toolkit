"""Sample one MDM2-p53 structure with the supplied per-chain alignments."""

import argparse
from pathlib import Path

from bioemu2 import EDMSampler, EmuSession, System, get_msa, load_model, save_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    input_path = args.input.resolve()
    system = System.from_yaml(input_path)
    msa_dir = input_path.parent.parent / "msas/a3m"
    msa = get_msa(
        system,
        {"A": msa_dir / "mdm2_p53.a3m", "B": msa_dir / "p53.a3m"},
    )
    session = EmuSession(system, model=load_model(), mode="structure", msa=msa)
    samples = session.sample(
        num_samples=1,
        sampler=EDMSampler(num_steps=200, step_scale=1.5),
        seed=0,
    )
    save_frames(samples, input_path.parent / "out" / input_path.stem)


if __name__ == "__main__":
    main()
