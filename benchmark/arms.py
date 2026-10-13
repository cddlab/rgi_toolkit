"""Experiment arms shared by preparation, execution, and reporting."""

ARMS = ("off", "vanilla", "ms", "rgi", "g-only", "rgi+g")
GUIDANCE_MODELS = frozenset({"boltz_restr", "protenix_restr"})


def uses_rgi(arm):
    return arm in {"rgi", "rgi+g"}


def uses_guidance(arm):
    return arm in {"g-only", "rgi+g"}


def selected_arms(benchmark, requested=None, single_sequence=False):
    if requested is None:
        if benchmark == "conformer":
            return ["off", "g-only", "rgi"]
        return ["vanilla", "rgi"] if single_sequence else ["vanilla", "ms", "rgi"]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Arms must be nonempty and unique")
    if set(requested) - set(ARMS):
        raise ValueError("Unknown benchmark arm")
    if "ms" in requested and (benchmark == "conformer" or single_sequence):
        raise ValueError("MSA subsampling requires a protein benchmark with --msa")
    if "off" in requested and "vanilla" in requested:
        raise ValueError("Choose off or vanilla, which are equivalent controls")
    return list(requested)
