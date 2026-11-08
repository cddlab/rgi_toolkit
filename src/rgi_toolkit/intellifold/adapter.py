"""Framework-free adapter for IntelliFold's vendored AlphaFold 3 JAX engine."""

from rgi_toolkit.alphafold3.adapter import AF3RestraintAdapter


class IntelliFoldAdapter(AF3RestraintAdapter):
    """Use the shared AF3 token layout, chain mapping and ligand chemistry."""
