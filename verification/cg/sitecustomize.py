"""Opt-in verification instrumentation; never installed with rgi_toolkit."""

import os

if os.environ.get("RGI_VERIFY_TRACE"):
    from trace_rgi import install

    install()
