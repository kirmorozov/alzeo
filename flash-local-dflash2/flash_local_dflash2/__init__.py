"""The flash-local-dflash2 vLLM plugin, mounted at /opt/df2-plugin on PYTHONPATH.

It ships as a plain package plus a .dist-info, so importlib.metadata finds its
entry point from the PYTHONPATH entry alone, with no pip install. The lane's
installer copies this directory to $CONFIG_DIR/flash-local-dflash2, and the
unit mounts it read-only.
"""
