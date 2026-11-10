"""The flash-local-experts vLLM plugin, mounted read-only at /opt/experts-plugin on PYTHONPATH.

A plain package plus a .dist-info, so importlib.metadata finds its entry point from the
PYTHONPATH entry alone, with no pip install. The lane's installer copies this directory to
$CONFIG_DIR/flash-local-experts, and the unit mounts it.

It counts, per MoE layer, how many of the step's routed tokens went to each expert, and
writes the matrix to a file under /dev/shm twice a second. The cockpit's Experts view reads
that file (the unit runs with --ipc=host, so the host's /dev/shm is the container's) and
draws it as a heatmap over the model's layers. wire.py is the file's format, counter.py the
count, plugin.py the hook.
"""
