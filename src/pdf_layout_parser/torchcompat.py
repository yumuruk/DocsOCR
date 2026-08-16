"""Compatibility shim for loading doclayout_yolo checkpoints on PyTorch >= 2.6.

PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default, which
refuses any pickle containing classes outside an allowlist. doclayout_yolo
checkpoints pickle the entire model object graph (their own module classes), so
the allowlist route would mean enumerating and chasing their internals across
versions. Training itself completes and writes best.pt -- the failure happens in
doclayout_yolo's own post-train reload, and again any time we open a checkpoint
for inference.

The shim scopes ``weights_only=False`` to the block that loads a checkpoint.
That flag means "trust this pickle to execute code", which is acceptable here
and only here because every file loaded through it is one of ours: the
DocStructBench weights placed in ``models/`` by the operator, or a checkpoint
this codebase just trained. Do not widen it into a global default, and do not
route user-uploaded files through it.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager


@contextmanager
def trusted_checkpoint_loading():
    """Within this block, ``torch.load`` defaults to ``weights_only=False``.

    Callers that pass ``weights_only`` explicitly are left untouched. The
    original function is always restored, exception or not.
    """
    import torch

    original = torch.load

    @functools.wraps(original)
    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original
