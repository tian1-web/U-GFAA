"""Replay observed constructor directory order across Windows/Linux.

Only dataset construction is wrapped. Input pixels, labels, sampling draws,
transforms, model code, loss and optimizer are untouched. The inventory is
specific to the audited seed-0 input selection.
"""
from contextlib import contextmanager
import glob
import json
import os
from pathlib import Path

@contextmanager
def replay_enumeration(data_root, manifest_path):
    root = Path(data_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    original_listdir, original_glob, original_walk = os.listdir, glob.glob, os.walk
    def key(path):
        try: return Path(path).relative_to(root).as_posix()
        except (ValueError, TypeError): return None
    def listdir(path):
        name = key(path)
        if name in manifest['listdir']: return list(manifest['listdir'][name])
        return original_listdir(path)
    def glob_paths(pattern, *args, **kwargs):
        name = key(pattern)
        if name in manifest['glob']:
            return [str(root / p) for p in manifest['glob'][name]]
        return original_glob(pattern, *args, **kwargs)
    def walk(path, *args, **kwargs):
        name = key(path)
        if name in manifest['walk']:
            return iter((str(root / p), list(d), list(f)) for p,d,f in manifest['walk'][name])
        return original_walk(path, *args, **kwargs)
    os.listdir, glob.glob, os.walk = listdir, glob_paths, walk
    try: yield
    finally: os.listdir, glob.glob, os.walk = original_listdir, original_glob, original_walk
