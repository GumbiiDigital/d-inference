#!/usr/bin/env python3
"""Write or verify manifest.json for this evidence bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCOPE = (
    "MLX-ROCm on Strix Halo gfx1151: build, token match, speed vs llama.cpp, "
    "decode profile; 2026-10-12"
)
CAVEATS = [
    "One box, one day",
    "MLX and llama.cpp use different benchmark tools; ratios are approximate",
    "Fused-projection and env-switch results are prototypes, not shipped code",
    "The qmv_wide2 patches are a measured negative result, not a proposal",
    "M5 reference files come from a separate Apple M5 Max run",
]
SKIP_NAMES = {".git", "make_manifest.py", "manifest.json"}


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def file_entry(rel: str, path: Path) -> dict:
    st = path.stat()
    digest, nbytes = sha256_file(path)
    ent: dict = {
        "path": rel,
        "sha256": digest,
        "bytes": nbytes,
        "mode": stat.S_IMODE(st.st_mode),
    }
    if path.suffix == ".gz":
        raw = gzip.decompress(path.read_bytes())
        ent["uncompressed_bytes"] = len(raw)
        ent["uncompressed_sha256"] = hashlib.sha256(raw).hexdigest()
    return ent


def collect_files() -> list[dict]:
    files: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_NAMES]
        for name in filenames:
            if name in SKIP_NAMES:
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(ROOT).as_posix()
            files.append(file_entry(rel, path))
    files.sort(key=lambda e: e["path"])
    return files


def write_manifest() -> Path:
    doc = {
        "schema": 1,
        "scope": SCOPE,
        "caveats": CAVEATS,
        "files": collect_files(),
    }
    out = ROOT / "manifest.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out


def check_manifest() -> int:
    manifest_path = ROOT / "manifest.json"
    if not manifest_path.is_file():
        print("missing manifest.json", file=sys.stderr)
        return 1
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {e["path"] for e in doc["files"]}
    on_disk = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_NAMES]
        for name in filenames:
            if name in SKIP_NAMES:
                continue
            on_disk.add((Path(dirpath) / name).relative_to(ROOT).as_posix())

    err = 0
    for path in sorted(on_disk - listed):
        print(f"not listed: {path}", file=sys.stderr)
        err = 1
    for path in sorted(listed - on_disk):
        print(f"listed but missing: {path}", file=sys.stderr)
        err = 1

    for ent in doc["files"]:
        rel = ent["path"]
        path = ROOT / rel
        if not path.is_file():
            continue
        digest, nbytes = sha256_file(path)
        if digest != ent["sha256"] or nbytes != ent["bytes"]:
            print(f"mismatch: {rel}", file=sys.stderr)
            err = 1
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != ent["mode"]:
            print(f"mode mismatch: {rel}", file=sys.stderr)
            err = 1
        if path.suffix == ".gz":
            raw = gzip.decompress(path.read_bytes())
            ubytes = len(raw)
            usha = hashlib.sha256(raw).hexdigest()
            if ubytes != ent.get("uncompressed_bytes") or usha != ent.get(
                "uncompressed_sha256"
            ):
                print(f"uncompressed mismatch: {rel}", file=sys.stderr)
                err = 1
    return err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--check",
        action="store_true",
        help="verify manifest.json against files on disk",
    )
    args = p.parse_args()
    if args.check:
        raise SystemExit(check_manifest())
    write_manifest()
    print(f"wrote {ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
