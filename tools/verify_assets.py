#!/usr/bin/env python3
"""Verify published engineering asset bytes without opening hardware."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/asset-manifest.json").read_text())
    checked = 0
    errors = []
    for row in manifest["assets"]:
        path = root / row["repository_path"]
        if not path.is_file():
            errors.append(f"Missing: {row['repository_path']}")
        elif path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            errors.append(f"Size/hash mismatch: {row['repository_path']}")
        else:
            checked += 1
    for error in errors:
        print(error)
    print(f"Verified {checked} published assets; {len(errors)} failures.")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
