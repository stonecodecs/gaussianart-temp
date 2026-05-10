#!/usr/bin/env python3
"""
Extract all MPArt-90 (or other) scene archives into ./data.

Safely unpacks every .zip under the given source directories (recursive),
skipping entries that would escape the destination (zip-slip).

Example:
  python unzip_mpart_data.py --source MPArt-90
  python unzip_mpart_data.py --source MPArt-90/.cache/huggingface/download ./downloads
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def _safe_members(zf: zipfile.ZipFile, dest: Path) -> list[zipfile.ZipInfo]:
    dest_resolved = dest.resolve()
    members: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"Unsafe path in archive: {name!r}")
        target = (dest_resolved / name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as e:
            raise ValueError(f"Zip-slip blocked: {name!r}") from e
        members.append(info)
    return members


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_members(zf, dest)
        zf.extractall(dest)


def collect_zips(sources: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for src in sources:
        if not src.exists():
            print(f"Warning: source path does not exist, skipping: {src}", file=sys.stderr)
            continue
        if src.is_file() and src.suffix.lower() == ".zip":
            p = src.resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
            continue
        for z in sorted(src.rglob("*.zip")):
            p = z.resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def already_have_scene(dest: Path, zip_path: Path) -> bool:
    """Return True if this archive appears already extracted under dest."""
    stem = zip_path.stem
    if (dest / stem / "gt" / "trans.json").is_file():
        return True
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            roots: set[str] = set()
            for name in zf.namelist():
                parts = [p for p in name.split("/") if p]
                if parts:
                    roots.add(parts[0])
            if len(roots) == 1:
                only = roots.pop()
                if (dest / only / "gt" / "trans.json").is_file():
                    return True
    except zipfile.BadZipFile:
        pass
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Unzip all scene .zip files into ./data")
    parser.add_argument(
        "--source",
        nargs="+",
        type=Path,
        default=None,
        help="Files or directories to search (recursive for dirs). "
        "Default: MPArt-90 (if present) else repository root.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root / "data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a matching scene already exists under --dest",
    )
    args = parser.parse_args()

    if args.source is None:
        default_dirs = [repo_root / "MPArt-90", repo_root]
        sources = [p for p in default_dirs if p.exists()]
        if not sources:
            sources = [repo_root]
    else:
        sources = [p.resolve() if p.is_absolute() else (repo_root / p).resolve() for p in args.source]

    dest = args.dest.resolve()
    if not dest.is_absolute():
        dest = (repo_root / dest).resolve()

    zips = collect_zips(sources)
    if not zips:
        print("No .zip files found.", file=sys.stderr)
        return 1

    print(f"Destination: {dest}")
    print(f"Found {len(zips)} archive(s).")

    for zp in zips:
        if not args.force and already_have_scene(dest, zp):
            print(f"Skip (already present): {zp.name}")
            continue
        print(f"Extracting: {zp}")
        try:
            extract_zip(zp, dest)
        except (zipfile.BadZipFile, ValueError) as e:
            print(f"Failed: {zp} — {e}", file=sys.stderr)
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
