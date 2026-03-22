from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return app_root()
    return app_root()


def search_roots() -> tuple[Path, ...]:
    roots: list[Path] = [app_root(), bundle_root(), Path.cwd()]
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def resource_path(*parts: str) -> Path:
    relative = Path(*parts)
    for root in search_roots():
        candidate = root / relative
        if candidate.exists():
            return candidate
    return app_root() / relative


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded

    for root in search_roots():
        candidate = root / expanded
        if candidate.exists():
            return candidate
    return expanded
