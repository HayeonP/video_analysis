"""Resolve project and saved-result paths without importing the analysis stack."""
from __future__ import annotations

import hashlib
from pathlib import Path


def find_project_root(start: Path) -> Path | None:
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / "run.py").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    return None


PROJECT_ROOT = find_project_root(Path(__file__).parent) or Path(__file__).resolve().parents[1]


def project_input(path: Path) -> Path:
    """Honor existing caller paths, then try paths relative to this workspace."""
    path = path.expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate
    return path.resolve()


def result_directory(result: Path, filename: str) -> Path:
    result = project_input(result)
    if result.is_dir():
        return result
    if result.name == filename:
        return result.parent
    if result.is_file() or result.suffix.lower() == ".json":
        raise ValueError(f"분석 결과 폴더 또는 {filename}을 지정하세요: {result}")
    return result


def resolve_video(source: dict, base: Path, override: Path | None = None) -> Path:
    """Find a moved video, checking saved size/hash before accepting a fallback."""
    if override is not None:
        explicit = project_input(override)
        if explicit.is_file():
            return explicit
        raise ValueError(f"--video로 지정한 영상 파일이 없습니다: {explicit}")
    original = source.get("path")
    if not original:
        raise ValueError("원본 영상 경로가 없습니다. --video로 영상 파일을 지정하세요.")
    original = Path(original).expanduser()
    recorded = (base / original).resolve()
    if recorded.is_file():
        return recorded

    def matches(path: Path) -> bool:
        if not path.is_file():
            return False
        if source.get("size") is not None and path.stat().st_size != source["size"]:
            return False
        if source.get("sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest() == source["sha256"]
        return True

    relative = source.get("relative_path")
    if relative:
        relocated = (base / relative).resolve()
        if matches(relocated):
            return relocated

    # Older manifests only stored an absolute path. Rebase its directory suffix
    # against the moved project/result, e.g. old/videos/x.mp4 -> new/videos/x.mp4.
    roots = {base, base.parent, PROJECT_ROOT}
    project = find_project_root(base)
    if project:
        roots.add(project)
    if base.parent.name == "outputs":
        roots.add(base.parent.parent)
    parts = original.parts[1:] if original.is_absolute() else original.parts
    candidates = {(root.joinpath(*parts[index:])).resolve()
                  for root in roots for index in range(len(parts))}
    found = sorted(path for path in candidates if matches(path))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError("이동된 원본 영상 후보가 여러 개입니다. --video로 지정하세요: "
                         + ", ".join(str(path) for path in found))
    raise ValueError(f"원본 영상을 찾을 수 없습니다. --video로 새 경로를 지정하세요: {recorded}")
