"""Bounded cleanup for reproducible project-local intermediate caches."""

from dataclasses import dataclass
from pathlib import Path

from travelmovieai.application.context import ProjectContext

MEBIBYTE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CacheCleanupResult:
    before_bytes: int
    after_bytes: int
    removed_bytes: int
    removed_files: int
    failed_files: int
    limit_bytes: int
    target_bytes: int


def cleanup_context_cache(context: ProjectContext) -> CacheCleanupResult:
    return cleanup_cache_roots(
        [context.cache_dir, context.frames_dir],
        limit_bytes=context.settings.project_cache_limit_mb * MEBIBYTE,
        target_ratio=context.settings.project_cache_target_ratio,
    )


def cleanup_cache_roots(
    roots: list[Path],
    *,
    limit_bytes: int,
    target_ratio: float = 0.85,
) -> CacheCleanupResult:
    if limit_bytes < 0:
        raise ValueError("Cache limit must not be negative.")
    if not 0.1 <= target_ratio <= 1:
        raise ValueError("Cache target ratio must be between 0.1 and 1.0.")
    resolved_roots = [root.expanduser().resolve() for root in roots]
    candidates: list[tuple[int, str, Path, int, Path]] = []
    before_bytes = 0
    for root in resolved_roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            before_bytes += stat.st_size
            candidates.append(
                (
                    stat.st_mtime_ns,
                    path.as_posix().casefold(),
                    path,
                    stat.st_size,
                    root,
                )
            )
    target_bytes = int(limit_bytes * target_ratio) if limit_bytes else 0
    if limit_bytes == 0 or before_bytes <= limit_bytes:
        return CacheCleanupResult(
            before_bytes=before_bytes,
            after_bytes=before_bytes,
            removed_bytes=0,
            removed_files=0,
            failed_files=0,
            limit_bytes=limit_bytes,
            target_bytes=target_bytes,
        )

    removed_bytes = 0
    removed_files = 0
    failed_files = 0
    remaining = before_bytes
    for _, _, path, size, root in sorted(candidates):
        if remaining <= target_bytes:
            break
        try:
            resolved_path = path.resolve(strict=True)
            if not resolved_path.is_relative_to(root) or path.is_symlink():
                failed_files += 1
                continue
            path.unlink()
        except FileNotFoundError:
            remaining -= size
            continue
        except OSError:
            failed_files += 1
            continue
        remaining -= size
        removed_bytes += size
        removed_files += 1

    _remove_empty_directories(resolved_roots)
    return CacheCleanupResult(
        before_bytes=before_bytes,
        after_bytes=max(0, remaining),
        removed_bytes=removed_bytes,
        removed_files=removed_files,
        failed_files=failed_files,
        limit_bytes=limit_bytes,
        target_bytes=target_bytes,
    )


def _remove_empty_directories(roots: list[Path]) -> None:
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        directories = sorted(
            (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                continue
