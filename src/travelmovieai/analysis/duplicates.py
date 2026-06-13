"""Perceptual duplicate detection for representative scene frames."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from travelmovieai.domain.models import (
    DuplicateDetectionReport,
    DuplicateGroup,
    Scene,
)


def detect_duplicate_scenes(
    scenes: list[Scene],
    similarity_threshold: float = 0.92,
) -> tuple[DuplicateDetectionReport, list[Scene]]:
    """Mark near-identical scenes while retaining the strongest representative."""
    fingerprints = {
        scene.id: fingerprint
        for scene in scenes
        if (fingerprint := _scene_fingerprint(scene.keyframe_path)) is not None
    }
    scene_by_id = {scene.id: scene for scene in scenes}
    parent = {scene_id: scene_id for scene_id in fingerprints}
    buckets: dict[tuple[int, int], list[UUID]] = {}

    for scene_id, fingerprint in fingerprints.items():
        candidates: set[UUID] = set()
        for block_index in range(8):
            block = (fingerprint >> (block_index * 8)) & 0xFF
            key = (block_index, block)
            candidates.update(buckets.get(key, []))
            buckets.setdefault(key, []).append(scene_id)
        for candidate_id in candidates:
            similarity = _hash_similarity(fingerprint, fingerprints[candidate_id])
            if similarity >= similarity_threshold:
                _union(parent, scene_id, candidate_id)

    components: dict[UUID, list[UUID]] = {}
    for scene_id in fingerprints:
        components.setdefault(_find(parent, scene_id), []).append(scene_id)

    groups: list[DuplicateGroup] = []
    duplicate_of: dict[UUID, tuple[UUID, float]] = {}
    for component in components.values():
        if len(component) < 2:
            continue
        keeper_id = max(component, key=lambda scene_id: _keeper_score(scene_by_id[scene_id]))
        duplicate_ids = [scene_id for scene_id in component if scene_id != keeper_id]
        similarities = [
            _hash_similarity(fingerprints[keeper_id], fingerprints[scene_id])
            for scene_id in duplicate_ids
        ]
        group_similarity = min(similarities)
        groups.append(
            DuplicateGroup(
                keeper_scene_id=keeper_id,
                duplicate_scene_ids=duplicate_ids,
                similarity=group_similarity,
            )
        )
        duplicate_of.update(
            (
                scene_id,
                (
                    keeper_id,
                    _hash_similarity(fingerprints[keeper_id], fingerprints[scene_id]),
                ),
            )
            for scene_id in duplicate_ids
        )

    updated = []
    keeper_ids = {group.keeper_scene_id for group in groups}
    for scene in scenes:
        metadata = {**scene.metadata}
        fingerprint = fingerprints.get(scene.id)
        if fingerprint is not None:
            metadata["perceptual_hash"] = f"{fingerprint:016x}"
        if scene.id in duplicate_of:
            keeper_id, similarity = duplicate_of[scene.id]
            metadata.update(
                {
                    "duplicate_of": str(keeper_id),
                    "duplicate_similarity": similarity,
                    "duplicate_status": "duplicate",
                }
            )
        elif scene.id in keeper_ids:
            metadata.update(
                {
                    "duplicate_of": None,
                    "duplicate_similarity": 1.0,
                    "duplicate_status": "keeper",
                }
            )
        else:
            metadata.update(
                {
                    "duplicate_of": None,
                    "duplicate_similarity": 0.0,
                    "duplicate_status": "unique",
                }
            )
        updated.append(scene.model_copy(update={"metadata": metadata}))

    duplicate_count = sum(len(group.duplicate_scene_ids) for group in groups)
    report = DuplicateDetectionReport(
        created_at=datetime.now(UTC),
        groups=groups,
        unique_count=len(scenes) - duplicate_count,
        duplicate_count=duplicate_count,
    )
    return report, updated


def _scene_fingerprint(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        with Image.open(path) as source:
            image = source.convert("L")
            width, height = image.size
            if width >= height * 2.2:
                panel_width = width // 3
                image = image.crop((panel_width, 0, panel_width * 2, height))
            resized = image.resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(resized.getdata())
    except (OSError, UnidentifiedImageError):
        return None
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def _hash_similarity(first: int, second: int) -> float:
    return 1 - (first ^ second).bit_count() / 64


def _keeper_score(scene: Scene) -> tuple[float, float, float]:
    override = str(scene.metadata.get("selection_override", "auto"))
    forced = 1.0 if override == "include" else 0.0
    importance = scene.importance_score or 0.0
    quality = scene.quality_score or 0.0
    return forced, importance, quality


def _find(parent: dict[UUID, UUID], item: UUID) -> UUID:
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != item:
        next_item = parent[item]
        parent[item] = root
        item = next_item
    return root


def _union(parent: dict[UUID, UUID], first: UUID, second: UUID) -> None:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root
