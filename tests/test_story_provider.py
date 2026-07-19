from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import DependencyUnavailableError, StoryGenerationError
from travelmovieai.domain.enums import (
    ActivityType,
    LocationType,
    MediaType,
    StageStatus,
    StoryStyle,
)
from travelmovieai.domain.models import Event, MediaAsset, Scene, Storyboard
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.story import (
    LocalTransformersStoryProvider,
    parse_story_model_output,
)
from travelmovieai.pipeline.stages import storyboard as storyboard_stage
from travelmovieai.pipeline.stages.storyboard import StoryBuilderStage


def test_story_model_output_is_validated_and_derives_trusted_scene_ids() -> None:
    events, scenes = _events_and_scenes(Path("media"))
    content = _model_response(events)

    storyboard = parse_story_model_output(
        content,
        events,
        StoryStyle.CINEMATIC,
        provider="local-transformers",
        model="local-test-model",
    )

    assert storyboard.provider == "local-transformers"
    assert storyboard.model == "local-test-model"
    assert storyboard.event_ids == [event.id for event in events]
    assert storyboard.sections[0].scene_ids == [scenes[0].id]
    assert storyboard.sections[1].scene_ids == [scenes[1].id]


def test_story_model_output_rejects_unknown_or_missing_events() -> None:
    events, _ = _events_and_scenes(Path("media"))
    invalid = (
        '{"title":"Trip","sections":['
        f'{{"role":"opening","title":"Start","event_ids":["{events[0].id}"]}}]}}'
    )

    with pytest.raises(StoryGenerationError, match="every known event exactly once"):
        parse_story_model_output(
            invalid,
            events,
            StoryStyle.CINEMATIC,
            provider="local-transformers",
            model="local-test-model",
        )


def test_local_story_provider_is_lazy_and_honors_cache_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _Factory:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def from_pretrained(self, model: str, **kwargs: object) -> object:
            calls.append((f"{self.kind}:{model}", kwargs))
            if self.kind == "tokenizer":
                return object()
            return _FakeLoadedModel()

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_transformers = SimpleNamespace(
        AutoTokenizer=_Factory("tokenizer"),
        AutoModelForCausalLM=_Factory("model"),
    )

    def fake_import(name: str) -> object:
        return {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "accelerate": object(),
        }[name]

    monkeypatch.setattr("travelmovieai.infrastructure.story.importlib.import_module", fake_import)
    provider = LocalTransformersStoryProvider(
        "local-test-model",
        device="cpu",
        cache_dir=tmp_path / "models",
        allow_download=False,
    )

    assert provider._loaded_model is None
    assert not provider.cache_dir.exists()

    provider._ensure_loaded()

    assert provider.cache_dir.is_dir()
    assert [name for name, _ in calls] == [
        "tokenizer:local-test-model",
        "model:local-test-model",
    ]
    assert all(options["local_files_only"] is True for _, options in calls)


def test_local_story_provider_generates_with_deterministic_decoding() -> None:
    events, scenes = _events_and_scenes(Path("media"))
    provider = LocalTransformersStoryProvider("local-test-model", device="cpu")
    tokenizer = _FakeTokenizer(_model_response(events))
    model = _FakeGeneratingModel()
    provider._tokenizer = tokenizer
    provider._loaded_model = model
    provider._torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        inference_mode=nullcontext,
    )

    storyboard = provider.build(events, scenes, StoryStyle.CINEMATIC)

    assert storyboard.provider == "local-transformers"
    assert tokenizer.chat_template_calls == 1
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["use_cache"] is True


def test_local_story_provider_reports_missing_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingFactory:
        @staticmethod
        def from_pretrained(*_: object, **kwargs: object) -> object:
            assert kwargs["local_files_only"] is True
            raise OSError("model is not cached")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_transformers = SimpleNamespace(
        AutoTokenizer=_MissingFactory,
        AutoModelForCausalLM=_MissingFactory,
    )
    monkeypatch.setattr(
        "travelmovieai.infrastructure.story.importlib.import_module",
        lambda name: {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "accelerate": object(),
        }[name],
    )
    provider = LocalTransformersStoryProvider(
        "missing-local-model",
        device="cpu",
        cache_dir=tmp_path / "models",
        allow_download=False,
    )

    with pytest.raises(StoryGenerationError, match="cache-only mode"):
        provider._ensure_loaded()


def test_story_builder_stage_uses_local_provider_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, events, _ = _seed_context(tmp_path, story_provider="local")
    provider = _FakeStoryProvider(events)
    factory_calls = 0

    def fake_factory(**_: object) -> _FakeStoryProvider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    monkeypatch.setattr(storyboard_stage, "build_story_provider", fake_factory)

    first = StoryBuilderStage().run(context)
    second = StoryBuilderStage().run(context)
    storyboard = Storyboard.model_validate_json(
        (context.artifacts_dir / "storyboard.json").read_text(encoding="utf-8")
    )

    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert storyboard.provider == "local-transformers"
    assert storyboard.fallback_used is False
    assert factory_calls == 1
    assert provider.build_calls == 1
    assert provider.release_calls == 1


def test_story_builder_cache_invalidates_when_local_provider_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deterministic_context, events, _ = _seed_context(
        tmp_path,
        story_provider="deterministic",
    )
    first = StoryBuilderStage().run(deterministic_context)
    provider = _FakeStoryProvider(events)
    monkeypatch.setattr(storyboard_stage, "build_story_provider", lambda **_: provider)
    local_context = ProjectContext(
        input_path=deterministic_context.input_path,
        workspace=deterministic_context.workspace,
        settings=Settings(
            story_provider="local",
            story_model="local-test-model",
            allow_model_download=False,
        ),
    )

    changed = StoryBuilderStage().run(local_context)

    assert first.status is StageStatus.COMPLETED
    assert changed.status is StageStatus.COMPLETED
    assert provider.build_calls == 1
    assert "local-transformers" in changed.message


def test_local_story_cache_invalidates_when_multimodal_context_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, events, _ = _seed_context(tmp_path, story_provider="local")
    providers: list[_FakeStoryProvider] = []

    def fake_factory(**_: object) -> _FakeStoryProvider:
        provider = _FakeStoryProvider(events)
        providers.append(provider)
        return provider

    monkeypatch.setattr(storyboard_stage, "build_story_provider", fake_factory)
    first = StoryBuilderStage().run(context)
    repository = MediaAssetRepository(context.database_path)
    changed_scenes = [
        scene.model_copy(update={"transcript": "New local narration context."})
        for scene in repository.list_scenes()
    ]
    repository.synchronize_scenes(changed_scenes)

    changed = StoryBuilderStage().run(context)

    assert first.status is StageStatus.COMPLETED
    assert changed.status is StageStatus.COMPLETED
    assert len(providers) == 2
    assert all(provider.build_calls == 1 for provider in providers)


@pytest.mark.parametrize(
    "failure",
    [
        DependencyUnavailableError("offline cache is incomplete"),
        StoryGenerationError("invalid structured output"),
    ],
)
def test_story_builder_falls_back_without_poisoning_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    context, events, _ = _seed_context(tmp_path, story_provider="local")
    providers: list[_FailingStoryProvider] = []

    def fake_factory(**_: object) -> _FailingStoryProvider:
        provider = _FailingStoryProvider(events, failure)
        providers.append(provider)
        return provider

    monkeypatch.setattr(storyboard_stage, "build_story_provider", fake_factory)

    first = StoryBuilderStage().run(context)
    second = StoryBuilderStage().run(context)
    storyboard = Storyboard.model_validate_json(
        (context.artifacts_dir / "storyboard.json").read_text(encoding="utf-8")
    )

    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.COMPLETED
    assert storyboard.fallback_used is True
    assert storyboard.provider == "local-transformers"
    assert "will be retried" in first.message
    assert len(providers) == 2
    assert all(provider.release_calls == 1 for provider in providers)
    assert not (context.artifacts_dir / "storyboard.cache.json").exists()
    assert (
        storyboard_stage._cached_storyboard_valid(
            context.artifacts_dir / "storyboard.json",
            MediaAssetRepository(context.database_path).list_scenes(),
        )
        is False
    )


class _FakeLoadedModel:
    def to(self, _: str) -> "_FakeLoadedModel":
        return self

    def eval(self) -> None:
        return None


class _FakeTokenizer:
    def __init__(self, response: str) -> None:
        self.response = response
        self.chat_template_calls = 0

    def apply_chat_template(self, *_: object, **__: object) -> str:
        self.chat_template_calls += 1
        return "formatted prompt"

    def __call__(self, *_: object, **__: object) -> "_FakeInputs":
        return _FakeInputs()

    def decode(self, *_: object, **__: object) -> str:
        return self.response


class _FakeInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__({"input_ids": SimpleNamespace(shape=(1, 3))})

    def to(self, _: str) -> "_FakeInputs":
        return self


class _FakeGeneratingModel:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 4, 5]]


class _FakeStoryProvider:
    name = "local-transformers"
    model = "local-test-model"

    def __init__(self, events: list[Event]) -> None:
        self.events = events
        self.build_calls = 0
        self.release_calls = 0

    def build(
        self,
        events: list[Event],
        scenes: list[Scene],
        style: StoryStyle,
    ) -> Storyboard:
        assert events == self.events
        assert {scene.id for scene in scenes} == {
            scene_id for event in events for scene_id in event.scene_ids
        }
        self.build_calls += 1
        return parse_story_model_output(
            _model_response(events),
            events,
            style,
            provider=self.name,
            model=self.model,
        )

    def release(self) -> None:
        self.release_calls += 1


class _FailingStoryProvider(_FakeStoryProvider):
    def __init__(self, events: list[Event], failure: Exception) -> None:
        super().__init__(events)
        self.failure = failure

    def build(
        self,
        events: list[Event],
        scenes: list[Scene],
        style: StoryStyle,
    ) -> Storyboard:
        del events, scenes, style
        self.build_calls += 1
        raise self.failure


def _seed_context(
    tmp_path: Path,
    *,
    story_provider: str,
) -> tuple[ProjectContext, list[Event], list[Scene]]:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings.model_validate(
            {
                "story_provider": story_provider,
                "story_model": "local-test-model",
                "allow_model_download": False,
            }
        ),
    )
    context.prepare()
    events, scenes = _events_and_scenes(tmp_path / "media")
    assets = [
        _asset(tmp_path / "media" / f"scene-{index}.mp4", scene.asset_id, index)
        for index, scene in enumerate(scenes)
    ]
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize(assets, datetime.now(UTC))
    repository.synchronize_scenes(scenes)
    repository.synchronize_events(events)
    return context, events, scenes


def _events_and_scenes(root: Path) -> tuple[list[Event], list[Scene]]:
    asset_ids = [
        UUID("00000000-0000-0000-0000-000000000101"),
        UUID("00000000-0000-0000-0000-000000000102"),
    ]
    scene_ids = [
        UUID("00000000-0000-0000-0000-000000000201"),
        UUID("00000000-0000-0000-0000-000000000202"),
    ]
    event_ids = [
        UUID("00000000-0000-0000-0000-000000000301"),
        UUID("00000000-0000-0000-0000-000000000302"),
    ]
    scenes = [
        Scene(
            id=scene_id,
            asset_id=asset_id,
            start_seconds=0,
            end_seconds=5,
            caption=f"Scene {index}",
            importance_score=80 + index,
            metadata={"event_id": str(event_id)},
        )
        for index, (scene_id, asset_id, event_id) in enumerate(
            zip(scene_ids, asset_ids, event_ids, strict=True)
        )
    ]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        Event(
            id=event_id,
            title=f"Event {index}",
            scene_ids=[scene.id],
            summary=f"Summary {index}",
            importance_score=80 + index,
            start_at=start + timedelta(hours=index),
            end_at=start + timedelta(hours=index, seconds=5),
            location_type=LocationType.CITY,
            activity=ActivityType.SIGHTSEEING,
            confidence=0.9,
        )
        for index, (event_id, scene) in enumerate(zip(event_ids, scenes, strict=True))
    ]
    del root
    return events, scenes


def _asset(path: Path, asset_id: UUID, index: int) -> MediaAsset:
    created_at = datetime(2026, 1, 1, index, tzinfo=UTC)
    return MediaAsset(
        id=asset_id,
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=5,
    )


def _model_response(events: list[Event]) -> str:
    return (
        '{"title":"A Local Journey","sections":['
        f'{{"role":"opening","title":"Arrival","event_ids":["{events[0].id}"]}},'
        f'{{"role":"finale","title":"Finale","event_ids":["{events[1].id}"]}}]}}'
    )
