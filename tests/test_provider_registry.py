from dataclasses import dataclass

import pytest

from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import ConfigurationError, DependencyUnavailableError
from travelmovieai.infrastructure import providers
from travelmovieai.infrastructure.providers import (
    ProviderDescriptor,
    ProviderRegistry,
)


def test_provider_registry_keeps_model_factory_lazy() -> None:
    registry = ProviderRegistry()
    calls = 0

    def factory(settings: Settings) -> object:
        nonlocal calls
        calls += 1
        return {"model": settings.vision_model}

    registry.register(
        ProviderDescriptor(
            name="local-test",
            kind="vision",
            version="1",
            model_heavy=True,
        ),
        factory,
    )

    assert calls == 0
    assert registry.descriptors("vision")[0].name == "local-test"
    assert registry.create("vision", "local-test", Settings()) == {"model": "auto"}
    assert calls == 1


def test_provider_registry_rejects_remote_and_duplicate_providers() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ConfigurationError, match="Remote providers"):
        registry.register(
            ProviderDescriptor(
                name="remote-test",
                kind="story",
                version="1",
                local_only=False,
            ),
            lambda _: object(),
        )
    descriptor = ProviderDescriptor(name="local-test", kind="story", version="1")
    registry.register(descriptor, lambda _: object())

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(descriptor, lambda _: object())


def test_provider_registry_reports_missing_provider() -> None:
    with pytest.raises(DependencyUnavailableError, match="not registered"):
        ProviderRegistry().create("voice", "missing", Settings())


def test_provider_plugins_load_only_on_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ProviderRegistry()
    calls: list[str] = []

    @dataclass
    class Plugin:
        def register(self, target: ProviderRegistry) -> None:
            calls.append("registered")
            target.register(
                ProviderDescriptor(name="plugin-voice", kind="voice", version="1"),
                lambda _: object(),
            )

    @dataclass
    class FakeEntryPoint:
        name: str

        def load(self) -> Plugin:
            calls.append("loaded")
            return Plugin()

    monkeypatch.setattr(
        providers,
        "entry_points",
        lambda **_: [FakeEntryPoint("test-plugin")],
    )

    assert calls == []
    assert registry.load_entry_points() == ("test-plugin",)
    assert calls == ["loaded", "registered"]
    assert registry.descriptors("voice")[0].name == "plugin-voice"


def test_provider_plugin_must_implement_register(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEntryPoint:
        name = "invalid"

        def load(self) -> object:
            return object()

    monkeypatch.setattr(providers, "entry_points", lambda **_: [FakeEntryPoint()])

    with pytest.raises(ConfigurationError, match="register"):
        ProviderRegistry().load_entry_points()
