"""Explicit local-only provider and plugin registry contract."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Literal, Protocol

from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import ConfigurationError, DependencyUnavailableError

ProviderKind = Literal["vision", "speech", "embeddings", "story", "music", "voice"]
ProviderFactory = Callable[[Settings], object]
PROVIDER_ENTRY_POINT_GROUP = "travelmovieai.providers"
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9-]{1,63}$")


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    name: str
    kind: ProviderKind
    version: str
    local_only: bool = True
    optional_dependency: str | None = None
    model_heavy: bool = False


class ProviderPlugin(Protocol):
    def register(self, registry: "ProviderRegistry") -> None: ...


class ProviderRegistry:
    """Register factories without importing or initializing their models."""

    def __init__(self) -> None:
        self._providers: dict[
            tuple[ProviderKind, str], tuple[ProviderDescriptor, ProviderFactory]
        ] = {}

    def register(
        self,
        descriptor: ProviderDescriptor,
        factory: ProviderFactory,
    ) -> None:
        if not _PROVIDER_NAME.fullmatch(descriptor.name):
            raise ConfigurationError(
                "Provider name must be a lowercase slug between 2 and 64 characters."
            )
        if not descriptor.version.strip():
            raise ConfigurationError("Provider version must not be empty.")
        if not descriptor.local_only:
            raise ConfigurationError(
                "Remote providers are not allowed in this local-first registry."
            )
        key = (descriptor.kind, descriptor.name)
        if key in self._providers:
            raise ConfigurationError(
                f"Provider {descriptor.kind}:{descriptor.name} is already registered."
            )
        self._providers[key] = (descriptor, factory)

    def descriptors(self, kind: ProviderKind | None = None) -> tuple[ProviderDescriptor, ...]:
        values = (
            descriptor
            for descriptor, _ in self._providers.values()
            if kind is None or descriptor.kind == kind
        )
        return tuple(sorted(values, key=lambda item: (item.kind, item.name)))

    def create(self, kind: ProviderKind, name: str, settings: Settings) -> object:
        registered = self._providers.get((kind, name))
        if registered is None:
            raise DependencyUnavailableError(f"Local provider {kind}:{name} is not registered.")
        _, factory = registered
        return factory(settings)

    def load_entry_points(self) -> tuple[str, ...]:
        """Explicitly load installed plugins; never called during package import."""

        loaded: list[str] = []
        selected = entry_points(group=PROVIDER_ENTRY_POINT_GROUP)
        for entry_point in sorted(selected, key=lambda item: item.name):
            _load_plugin(entry_point, self)
            loaded.append(entry_point.name)
        return tuple(loaded)


def _load_plugin(entry_point: EntryPoint, registry: ProviderRegistry) -> None:
    try:
        plugin = entry_point.load()
    except (ImportError, AttributeError, ModuleNotFoundError) as error:
        raise DependencyUnavailableError(
            f"Could not load local provider plugin {entry_point.name!r}."
        ) from error
    register = getattr(plugin, "register", None)
    if not callable(register):
        raise ConfigurationError(
            f"Provider plugin {entry_point.name!r} does not implement register(registry)."
        )
    register(registry)
