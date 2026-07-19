import tomllib
from pathlib import Path


def test_windows_installer_is_per_user_and_launches_desktop_shell() -> None:
    repository = Path(__file__).parents[1]
    installer = (repository / "installer" / "TravelMovieAI.iss").read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in installer
    assert "TravelMovieAI.exe" in installer
    assert 'WorkingDir: "{localappdata}\\{#AppName}"' in installer
    assert "[Dirs]" in installer
    assert "runascurrentuser" not in installer.casefold()


def test_pyinstaller_bundle_keeps_heavy_ai_models_optional() -> None:
    repository = Path(__file__).parents[1]
    specification = (repository / "installer" / "travelmovieai.spec").read_text(encoding="utf-8")

    assert '"travelmovieai/web/static"' in specification
    assert '"torch"' in specification
    assert '"transformers"' in specification
    assert "excludes=" in specification


def test_desktop_and_installer_are_explicit_optional_groups() -> None:
    repository = Path(__file__).parents[1]
    with (repository / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    optional = project["project"]["optional-dependencies"]
    scripts = project["project"]["scripts"]
    assert any(dependency.startswith("PySide6") for dependency in optional["desktop"])
    assert any(dependency.startswith("pyinstaller") for dependency in optional["installer"])
    assert scripts["travelmovieai-desktop"] == "travelmovieai.desktop:run"


def test_installer_build_script_uses_project_local_environment() -> None:
    repository = Path(__file__).parents[1]
    script = (repository / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8")

    assert ".cache\\installer-venv" in script
    assert "ISCC.exe" in script
    assert "$HOME" not in script
