import tomllib
from pathlib import Path

from travelmovieai.desktop import APP_MUTEX_NAME


def test_windows_installer_is_per_user_and_launches_desktop_shell() -> None:
    repository = Path(__file__).parents[1]
    installer = (repository / "installer" / "TravelMovieAI.iss").read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in installer
    assert "AppMutex=" in installer
    assert f"AppMutex={APP_MUTEX_NAME}" in installer
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
    assert ").FullName" in script
    assert "& $isccPath" in script
    assert "$iscc.Source" not in script
    assert "['project']['version']" in script
    assert "sys.argv[1]" in script
    assert "without producing an installer" in script
    assert "TravelMovieAI-$Version-setup.exe" in script
    assert "Get-ChildItem -LiteralPath $installerOutput -Filter '*.exe'" not in script
    assert "Get-FileHash" in script
    assert "signtool.exe" in script
    assert "$HOME" not in script


def test_one_click_launcher_keeps_heavy_ai_optional() -> None:
    repository = Path(__file__).parents[1]
    launcher = (repository / "scripts" / "run_web.bat").read_text(encoding="utf-8")
    setup = (repository / "scripts" / "setup_windows.bat").read_text(encoding="utf-8")

    assert "--base-only --non-interactive" in launcher
    assert "import torch" not in launcher
    assert 'set "INSTALL_SPEC=.[video]"' in setup
    assert "if not defined NONINTERACTIVE pause" in setup
