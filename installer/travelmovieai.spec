# PyInstaller specification for the optional Windows desktop shell.

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

repository = Path(SPECPATH).parent.resolve()
data_files = [
    (str(repository / "src" / "travelmovieai" / "web" / "static"), "travelmovieai/web/static"),
    (str(repository / "configs"), "configs"),
]
for asset_directory in (repository / "assets" / "music", repository / "assets" / "fonts"):
    if asset_directory.is_dir():
        data_files.append((str(asset_directory), f"assets/{asset_directory.name}"))

analysis = Analysis(
    [str(repository / "src" / "travelmovieai" / "desktop.py")],
    pathex=[str(repository / "src")],
    binaries=[],
    datas=data_files,
    hiddenimports=[
        *collect_submodules("uvicorn"),
        "travelmovieai.web.app",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "transformers", "sentence_transformers", "faiss"],
    noarchive=False,
)
python_archive = PYZ(analysis.pure)
executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="TravelMovieAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TravelMovieAI",
)
