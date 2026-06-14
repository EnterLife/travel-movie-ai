"""Native directory selection for the local web interface."""

from pathlib import Path


def select_directory(
    initial_path: Path | None,
    title: str,
    must_exist: bool,
) -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    initial_directory = _existing_directory(initial_path)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=str(initial_directory) if initial_directory else None,
            mustexist=must_exist,
        )
    finally:
        root.destroy()
    return Path(selected).resolve() if selected else None


def _existing_directory(path: Path | None) -> Path | None:
    candidate = path.expanduser() if path else None
    while candidate is not None and not candidate.is_dir():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    return candidate
