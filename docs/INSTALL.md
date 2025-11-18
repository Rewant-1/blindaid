# BlindAid Installation Guide

Follow these steps to get a clean developer workstation ready for BlindAid. The project ships as a
standard Python package, so anything that can install `pyproject` based projects (pip, uv, pipx)
will work.

## 1. Prerequisites

- Python 3.9 or newer
- Git
- A working C/C++ toolchain (needed for `dlib` and some Paddle components)
- ffmpeg/espeak/portaudio on the target OS for audio output (see README for OS specific notes)

## 2. Create an Isolated Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel
```

> **Tip:** Feel free to swap the commands above for uv (`uv venv` / `uv pip install`) or conda if
> your team standardises on a different environment manager.

## 3. Install Runtime Dependencies

The root `requirements.txt` now bootstraps the editable package with every runtime feature
(including captions + depth):

```bash
pip install -r requirements.txt
```

To add developer tooling (ruff, black, pytest, mypy):

```bash
pip install -r requirements-dev.txt
```

For Raspberry Pi targets you can use the lighter `requirements-pi.txt`, which installs the package
without the heavyweight optional models by default. Advanced extras can be layered on top when
there is enough RAM and swap available.

## 4. Provision Models and Face Embeddings

```text
resources/
  models/
    object_blind_aide.pt
    yolov9t-face-lindevs.pt
  known_faces/
    <person_name>/photo01.jpg
```

- Store YOLO weights inside `resources/models/` (paths are configurable via `blindaid.core.config`).
- Copy cropped face images into `resources/known_faces/<person_name>/`.
- Keep large binaries out of git – `.gitignore` already excludes these folders except for `.gitkeep`.

## 5. Smoke Test

```bash
python -m blindaid --help
python -m blindaid                 # Integrated controller
python -m blindaid --mode ocr      # Single mode smoke tests
python -m pytest                   # Optional regression tests
```

If the CLI opens without stack traces the install is complete. Head over to `README.md` for mode
specific hotkeys and to `docs/RASPBERRY_PI.md` for ARM optimisation notes.

## 6. Helper Scripts

- `./setup.sh [--dev]` automates the steps above on Linux/macOS
- `./setup.ps1 [-Dev]` does the same on Windows PowerShell
