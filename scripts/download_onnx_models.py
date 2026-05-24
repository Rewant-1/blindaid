"""Download ONNX models for BlindAid research branch.

Usage:
    python scripts/download_onnx_models.py

Phase 1: MiDaS-Small (~24MB)
(More models added in later phases)
"""
import sys
import urllib.request
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent.parent / "resources" / "models"

MODELS = {
    "midas_small.onnx": {
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "size_mb": 24,
        "description": "MiDaS-Small — depth estimation (Phase 1)",
    },
}


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Download a file with progress."""
    if dest.exists():
        size = dest.stat().st_size / 1024 / 1024
        print(f"  [OK] {dest.name} already exists ({size:.1f} MB)")
        return True

    print(f"  [..] Downloading {description or dest.name}...")
    print(f"       URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(dest))
        size = dest.stat().st_size / 1024 / 1024
        print(f"  [OK] Downloaded {dest.name} ({size:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("BlindAid — ONNX Model Downloader")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nModel directory: {MODELS_DIR.resolve()}\n")

    success = 0
    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if download_file(info["url"], dest, info["description"]):
            success += 1
        print()

    print(f"Result: {success}/{len(MODELS)} models ready")

    if success < len(MODELS):
        print("Some downloads failed. Check internet connection.")
        sys.exit(1)
    else:
        print("All models ready!")


if __name__ == "__main__":
    main()
