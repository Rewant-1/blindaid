"""`python -m blindaid` entry point."""
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("PADDLEOCR_VERBOSE", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

from blindaid.app import main as app_main


def main() -> int:
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
