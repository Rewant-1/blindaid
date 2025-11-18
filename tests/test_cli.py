from blindaid import app
from blindaid.core import config


def test_parse_arguments_defaults():
    args = app.parse_arguments([])
    assert args.start_mode == "scene"
    assert args.camera == config.DEFAULT_CAMERA_INDEX
    assert args.audio is True


def test_parse_arguments_custom_mode():
    args = app.parse_arguments(["--start-mode", "reading", "--camera", "2", "--no-audio"])
    assert args.start_mode == "reading"
    assert args.camera == 2
    assert args.audio is False
