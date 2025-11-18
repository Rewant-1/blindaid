from blindaid.core import config


def test_resource_directories_exist():
    assert config.RESOURCES_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.KNOWN_FACES_DIR.exists()


def test_default_model_paths_point_inside_resources():
    assert str(config.OBJECT_DETECTION_MODEL).startswith(str(config.MODELS_DIR))
    assert str(config.FACE_RECOGNITION_MODEL).startswith(str(config.MODELS_DIR))
