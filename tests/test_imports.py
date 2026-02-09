import importlib


def test_src_packages_importable() -> None:
    package_names = [
        "src",
        "src.data",
        "src.features",
        "src.models",
        "src.reco",
        "src.pipelines",
        "src.serving",
    ]

    for package_name in package_names:
        assert importlib.import_module(package_name) is not None
