"""Install the package using standard setuptools metadata and requirement files."""

from pathlib import Path
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements(filename):
    """Read pinned dependencies without importing either numerical runtime."""
    return [
        line.strip()
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_version():
    source = (ROOT / "src/one2n_inference/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', source, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("The package version is missing from __init__.py")
    return match.group(1)


if __name__ == "__main__":
    setup(
        name="one2n-inference",
        version=read_version(),
        description="Autoregressive One2N atmospheric inference with a NeuralGCM baseline",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        python_requires=">=3.12,<3.13",
        license="MIT",
        license_files=["LICENSE"],
        package_dir={"": "src"},
        packages=find_packages(where=str(ROOT / "src")),
        install_requires=read_requirements("requirements.txt"),
        extras_require={"neuralgcm": read_requirements("requirements-neuralgcm.txt")},
        entry_points={"console_scripts": ["one2n=one2n_inference.cli:main"]},
    )
