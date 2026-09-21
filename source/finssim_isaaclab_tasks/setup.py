"""Install the FinsSim Isaac Lab task extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
metadata = toml.load(ROOT / "config" / "extension.toml")["package"]

setup(
    name="finssim-isaaclab-tasks",
    version=metadata["version"],
    description=metadata["description"],
    author=metadata["author"],
    maintainer=metadata["maintainer"],
    url=metadata["repository"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "finssim_isaaclab_tasks": [
            "assets/warpauv/data/warpauv/**",
            "tasks/**/agents/*.yaml",
        ]
    },
    python_requires=">=3.12,<3.13",
    install_requires=["psutil"],
    zip_safe=False,
)
