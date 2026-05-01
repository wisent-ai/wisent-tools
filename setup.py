from setuptools import setup, find_namespace_packages
setup(
    name="wisent-tools",
    version="0.1.15",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Operational scripts and benchmark-evaluation runners for the wisent package family",
    url="https://github.com/wisent-ai/wisent-tools",
    packages=find_namespace_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=["wisent>=0.11.21", "wisent-evaluators>=0.1.0", "matplotlib>=3.0"],
    include_package_data=True,
    package_data={"wisent": ["scripts/*.sh"]},
)
