from setuptools import setup, find_packages
setup(
    name="wisent-tools",
    version="0.1.2",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Operational scripts and benchmark-evaluation runners for the wisent package family",
    url="https://github.com/wisent-ai/wisent-tools",
    packages=find_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=["wisent>=0.10.0", "wisent-evaluators>=0.1.0"],
    include_package_data=True,
    package_data={"wisent": ["scripts/*.sh"]},
)
