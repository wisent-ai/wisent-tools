from setuptools import setup, find_namespace_packages
setup(
    name="wisent-tools",
    version="0.2.0",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Operational scripts and benchmark-evaluation runners for the wisent package family",
    url="https://github.com/wisent-ai/wisent-tools",
    packages=find_namespace_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=[
        "wisent>=0.11.21",
        "wisent-evaluators>=0.1.0",
        "matplotlib>=3.0",
        # Pinned: the failure vocabulary must not move under this CLI without a
        # commit here. Bump deliberately.
        "wisent-errors @ git+https://github.com/wisent-ai/wisent-errors"
        "@e3014d2c900e499e171aeed8804da10bc7d93bf8#subdirectory=python",
    ],
    include_package_data=True,
    package_data={"wisent": ["scripts/*.sh"]},
    entry_points={"console_scripts": ["wisent-tools-onboarding=wisent.onboarding:main"]},
)
