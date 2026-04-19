from setuptools import find_packages, setup


setup(
    name="eurobot-2026-planning-poc",
    version="0.1.0",
    description="A lightweight Eurobot 2026 strategy POC simulator.",
    packages=find_packages(include=["poc", "poc.*"]),
    install_requires=[
        "matplotlib>=3.7",
        "numpy>=1.26",
        "PyYAML>=6.0",
    ],
)
