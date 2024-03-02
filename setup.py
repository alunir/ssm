from setuptools import setup

setup(
    name="s4",
    version="1.0.0",
    install_requires=[
        "skorch",
        "pytorch_lightning",
        "einops",
        "jax",
    ],
)
