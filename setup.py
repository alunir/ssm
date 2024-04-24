from setuptools import setup

setup(
    name="ssm",
    version="1.1.1",
    install_requires=[
        "skorch",
        "pytorch_lightning",
        "einops",
        "jax",
    ],
)
