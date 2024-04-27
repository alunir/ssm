from setuptools import setup

setup(
    name="ssm",
    version="1.2.0",
    install_requires=[
        "skorch",
        "pytorch_lightning",
        "einops",
        "jax",
        "flax",
        "equinox",
    ],
)
