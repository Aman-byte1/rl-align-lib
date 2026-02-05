from setuptools import setup, find_packages

setup(
    name="rl_align",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "tqdm",
        "numpy",
        "bitsandbytes",
    ],
    author="Manus",
    description="A comprehensive RL alignment library",
)
