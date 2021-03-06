from setuptools import find_packages, setup

setup(name="tamamo",
    version="0.1",
    description="pytorch evolution (minimum version) + best foxgirl waifu",
    author="Andros Tjandra",
    author_email='andros.tjandra@gmail.com',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="BSD",
    url="",
    packages=find_packages(),
    install_requires=['numpy','scipy', 'torch', 'pytest']);
