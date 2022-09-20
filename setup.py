from setuptools import find_packages, setup

setup(
    name="numenor",
    version="0.0.1",
    author="Mark Weiss",
    author_email="mark.s.weiss.ma@gmail.com",
    description=open("README.md").read(),
    url="http://github.com/marksweissma/numenor",
    packages=find_packages(),
    install_requires=["scikit-learn", "pandas", "attrs", "fire", "variants"],
    summary="ml workflow tooling",
    license="MIT",
    zip_safe=False,
)
