import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HidKim_SurvPP",
    version="0.0.1",
    author="Hideaki Kim",
    author_email="dedeccokim@gmail.com",
    description="Survival permanental process implemented in Tensorflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HidKim/SurvPP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
