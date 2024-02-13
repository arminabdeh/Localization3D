import setuptools
from setuptools import setup
# requirements = [
#     "numpy",
#     "pytorch",
#     "click",
#     "deprecated",
#     "gitpython>=3.1",
#     "h5py",
#     "importlib_resources",
#     "matplotlib",
#     "pandas",
#     "pytest",
#     "pyyaml",
#     "requests",
#     "scipy",
#     "seaborn==0.10",
#     "scikit-image",
#     "scikit-learn",
#     "tensorboard",
#     "tifffile",
#     "tqdm", "decode",
# ]
requirements = []
setup(
    name='luenn',
    version='0.10.02',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    license='UB',
    author='Armin Abdehkakha',
    author_email='arminabd@buffalo.edu',
    description='luenn is a python package for the analysis of super resolution microscopy data, localization uncertainty estimation, and neural network training.',
    long_description=open('README.md').read(),
    url='',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
)