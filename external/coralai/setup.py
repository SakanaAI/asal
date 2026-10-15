from setuptools import setup, find_packages

setup(
    name='coralai',
    packages=find_packages(),
    version='0.0.1',
    description='Emergent Ecosystems of Evolved Neural Cellular Automata',
    author='Aidan Arnaud Barbieux',
    author_email='aidanbx@gmail.com',
    url='https://github.com/aidanbx/coralai',
    keywords=['pytorch', 'evolutionary-algorithms', 'taichi', 'emergent-behavior',
              'artificial-life', 'neural-cellular-automata'],
    classifiers=[],
    python_requires='>=3.9',
    install_requires=[
        'torch',
        'taichi>=1.6.0',
        'neat-python==0.92',
        'numpy',
        'scipy',
        'click',
        'dask',
        'noise',
    ],
)