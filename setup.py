import os

from setuptools import find_packages, setup

import kaggle_cassava

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))


def _load_requirements(fname='requirements.txt'):
    with open(os.path.join(_PATH_HERE, fname), encoding='utf-8') as fp:
        reqs = [rq.rstrip() for rq in fp.readlines()]
    reqs = [ln[:ln.index('#')] if '#' in ln else ln for ln in reqs]
    reqs = [ln for ln in reqs if ln]
    return reqs


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="kaggle-cassava",
    version=kaggle_cassava.__version__,
    description=kaggle_cassava.__docs__,
    author=kaggle_cassava.__author__,
    author_email=kaggle_cassava.__author_email__,
    url=kaggle_cassava.__homepage__,
    license=kaggle_cassava.__license__,
    packages=find_packages(exclude=['tests', 'tests/*']),
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=_load_requirements(),
    project_urls={
        "Source Code": kaggle_cassava.__homepage__,
    },
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        # Pick your license as you wish
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
)
