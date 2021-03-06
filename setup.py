# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

import re

from setuptools import setup


packages = [
    'portia',
    'portia.etel',
    'portia.gt',
    'portia.insilico',
    'portia.optim'
]

with open('README.md', 'r') as f:
    long_description = f.read()


def get_property(prop, project):
    # Solution proposed in:
    # https://stackoverflow.com/questions/17791481/creating-a-version-attribute-for-python-packages-without-getting-into-troubl
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)


setup(
    name='portia-grn',
    version=get_property('__version__', 'portia'),
    author='Antoine Passemiers',
    author_email='antoine.passemiers@kuleuven.be',
    packages=packages,
    url='https://github.com/AntoinePassemiers/PORTIA',
    description='PORTIA: Fast and Accurate Inference of Gene Regulatory Networks through Robust Precision Matrix Estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.5',
        'scipy',
        'scikit-learn'
    ],
)
