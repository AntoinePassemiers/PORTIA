# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

from setuptools import setup


packages = [
    'portia',
    'portia.optim'
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='portia-grn',
    version='0.0.3',
    author='Antoine Passemiers',
    author_email='antoine.passemiers@kuleuven.be',
    packages=packages,
    package_data={},
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
