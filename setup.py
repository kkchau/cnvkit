#!/usr/bin/env python
"""Copy number variation toolkit for high-throughput sequencing."""

from setuptools import setup

setup(
    extras_require={
        "hmm": [
            "pomegranate==1.0.0",
        ]
    }
)
