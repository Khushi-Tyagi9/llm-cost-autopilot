"""
Decides whether a given request should go through verification,
based on the configured sampling rate. This is what lets you taper
verification down from 100% once the classifier proves reliable,
per the brief's cost note.
"""
import random


def should_verify(sample_rate: float) -> bool:
    return random.random() < sample_rate