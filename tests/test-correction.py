# -*- coding: utf-8 -*-
# test-correction.py
# author: Antoine Passemiers

import numpy as np

from portia.correction import apply_correction


def test_corrections():
    scores = np.asarray([
        [4, 0.2, 1.1, 5.6],
        [0, 0.05, 6.3, 1.2],
        [2.3, 0.1, 0.9, 5.3]])

    corrected = apply_correction(scores, method='rcw')
    assert np.argmin(corrected) == 4
    assert np.argmax(corrected) == 6
    assert np.all(corrected >= 0)

    corrected = apply_correction(scores, method='asc')
    assert np.argmin(corrected) == 4
    assert np.argmax(corrected) == 6

    corrected = apply_correction(scores, method='apc')
    assert np.argmin(corrected) == 4
    assert np.argmax(corrected) == 6


def test_zeros():
    scores = np.zeros((100, 100))
    assert np.all(apply_correction(scores, method='apc') == 0)
    assert np.all(apply_correction(scores, method='asc') == 0)
    assert np.all(apply_correction(scores, method='rcw') == 0)
