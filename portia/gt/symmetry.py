# -*- coding: utf-8 -*-
#
#  symmetry.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import numpy as np

from portia.gt.grn import GRN


def matrix_symmetry(A):
    if isinstance(A, GRN):
        tf_idx = A.tf_idx
        A = A.asarray()
        A = A[tf_idx, :][:, tf_idx]

    A = np.asarray(A)
    Aa = 0.5 * (A - A.T)
    As = 0.5 * (A + A.T)
    mask = ~np.logical_or(np.isnan(Aa), np.isnan(As))
    norm_aa = np.linalg.norm(Aa[mask])
    norm_as = np.linalg.norm(As[mask])
    if norm_as + norm_aa == 0:
        return 1
    else:
        return (norm_as - norm_aa) / (norm_as + norm_aa)
