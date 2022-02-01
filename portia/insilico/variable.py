# -*- coding: utf-8 -*-
#
#  variable.py
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


class Variable:

    def __init__(self):
        self.parents = []
        self.bias = np.random.normal(0, 1)
        self.std = np.random.rand() * 2
        self.a = np.random.rand() * 3
        self.value = None

    def add_parent(self, var, coefficient):
        self.parents.append((var, coefficient))

    def sample(self):
        if self.value is None:
            value = 0
            for var, coefficient in self.parents:
                value += (1. / len(self.parents)) * coefficient * var.sample()
            value += np.random.normal(0, 1) * self.std
            self.value = self.a * np.exp(0.1 * value)
        return self.value

    def knock_out(self):
        self.value = 1e-5

    def reset(self):
        self.value = None
