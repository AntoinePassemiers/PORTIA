# -*- coding: utf-8 -*-
#
#  local_convergence.py
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


class LocalConvergence:

    def __init__(self, n_steps_without_improvements, max_n_iter=1000, tau=1e-5):
        self.n_steps_without_improvements = n_steps_without_improvements
        self.max_n_iter = max_n_iter
        self.tau = tau
        self.iterations = 0
        self.nwi = 0
        self.best_loss = np.nan_to_num(np.inf)

    def step(self, loss):
        gain = (self.best_loss - loss) / np.abs(self.best_loss)
        if gain > self.tau:
            self.nwi = 0
        else:
            self.nwi += 1
        if loss < self.best_loss:
            self.best_loss = loss
        self.iterations += 1

    def __call__(self):
        if self.iterations >= self.max_n_iter:
            return True
        else:
            return self.nwi >= self.n_steps_without_improvements
