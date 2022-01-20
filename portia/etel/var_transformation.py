# -*- coding: utf-8 -*-
#
#  var_transformation.py
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

from abc import ABCMeta, abstractmethod

import torch


class VariableTransformation(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, X):
        Y = self._forward(X)
        log_jac = self.log_jacobian(X)
        return Y, log_jac

    @abstractmethod
    def _forward(self, X):
        pass

    @abstractmethod
    def log_jacobian(self, X):
        pass
