# Copyright 2018, (Rolinek, Martius)
# MIT License Agreement
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Keras implementation of L4 stepsize adaptation scheme proposed in (Rolinek, Martius, 2018)"""

from __future__ import absolute_import
from .L4_core import L4Mom as core_L4Mom, L4Adam as core_L4Adam
from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces


class L4Mom(Optimizer):
    """Wrapper class for native TensorFlow L4Mom optimizer.
    """

    def __init__(self, **kwargs):
        self.optimizer = core_L4Mom(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.optimizer.compute_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        opt_update = self.optimizer.apply_gradients(grads, loss, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    @property
    def weights(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def from_config(self, config):
        raise NotImplementedError


class L4Adam(Optimizer):
    """Wrapper class for native TensorFlow L4Adam optimizers.
    """

    def __init__(self, **kwargs):
        self.optimizer = core_L4Adam(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.optimizer.compute_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        opt_update = self.optimizer.apply_gradients(grads, loss, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    @property
    def weights(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def from_config(self, config):
        raise NotImplementedError
