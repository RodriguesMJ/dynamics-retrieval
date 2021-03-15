# -*- coding: utf-8 -*-
import numpy
for fraction in numpy.arange(0.01, 0.5, 0.01):
    A = 2 * fraction * fraction - 2 * fraction + 1
    B = 2 * fraction * (1 - fraction)
    print A+B

