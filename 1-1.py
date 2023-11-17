import math

import numpy


def softmax(inMatrix):
    m, n = numpy.shape(inMatrix)
    outMartix = numpy.mat(numpy.zeros((m, n)))
    soft_sum = 0
    for idx in range(0, n):
        outMartix[0, idx] = math.exp(inMatrix[0, idx])
        soft_sum += outMartix[0, idx]
    for idx in range(0, n):
        outMartix[0, idx] = outMartix[0, idx] / soft_sum
    return outMartix


a = numpy.array([[1, 2, 1, 2, 1, 1, 3]])
print(softmax(a))
