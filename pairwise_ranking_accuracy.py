import numpy
from chainer import function

class PairwiseRankingAccuracy(function.Function):
    def check_type_forward(self, in_types):
        pass
    
    def forward_cpu(self, inputs):
        y0, y1 = inputs
        return numpy.array(float((y0 > y1).sum()) / y0.size, numpy.float32),
