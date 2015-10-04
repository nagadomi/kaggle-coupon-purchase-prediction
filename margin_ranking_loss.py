import numpy
from chainer import function
from chainer.utils import type_check

# This loss function is not currently used.

class MarginRankingLoss(function.Function):
    def check_type_forward(self, in_types):
        pass
    
    def forward_cpu(self, inputs):
        y0, y1 = inputs
        self.loss = numpy.maximum(-((y0 - y1) - 1.0), 0.0)
        loss = numpy.sum(self.loss)
        return numpy.array(loss / self.loss.size, numpy.float32),

    def backward_cpu(self, inputs, gy):
        coeff = gy[0] / self.loss.size
        gy0 = coeff * self.loss
        return -gy0, gy0

def margin_ranking_loss(y0, y1):
    return MarginRankingLoss()(y0, y1)
