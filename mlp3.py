from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F
import numpy as np

# 3-layer MLP
class MLP3(FunctionSet):
    def __init__(self, params):
        self.params = params
        self.optimizer = optimizers.MomentumSGD(lr=params["lr"])
        super(MLP3, self).__init__(
            l1=F.Linear(params["input"], params["h1"]),
            l2=F.Linear(params["h1"], params["h2"]),
            l3=F.Linear(params["h2"], 1))
        self.optimizer.setup(self)

    def forward_raw(self, x, train=True):
        h = x
        h = F.dropout(F.relu(self.l1(h)), ratio=self.params["dropout1"], train=train)
        h = F.dropout(F.relu(self.l2(h)), ratio=self.params["dropout2"], train=train)
        y = self.l3(h)
        return y

    def predict(self, x_data, y_data=None):
        x = Variable(x_data)
        y = self.forward_raw(x, train=False)
        if y_data is None:
            return F.sigmoid(y).data.reshape(x_data.shape[0])
        else:
            return F.sigmoid_cross_entropy(y, Variable(y_data)).data
    
    def forward(self, x_data, y_data):
        x, t = Variable(x_data), Variable(y_data)
        y = self.forward_raw(x, train=True)
        return F.sigmoid_cross_entropy(y, t)

    def learning_rate_decay(self, rate):
        self.optimizer.lr *= rate
        
    def train(self, x, y, evalset=None, batchsize=128, verbose=True):
        perm = np.random.permutation(x.shape[0])
        sum_loss = 0.0
        c = 0
        for i in xrange(0, x.shape[0], batchsize):
            x_batch = x[perm[i:i+batchsize]]
            y_batch = y[perm[i:i+batchsize]]
            self.optimizer.zero_grads()
            loss = self.forward(x_batch, y_batch)
            loss.backward()
            self.optimizer.update()
            loss = float(loss.data)
            sum_loss += loss * batchsize
            c += x_batch.shape[0]
        if verbose:
            print("train logloss: {}".format(sum_loss/c))
            if evalset is not None:
                x, y = evalset[0], evalset[1]
                perm = np.random.permutation(x.shape[0])
                for i in xrange(0, x.shape[0], batchsize):
                    x_batch = x[perm[i:i+batchsize]]
                    y_batch = y[perm[i:i+batchsize]]
                    loss = self.predict(x_batch, y_batch)
                    sum_loss += loss * batchsize
                    c += x_batch.shape[0]
                print("valid logloss: {}".format(sum_loss/c))
