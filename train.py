import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from chainer import Variable
import argparse
import sys
from mlp3 import MLP3
from dataset import Dataset
from pairwise_ranking_accuracy import PairwiseRankingAccuracy

NEGA_WEIGHT = 2
N_EPOCH = 120
BATCH_SIZE = 128

def train():
    parser = argparse.ArgumentParser(description='nagadomi-coupon-purchase-prediction-solution')
    parser.add_argument('--seed', '-s', default=71, type=int,
                        help='Random seed')
    parser.add_argument('--validation', '-v', action="store_true",
                        help='Validation mode')
    args = parser.parse_args()
    model_name = "mlp"

    if args.validation:
        dataset = Dataset.load_pkl("data/valid_28.pkl")
        model_name = model_name + "_valid"
    else:
        dataset = Dataset.load_pkl("data/all_data.pkl")
    
    np.random.seed(args.seed)
    
    model = MLP3({"input": dataset.dim(),
                  "lr": 0.01,
                  "h1": 512, "h2": 32,
                  "dropout1": 0.5,
                  "dropout2": 0.1,
                  })
    scaler = StandardScaler()

    # estimate mean,std
    x, y = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
    scaler.fit(x)
    
    if args.validation:
        x0_test, x1_test = dataset.gen_valid_data_pairwise(num_nega=20)
        x0_test = scaler.transform(x0_test)
        x1_test = scaler.transform(x1_test)

    # learning loop
    for epoch in xrange(1, N_EPOCH+1):
        print('**** epoch {}/{}'.format(epoch, N_EPOCH))
        if epoch == 100:
            model.learning_rate_decay(0.5)
        
        # resampling the training dataset
        x, y = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
        x = scaler.transform(x)

        # update
        model.train(x, y, batchsize=BATCH_SIZE, verbose=True)
        
        # evaluate
        if args.validation:
            acc = pairwise_ranking_accuracy(model, x0_test, x1_test)
            print("valid pairwise ranking accuracy: {}".format(float(acc.data)))
            if epoch % 10 == 0:
                eval_map(model, scaler, dataset, k=10)

        if epoch % 10 == 0:
            # save model
            with open("models/{}_{}_epoch_{}.pkl".format(model_name, args.seed, epoch), "wb") as f:
                pickle.dump([model, scaler], f)

    with open("models/{}_{}.pkl".format(model_name, args.seed), "wb") as f:
        pickle.dump([model, scaler], f)

# MAP@K
def eval_map(model, scaler, dataset, k=10):
    sum_map = [0.0]
    c = [0]
    def callback(rec):
        purchased_ids = dataset.users[rec["user_id"]]["valid_coupon_ids"]
        if len(purchased_ids) > 0:
            pred = model.predict(scaler.transform(rec["coupon_feats"]))
            results = zip(pred, rec["coupon_ids"])
            results = sorted(results, key=lambda score: -score[0])
            mapk = 0.0
            correct = 0.0
            for j in xrange(min(k, len(results))):
                if (results[j][1] in purchased_ids):
                    correct += 1.0
                    mapk += (correct / (j + 1.0))
            mapk = mapk / min(k, len(purchased_ids))
            sum_map[0] += mapk
        
        if c[0] % 100 == 0:
            print("validate .. %d/%d %f\r" % (c[0], len(dataset.users), sum_map[0] / (c[0] + 1))),
            sys.stdout.flush()
        c[0] += 1
    
    dataset.each_valid(callback)

    print("valid MAP@{}: {}\n".format(k, sum_map[0] / len(dataset.users)))

# accuracy of: f(x0) > f(x1)
def pairwise_ranking_accuracy(model, x0, x1):
    y0 = model.forward_raw(Variable(x0), train=False)
    y1 = model.forward_raw(Variable(x1), train=False)
    
    return PairwiseRankingAccuracy()(y0, y1)

train()
