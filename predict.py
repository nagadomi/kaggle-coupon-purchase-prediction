import numpy as np
import pickle
import mlp3
from mlp3 import MLP3
from sklearn.preprocessing import StandardScaler
from dataset import Dataset

def predict():
    preds = []
    models = []
    scalers = []
    dataset = Dataset.load_pkl("data/all_data.pkl")
    for i in [71, 72, 73, 74]: # load 4 models
        print("load {}".format(i))
        with open("models/mlp_{}.pkl".format(i), "rb") as f:
            m = pickle.load(f)
            model, scaler = m[0], m[1]
            models.append(model)
            scalers.append(scaler)
    
    def callback(rec):
        feats = rec["coupon_feats"]
        pred = np.zeros(len(feats), dtype=np.float32)
        
        for i, m in enumerate(models):
            pred += m.predict(scalers[i].transform(feats))
        pred /= len(models)
        
        scores = zip(pred, rec["coupon_ids"])
        scores = sorted(scores, key = lambda score: -score[0])
        coupon_ids = " ".join(map(lambda score: str(score[1]), scores[0:10]))
        preds.append([rec["user_id"], coupon_ids])
    
    dataset.each_test(callback)
    preds = sorted(preds, key=lambda rec: rec[0])
    fp = open("submission_mlp.csv", "w")
    fp.write("USER_ID_hash,PURCHASED_COUPONS\n")
    for pred in preds:
        fp.write("%s,%s\n" % (pred[0], pred[1]))
    fp.close()

predict()
