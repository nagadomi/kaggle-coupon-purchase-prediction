import sys
import os.path as path
import pickle
import pandas as pd
import numpy as np
from exceptions import NotImplementedError
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, datadir="./data"):
        self.train_coupon_vec = None
        self.test_coupon_vec = None
        self.train_coupon_df = None
        self.valid_coupon_df = None
        self.test_coupon_df = None
        self.users = None
        self.user_df = None
        self.datadir = datadir
    
    def save_pkl(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pkl(cls, filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def load(self, validation_timedelta=None):
        self.__load_coupons(validation_timedelta)
        self.__load_users()

    def dim(self):
        return len(self.users[0]["user"]) + (len(self.train_coupon_vec[0]) + 2 + 4) + len(self.train_coupon_vec[0])

    @staticmethod
    def __coupon_preproc(df):
        df["REDUCE_PRICE"] = df["CATALOG_PRICE"] - df["DISCOUNT_PRICE"]
        for key in ["DISCOUNT_PRICE", "CATALOG_PRICE", "REDUCE_PRICE"]:
            df[key + "_LOG"] = np.log(df[key] + 1.0).astype(np.float32)

        df["VALIDPERIOD_NA"] = np.array(pd.isnull(df["VALIDPERIOD"]), dtype=np.int32)
        df["DISPPERIOD_C"] = np.array(df["DISPPERIOD"].clip(0, 8), dtype=np.int32)
        df["PRICE_RATE"] = np.array(df.PRICE_RATE, dtype=np.float32)
        df["large_area_name"].fillna("NA", inplace=True)
        df["ken_name"].fillna("NA", inplace=True)
        df["small_area_name"].fillna("NA", inplace=True)
        df["LARGE_AREA_NAME"] = df["large_area_name"]
        df["PREF_NAME"] = df["large_area_name"] + ":" + df["ken_name"]
        df["SMALL_AREA_NAME"] = df["large_area_name"] + ":" + df["ken_name"] + ":" + df["small_area_name"]
        df["CATEGORY_NAME"] = df["CAPSULE_TEXT"] + df["GENRE_NAME"]

        usable_dates = ['USABLE_DATE_MON',
                        'USABLE_DATE_TUE',
                        'USABLE_DATE_WED',
                        'USABLE_DATE_THU',
                        'USABLE_DATE_FRI',
                        'USABLE_DATE_SAT',
                        'USABLE_DATE_SUN',
                        'USABLE_DATE_HOLIDAY',
                        'USABLE_DATE_BEFORE_HOLIDAY']        
        for key in usable_dates:
            df[key].fillna(0, inplace=True)
        df["USABLE_DATE_SUM"] = 0
        for key in usable_dates:
            df["USABLE_DATE_SUM"] += df[key]

        cols = df.columns.tolist()
        cols.remove("DISPFROM")
        cols.remove("DISPEND")
        for key in cols:
            df[key].fillna("NA", inplace=True)

    def __load_coupons(self, validation_timedelta):
        train_coupon_df = pd.read_csv(path.join(self.datadir, "coupon_list_train.csv"),
                                           parse_dates=["DISPFROM","DISPEND"])
        test_coupon_df = pd.read_csv(path.join(self.datadir, "coupon_list_test.csv"))

        train_coupon_df["DISPFROM"].fillna(pd.Timestamp("19000101"), inplace=True)
        train_coupon_df = train_coupon_df.sort(columns=["DISPFROM"]).reset_index(drop=True)

        if validation_timedelta:
            max_date = train_coupon_df["DISPFROM"].max()
            valid_start = max_date - validation_timedelta
            valid_coupon_df = train_coupon_df[(train_coupon_df["DISPFROM"] > valid_start)]
            train_coupon_df = train_coupon_df[~ (train_coupon_df["DISPFROM"] > valid_start)]
        else:
            valid_coupon_df = train_coupon_df[np.zeros(len(train_coupon_df), dtype=np.bool)].copy()

        # remove outlier data from the validation-set
        if len(valid_coupon_df) > 0:
            very_low_price = valid_coupon_df[valid_coupon_df.DISCOUNT_PRICE <= 100].COUPON_ID_hash
            very_long_time_display = valid_coupon_df[valid_coupon_df.DISPPERIOD > 20].COUPON_ID_hash
            valid_coupon_df = valid_coupon_df[~valid_coupon_df.COUPON_ID_hash.isin(very_long_time_display)]
            valid_coupon_df = valid_coupon_df[~valid_coupon_df.COUPON_ID_hash.isin(very_low_price)].reset_index(drop=True)

        # remove outlier data from the training-set
        very_long_time_display = train_coupon_df[train_coupon_df.DISPPERIOD > 20].COUPON_ID_hash
        train_coupon_df = train_coupon_df[~train_coupon_df.COUPON_ID_hash.isin(very_long_time_display)].reset_index(drop=True)

        # coupon features
        coupon_mapper = DataFrameMapper([
                ('CATEGORY_NAME', LabelBinarizer()),
                ('PRICE_RATE', None),
                ('CATALOG_PRICE_LOG', None),
                ('DISCOUNT_PRICE_LOG', None),
                ('REDUCE_PRICE_LOG', None),
                ('DISPPERIOD_C', LabelBinarizer()),
                ('VALIDPERIOD_NA', LabelBinarizer()),
                ('USABLE_DATE_SUM', None),
                ('LARGE_AREA_NAME', LabelBinarizer()),
                ('PREF_NAME', LabelBinarizer()),
                ('SMALL_AREA_NAME', LabelBinarizer()),
                ])
        config = {}
        self.__coupon_preproc(train_coupon_df)
        self.__coupon_preproc(valid_coupon_df)
        self.__coupon_preproc(test_coupon_df)
        
        coupon_mapper.fit(pd.concat([train_coupon_df, valid_coupon_df, test_coupon_df]))
        
        train_coupon_vec = coupon_mapper.transform(train_coupon_df.copy())
        if len(valid_coupon_df) > 0:
            valid_coupon_vec = coupon_mapper.transform(valid_coupon_df.copy())
        else:
            valid_coupon_vec = np.array([])
        test_coupon_vec = coupon_mapper.transform(test_coupon_df.copy())

        self.train_coupon_vec = train_coupon_vec
        self.valid_coupon_vec = valid_coupon_vec
        self.test_coupon_vec = test_coupon_vec
        self.train_coupon_df = train_coupon_df
        self.valid_coupon_df = valid_coupon_df
        self.test_coupon_df = test_coupon_df

    def __load_users(self):
        user_df = pd.read_csv(path.join(self.datadir,"user_list.csv"))
        details = pd.read_csv(path.join(self.datadir, "coupon_detail_train.csv"),
                                          parse_dates=["I_DATE"])
        details = details.sort(columns=["I_DATE"]).reset_index(drop=True)

        # user features
        user_mapper = DataFrameMapper([
                ('SEX_ID', LabelBinarizer()),
                ('PREF_NAME', LabelBinarizer()),
                ('AGE', None),
                ])
        user_df["PREF_NAME"].fillna("NA", inplace=True)
        user_vec = user_mapper.fit_transform(user_df.copy())
        
        users = []
        self.train_coupon_df["ROW_ID"] = pd.Series(self.train_coupon_df.index.tolist())
        self.valid_coupon_df["ROW_ID"] = pd.Series(self.valid_coupon_df.index.tolist())
        for i, user in user_df.iterrows():
            coupons = details[details.USER_ID_hash.isin([user["USER_ID_hash"]])]
            train_coupon_data = pd.merge(coupons[["COUPON_ID_hash","ITEM_COUNT","I_DATE"]],
                                         self.train_coupon_df,
                                         on="COUPON_ID_hash", how='inner',
                                         suffixes=["_x",""], copy=False)
            train_coupon_data = train_coupon_data.sort(columns=["I_DATE"])
            row_ids = train_coupon_data.ROW_ID.unique().tolist()

            valid_coupon_data = pd.merge(coupons[["COUPON_ID_hash","ITEM_COUNT","I_DATE"]],
                                         self.valid_coupon_df, on="COUPON_ID_hash",
                                         how='inner', suffixes=["_x",""], copy=False)
            valid_coupon_data = valid_coupon_data.sort(columns=["I_DATE"])
            valid_row_ids = valid_coupon_data.ROW_ID.unique().tolist()

            users.append({"user": user_vec[i],
                          "coupon_ids": row_ids,
                          "valid_coupon_ids": valid_row_ids})
            if i % 100 == 0:
                print "load users: %d/%d\r" % (i, len(user_df)),
                sys.stdout.flush()

        print "\n",

        self.users = users
        self.user_df = user_df

    def __maxmin_columns(self, coupon_ids):
        return self.train_coupon_df.ix[
            coupon_ids, ("CATALOG_PRICE","DISCOUNT_PRICE")
            ].as_matrix().astype(np.float32)

    def __purchase_history_features(self, user_coupon_vec, maxmin_columns, filter_idx=None):
        sum_vec = np.zeros(2, dtype=np.float32)
        maxmin_vec = np.zeros((4), dtype=np.float32)
        mean_coupon_vec = np.zeros(len(self.train_coupon_vec[0]), dtype=np.float32)
        
        if filter_idx is not None:
            if len(user_coupon_vec[filter_idx]) > 0:
                mean_coupon_vec[:] = user_coupon_vec[filter_idx].mean(0)
                sum_vec[0] = filter_idx.sum()
                sum_vec[1] = np.log(sum_vec[0] + 1.0)
                max_val = maxmin_columns[filter_idx].max(0)
                min_val = maxmin_columns[filter_idx].min(0)
                maxmin_vec[0] = max_val[0]
                maxmin_vec[1] = min_val[0]
                maxmin_vec[2] = max_val[1]
                maxmin_vec[3] = min_val[1]
        else:
            if len(user_coupon_vec) > 0:
                mean_coupon_vec = user_coupon_vec.mean(0)
                sum_vec[0] = len(user_coupon_vec)
                sum_vec[1] = np.log(sum_vec[0] + 1.0)
                max_val = maxmin_columns.max(0)
                min_val = maxmin_columns.min(0)
                maxmin_vec[0] = max_val[0]
                maxmin_vec[1] = min_val[0]
                maxmin_vec[2] = max_val[1]
                maxmin_vec[3] = min_val[1]

        return np.hstack((mean_coupon_vec, sum_vec, maxmin_vec))
    
    COUPON_DISP_NEAR = 400
    COUPON_DISP_NEAR_MIN = 10
    def gen_train_data(self, num_nega=2, verbose=True):
        x = []
        y = []
        c = 0
        for user in self.users:
            coupon_ids = np.array(user["coupon_ids"], dtype=np.int32)
            user_coupons = self.train_coupon_vec[coupon_ids]
            maxmin_columns = self.__maxmin_columns(coupon_ids)
            for i in xrange(len(user_coupons)):
                target_coupon_vec = user_coupons[i]
                rid = coupon_ids[i]
                nega_list = range(max(0, rid - self.COUPON_DISP_NEAR), rid)
                if len(nega_list) < self.COUPON_DISP_NEAR_MIN:
                    continue

                filter_idx = np.ones(user_coupons.shape[0], dtype=np.bool)
                
                # exclude coupons that was purchased after the target coupon
                filter_idx[i:] = False
                # exclude the target coupon (and remove duplicate)
                filter_idx[coupon_ids == coupon_ids[i]] = False
                
                hist_feat = self.__purchase_history_features(user_coupons,
                                                             maxmin_columns,
                                                             filter_idx)
                # feature vector (user_feature + purchase_history_feature + coupon_feature)
                purchased_feat = np.hstack((user["user"], hist_feat, target_coupon_vec))
                x.append(purchased_feat)
                y.append([1]) # posi

                # select random unpurchased coupons
                for j in xrange(num_nega):
                    found = False
                    for _ in xrange(10):
                        unpurchased_idx = np.random.choice(nega_list, 1)[0]
                        if unpurchased_idx not in user["coupon_ids"]:
                            found = True
                            break
                    if found:
                        unpurchased_feat = np.hstack((user["user"],
                                                      hist_feat,
                                                      self.train_coupon_vec[unpurchased_idx]))
                        x.append(unpurchased_feat)
                        y.append([0]) # nega

            c += 1
            if verbose and c % 100 == 0:
                print ("train data .. %d/%d\r" % (c, len(self.users))),
                sys.stdout.flush()
        if verbose:
            print "\n",

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        return x, y
    
    def gen_valid_data(self, num_nega=2):
        raise NotImplementedError("gen_valid_data")
    
    def gen_train_data_pairwise(self, num_nega=10):
        raise NotImplementedError("gen_train_data_pairwise")
    
    def gen_valid_data_pairwise(self, num_nega=10):
        x0 = []
        x1 = []
        
        for i, user in enumerate(self.users):
            coupon_ids = np.array(user["coupon_ids"], dtype=np.int32)
            user_coupons = self.train_coupon_vec[coupon_ids]
            maxmin_columns = self.__maxmin_columns(coupon_ids)
            hist_feat = self.__purchase_history_features(user_coupons, maxmin_columns)
            valid_ids = user["valid_coupon_ids"]
            
            for coupon_id in valid_ids:
                purchased_feat = np.hstack((user["user"],
                                            hist_feat,
                                            self.valid_coupon_vec[coupon_id]))
                for j in xrange(num_nega):
                    while True:
                        unpurchased_idx = np.random.randint(0, len(self.valid_coupon_vec))
                        if unpurchased_idx not in valid_ids:
                            break

                    unpurchased_feat = np.hstack((user["user"],
                                                  hist_feat,
                                                  self.valid_coupon_vec[unpurchased_idx]))
                    # pairwise 
                    x0.append(purchased_feat)
                    x1.append(unpurchased_feat)

        x0 = np.array(x0, dtype=np.float32)
        x1 = np.array(x1, dtype=np.float32)

        return x0, x1

    def each_valid(self, callback, verbose=False):
        for k, user in enumerate(self.users):
            user_id = k
            user_coupons = self.train_coupon_vec[user["coupon_ids"]]
            maxmin_columns = self.__maxmin_columns(user["coupon_ids"])
            hist_feat = self.__purchase_history_features(user_coupons, maxmin_columns)
            feats = np.empty((len(self.valid_coupon_vec),
                              len(user["user"]) + len(hist_feat) + len(self.valid_coupon_vec[0])),
                             dtype=np.float32)
            coupon_ids = []
            for i in xrange(len(self.valid_coupon_vec)):
                coupon_id = i
                feats[i][:] = np.hstack((user["user"], hist_feat, self.valid_coupon_vec[i]))
                coupon_ids.append(coupon_id)

            callback({"user_id": user_id, "coupon_ids": coupon_ids, "coupon_feats": feats})
            if verbose and (k % 100 == 0):
                print ("each valid .. %d/%d\r" % (k, len(self.users))),
                sys.stdout.flush()
        if verbose:
            print "\n",
    
    def each_test(self, callback, verbose=True):
        for k, user in enumerate(self.users):
            user_id = self.user_df["USER_ID_hash"][k]
            user_coupons = self.train_coupon_vec[user["coupon_ids"]]
            maxmin_columns = self.__maxmin_columns(user["coupon_ids"])
            hist_feat = self.__purchase_history_features(user_coupons, maxmin_columns)
            feats = np.empty((len(self.test_coupon_vec),
                              len(user["user"]) + len(hist_feat) + len(self.test_coupon_vec[0])),
                             dtype=np.float32)
            coupon_ids = []
            for i in xrange(len(self.test_coupon_vec)):
                coupon_id = self.test_coupon_df["COUPON_ID_hash"][i]
                feats[i][:] = np.hstack((user["user"], hist_feat, self.test_coupon_vec[i]))
                coupon_ids.append(coupon_id)

            callback({"user_id": user_id, "coupon_ids": coupon_ids, "coupon_feats": feats})
            if verbose and k % 100 == 0:
                print ("each test .. %d/%d\r" % (k, len(self.users))),
                sys.stdout.flush()
        if verbose:
            print "\n",
