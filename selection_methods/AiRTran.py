import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class AiRTran(object):
    def __init__(self, args):
        self.args = args
    
    def score(self, features, labels):

        start_time = time.time()

        if "scale" in self.args.method:
            scaler = StandardScaler().fit(features)
            features = scaler.transform(features)


        if "balance" in self.args.method:

            pos_ids = np.where(labels==1)[0]
            pos_features = features[pos_ids]
            pos_labels = labels[pos_ids]
            neg_ids = np.where(labels==0)[0]
            neg_ids = random.sample(neg_ids.tolist(), pos_features.shape[0])
            neg_features = features[neg_ids]
            neg_labels = labels[neg_ids]

            train_features = np.concatenate([pos_features, neg_features], 0)
            train_labels = np.concatenate([pos_labels, neg_labels], 0)
        else:
            train_features, train_labels = features, labels

        train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
        model = LinearRegression().fit(train_features, train_labels)
        predict_logits = model.predict(features).reshape(-1, int(self.args.candidate_size))

        src_size = predict_logits.shape[0]
        labels = [0] * src_size
        
        score = 0

        for idx in range(src_size):
            pred = np.argsort(-predict_logits[idx]).tolist()
            for r, p in enumerate(pred):
                if p == labels[idx]:
                    score += 1 / (r + 1)
                    break
        
        score = np.round(score / src_size, 4)

        end_time = time.time()

        return score, end_time - start_time