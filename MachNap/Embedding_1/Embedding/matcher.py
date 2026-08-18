import pickle
import numpy as np


class Matcher:

    def __init__(self, database_path, threshold=12.0):

        with open(database_path, "rb") as f:
            self.database = pickle.load(f)
        self.threshold = threshold
    def predict(self, feature):

        best_label = "Unknown"
        best_distance = np.inf
        
        for cls_name, feats in self.database.items():

            # feats có shape (N,1280)

            d = np.linalg.norm(feats - feature, axis=1)

            d = np.min(d)

            if d < best_distance:

                best_distance = d
                best_label = cls_name

        if best_distance > self.threshold:

            return "Unknown", best_distance

        return best_label, best_distance