# from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from catboost import Pool, CatBoostRegressor


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = CatBoostRegressor(
                        iterations=15000,
                        learning_rate=0.01,
                        l2_leaf_reg= 1.2036222698707393,
                        depth = 6,
                        bagging_temperature = 10.089437110816705,
                        #one_hot_max_size=20.3249397,
                        rsm =0.970876240566679,
                        subsample = 0.9482282383276974,
                        od_type='Iter',
                        od_wait=20
                        )


    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
