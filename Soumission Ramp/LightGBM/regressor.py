# from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from lightgbm import LGBMRegressor


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = LGBMRegressor(
            boosting_type='gbdt',
            n_estimators = 15000,
            num_leaves = 59,
            max_depth = 10,
            reg_alpha = 1.0883950410427778,
            reg_lambda = 7.853043072405197,
            min_child_samples = 14,
            learning_rate = 0.03,
            subsample_freq = 10,
            bagging_seed = 46
        )

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
    
#{'target': -0.4306169659171446, 'params': {'bagging_seed': 46.03524328560176, #'max_depth': 10.196446324575525, 'min_child_samples': 14.224148285112454, 'num_leaves': #58.57392334378522, 'reg_alpha': 1.0883950410427778, 'reg_lambda': 7.853043072405197, #'subsample_freq': 9.555169531251382}}