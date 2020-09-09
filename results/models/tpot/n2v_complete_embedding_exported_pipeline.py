import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.naive_bayes import GaussianNB
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9152098335727792
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    XGBClassifier(learning_rate=0.1, max_depth=5, min_child_weight=1, n_estimators=100, nthread=1, subsample=1.0)
)


exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)