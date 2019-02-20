import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Average CV score on the training set was:0.7622888546486963
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=LGBMClassifier(learning_rate=0.013093533273536108, max_depth=2, n_estimators=417, random_state=42))
    ),
    LGBMClassifier(learning_rate=0.002015337685941733, max_depth=3, n_estimators=1265, random_state=42)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
