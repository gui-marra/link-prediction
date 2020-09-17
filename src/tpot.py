from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def tpot (X_train, y_train, X_test = None, y_test = None,
          export_file = '../results/models/tpot/exported_pipeline.py', n_jobs = 1):
    
    if 'node' and 'target' in X_train.columns:
        X_train = X_train.drop(columns = ['node', 'target'])
    if 'node' and 'target' in X_test.columns:
        X_test = X_test.drop(columns = ['node', 'target'])

    tpot = TPOTClassifier(generations = 5, population_size = 40, cv=3, verbosity=2, scoring = 'f1', n_jobs=6)

    tpot.fit(X_train, y_train)
    tpot.export(export_file)
    print(tpot.score(X_test, y_test))