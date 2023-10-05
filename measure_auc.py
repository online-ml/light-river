import pandas as pd
from sklearn import metrics

scores = pd.read_csv('scores_py.csv', names=['score'])['score']
labels = pd.read_csv('creditcard.csv')['Class']
print(f"{metrics.roc_auc_score(labels, scores):.2%}")
