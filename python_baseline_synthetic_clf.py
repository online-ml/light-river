from river.forest import AMFClassifier
from time import time
import pandas as pd
from river.tree.mondrian.mondrian_tree_nodes import MondrianLeafClassifier
from river.tree.mondrian.mondrian_tree_nodes import MondrianBranchClassifier

mf = AMFClassifier(
    n_estimators=1,
    use_aggregation=False,
)

df = pd.read_csv("/home/robotics/light-river/syntetic_dataset_v2.3.csv")
# df = pd.read_csv("/home/robotics/light-river/syntetic_dataset_paper.csv")
X = df[["feature_1", "feature_2"]]
y = df["label"].values

score_total = 0
start = time()
counter = 0

import numpy as np


def count_nodes(node):
    if isinstance(node, MondrianLeafClassifier):
        return 1
    elif isinstance(node, MondrianBranchClassifier):
        return 1 + np.sum([count_nodes(c) for c in node.children])


# for i, ((_, x), true_label) in enumerate(zip(X.iterrows(), y), 1):
for i, ((_, x), true_label) in enumerate(zip(X.iterrows(), y)):
    pred_proba = mf.predict_proba_one(x.to_dict())
    if pred_proba:
        predicted_label = max(pred_proba, key=pred_proba.get)
        if predicted_label == true_label:
            score_total += 1

        # print(
        #     f"{score_total} / {i} = {score_total/i}, nodes:",
        #     count_nodes(mf.data[0]._root),
        # )

    # print("=M=1 x:", list(x.to_dict().values()))
    mf.learn_one(x.to_dict(), true_label)

    # if counter > 10:
    #     raise ()
    counter += 1

print(f"Time: {time() - start:.2f}s")

# with open('preds_py.csv', 'w') as f:
#     for pred in preds:
#         f.write(f"{pred}\n")
