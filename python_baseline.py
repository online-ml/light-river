from river import anomaly
from river import datasets
from time import time

scores = []
start = time()
hst = anomaly.HalfSpaceTrees(
    n_trees=50,
    height=6,
    window_size=1000,
)

for x, _ in datasets.CreditCard():
    score = hst.score_one(x)
    scores.append(score)
    hst.learn_one(x)
print(f"Time: {time() - start:.2f}s")

with open('scores_py.csv', 'w') as f:
    for score in scores:
        f.write(f"{score}\n")
