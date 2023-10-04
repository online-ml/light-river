from river import anomaly
from river import datasets
from time import time

start = time()
hst = anomaly.HalfSpaceTrees(
    n_trees=50,
    height=6,
    window_size=1000,
)

for x, _ in datasets.CreditCard():
    hst.score_one(x)
    hst.learn_one(x)
print(f"Time: {time() - start:.2f}s")
