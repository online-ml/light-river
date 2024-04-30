from sklearn.datasets import make_classification
import pandas as pd

# Generate synthetic dataset
X, y = make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)

# Create DataFrame
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['label'] = y

print(df.head())
df.to_csv('syntetic_dataset_int.csv', index=False)
