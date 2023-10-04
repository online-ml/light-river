## 2023-10-05

We test with:

- 50 trees
- Tree height of 6
- Window size of 1000

The Python baseline runs in ~60 seconds. It uses the classic left/right class implementation.

We coded a first array based implementation in Rust. It runs in ~6 seconds. Each tree is a struct. Each struct contains one array for each node attribute. We wonder if we can do better by storing all attributes in a matrix.
