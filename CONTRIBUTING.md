## 2023-10-04

We test with:

- 50 trees
- Tree height of 6
- Window size of 1000

The Python baseline runs in **~60 seconds** using Python 3.11 on MacOS. It uses the classic left/right class-based implementation.

We coded a first array based implementation in Rust. It runs in **~6 seconds**. Each tree is a struct. Each struct contains one array for each node attribute. We wonder if we can do better by storing all attributes in a matrix.

ROC AUC appears roughly similar between the Python and Rust implementations. Note that we didn't activate min-max scaling in both cases.

## 2023-10-05

- Using `with_capacity` on each `Vec` in `HST`, as well as the list of HSTs, we gain 1 second. We are now at **~5 seconds**.
- We can't find a nice profiler. So for now we comment code and measure time.
- Storing all attributes in a single array, instead of one array per tree, makes us reach **~3 seconds**.
- We removed the CSV logic from the benchmark, which brings us under **~2.5 second**.
- Fixing some algorithmic issues actually brings us to **~5 seconds** :(
- We tried using rayon to parallelize over trees, but it didn't bring any improvement whatsoever. Maybe we used it wrong, but we believe its because our loop is too cheap to be worth the overhead of spawning threads -- or whatever it is rayon does.
- There is an opportunity to do the scoring and update logic in one fell swoop. This is because of the nature of online anomaly detection. This would bring us to **~2.5 seconds**. We are not sure if this is a good design choice though, so we may revisit this later.
