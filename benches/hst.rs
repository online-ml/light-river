use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use light_river::anomaly::half_space_tree::HalfSpaceTree;

fn creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("creation");

    let features: Vec<String> = vec![
        String::from("V1"),
        String::from("V2"),
        String::from("V3"),
        String::from("V4"),
        String::from("V5"),
        String::from("V6"),
        String::from("V7"),
        String::from("V8"),
        String::from("V9"),
        String::from("V10"),
        String::from("V11"),
        String::from("V12"),
        String::from("V13"),
        String::from("V14"),
        String::from("V15"),
        String::from("V16"),
        String::from("V17"),
        String::from("V18"),
        String::from("V19"),
        String::from("V20"),
        String::from("V21"),
        String::from("V22"),
        String::from("V23"),
        String::from("V24"),
        String::from("V25"),
        String::from("V26"),
        String::from("V27"),
        String::from("V28"),
        String::from("V29"),
        String::from("V30"),
    ];

    for height in [2, 6, 10, 14].iter() {
        for n_trees in [3, 30, 300].iter() {
            let input = (*height, *n_trees);
            // Calculate the throughput based on the provided formula
            let throughput = ((2u32.pow(*height) - 1) * *n_trees) as u64;
            group.throughput(Throughput::Elements(throughput));
            group.bench_with_input(
                format!("height={}-n_trees={}", height, n_trees),
                &input,
                |b, &input| {
                    b.iter(|| HalfSpaceTree::new(0, input.1, input.0, Some(features.clone())));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, creation);
criterion_main!(benches);
