<h1>🦀 LightRiver • fast and simple online machine learning</h1>

<p>

<!-- Tests -->
<!-- <a href="https://github.com/online-ml/beaver/actions/workflows/unit-tests.yml">
<img src="https://github.com/online-ml/beaver/actions/workflows/unit-tests.yml/badge.svg" alt="tests">
</a> -->

<!-- Code quality -->
<!-- <a href="https://github.com/online-ml/beaver/actions/workflows/code-quality.yml">
<img src="https://github.com/online-ml/beaver/actions/workflows/code-quality.yml/badge.svg" alt="code_quality">
</a> -->

<!-- License -->
<a href="https://opensource.org/licenses/BSD-3-Clause">
<img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">
</a>

</p>

[![Discord](https://dcbadge.vercel.app/api/server/qNmrKEZMAn)](https://discord.gg/qNmrKEZMAn)

<div align="center" >
  <img src="https://github.com/online-ml/light-river/assets/8095957/fc8ea218-62f9-4643-b25d-f9265ef962f8" width="25%" align="right" />
</div>

Reed is an online machine learning library written in Rust. It is meant to be used in high-throughput environments, as well as TinyML systems.

This library is complementary to [River](https://github.com/online-ml/river/). The latter provides a wide array of online methods, but is not ideal when it comes to performance. The idea is to take the algorithms that work best in River, and implement them in a way that is more performant. As such, LightRiver is not meant to be a general purpose library. It is meant to be a fast online machine learning library that provides a few algorithms that are known to work well in online settings. This is a akin to the way [scikit-learn](https://scikit-learn.org/) and [LightGBM](https://lightgbm.readthedocs.io/en/stable/) are complementary to each other.

## 🧑‍💻 Usage

### 🚨 Anomaly detection

```sh
cargo run --release --example credit_card
```

### 📈 Regression

🏗️ We plan to implement Aggregated Mondrian Forests.

### 📊 Classification

🏗️ We plan to implement Aggregated Mondrian Forests.

### 🛒 Recsys

🏗️ [Vowpal Wabbit](https://vowpalwabbit.org/) is very good at recsys via contextual bandits. We don't plan to compete with it. Eventually we want to research a tree-based contextual bandit.

## 🚀 Performance

TODO: add a `benches` directory

## 📝 License

LightRiver is free and open-source software licensed under the [3-clause BSD license](LICENSE).
