<h1>🦀 LightRiver • fast online machine learning</h1>

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
  <img src="https://user-images.githubusercontent.com/8095957/202878607-9fa71045-6379-436e-9da9-41209f8b39c2.png" width="25%" align="right" />
</div>

LightRiver is an online machine learning library written in Rust. It is meant to be used in high-throughput environments, as well as TinyML systems.

This library is complementary to [River](https://github.com/online-ml/river/). The latter provides a wide array of online methods, but is not ideal when it comes to performance. The idea is to take the algorithms that work best in River, and implement them in a way that is more performant. As such, LightRiver is not meant to be a general purpose library. It is meant to be a fast online machine learning library that provides a few algorithms that are known to work well in online settings. This is a akin to the way [scikit-learn](https://scikit-learn.org/) and [LightGBM](https://lightgbm.readthedocs.io/en/stable/) are complementary to each other.

## 🚀 Performance

## 📝 License

LightRiver is free and open-source software licensed under the [3-clause BSD license](LICENSE).
