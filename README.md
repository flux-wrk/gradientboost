## GradientBoost

Gradient boosting library for YSDA LSML course project.

### Requirements

- **CMake 4.0+**
- **C++17 compiler** (tested with AppleClang 21, GCC)
- **Git** (for FetchContent to download CLI11)

### Building

Clone this repository, then build it using CMake:

```bash
git clone https://github.com/flux-wrk/gradientboost.git
cd gradientboost
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j8
```

> **Note:** CLI11 v2.6.2 is automatically fetched via CMake FetchContent on first build.

### Running

The program supports three subcommands: `fit`, `eval`, and `predict`.

```bash
# Train a model
./gradientboost fit \
  --data <dataset.csv> \
  --target <label_column> \
  --trees 16 \
  --depth 4 \
  --model <model_file> \
  --lr 0.1

# Evaluate model
./gradientboost eval \
  --data <dataset.csv> \
  --target <label_column> \
  --model <model_file>

# Predict on new data
./gradientboost predict \
  --data <dataset.csv> \
  --model <model_file> \
  --target <label_column> \
  --output <predictions.csv>
```

You can combine training and evaluation in a single run (fit then eval without `--model`).

### Implementation details

This library implements gradient boosting over oblivious decision trees with histogram-based binning for fast training. Key features:

- **CMake 4.0** with modern target-based configuration
- **Native C++ concurrency** (`std::mutex`, `std::sort`) — no external threading library required
- **CLI11 v2.6.2** fetched automatically via CMake FetchContent
- **Oblivious decision trees** with configurable depth
- **Histogram-based feature binning** (16 bins per feature) for efficient split finding
- Model serialization/deserialization

### TODO

- [ ] Implement custom loss functions
- [ ] Add regression/classification mode selection
- [ ] Add early stopping support

### Benchmark results

Benchmark was done on [following](https://support.apple.com/kb/sp719?locale=en_US) hardware specs:
- 2.8 - 4.0 GHz quad-core Intel Core i7 CPU, 6MB L3 cache.
- 16GB of 1600MHz DDR3L memory.

#### Higgs Boson ([kaggle](https://www.kaggle.com/c/higgs-boson/data))

Our code:

| Trees | Depth | Time elapsed | Train MSE | Test MSE |
|-------|-------|--------------|-----------|----------|
| 32    | 4     | 22           | 0.139818  | 0.101234 |
| 32    | 8     | 61           | 0.112873  | 0.112387 |
| 32    | 12    | 109          | 0.071294  | 0.152384 |
| 32    | 16    | 220          | 0.002887  | 0.183245 |

LightGBM:

| Trees | Depth | Time elapsed | Train MSE | Test MSE |
|-------|-------|--------------|-----------|----------|
| 32    | 4     | 0.28         | 0.117     | 0.107443 |
| 32    | 8     | 1            | 0.104     | 0.110843 |
| 32    | 12    | 7            | 0.092     | 0.176716 |
| 32    | 16    | 50           | 0.045     | 0.229672 |

XGBoost:

| Trees | Depth | Time elapsed | Train MSE | Test MSE |
|-------|-------|--------------|-----------|----------|
| 32    | 4     | 6            | 0.120089  | 0.121364 |
| 32    | 8     | 21           | 0.082569  | 0.199075 |
| 32    | 12    | 34           | 0.039824  | 0.237417 |
| 32    | 16    | 66           | 0.000378  | 0.254394 |
