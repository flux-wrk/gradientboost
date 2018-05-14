## GradientBoost
Gradient boosting library for YSDA LSML course project. 

### Building
Clone this repository, then build it using cmake:
```bash
git clone https://github.com/flux-wrk/gradientboost.git
cd gradientboost
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 gradientboost
```

### Running
Program currently have two modes - fit and eval.
Sample command line:
```bash
./gradientboost fit
  --data 		Dataset file to train on
  --target 		Target label (name of column in csv file)
  --trees 		Number of trees in ensemble (default - 16)
  --depth 		Depth of each tree (default - 4)
  --model 		Name of saved model file
  --lr        Learning rate of boosting process

./gradientboost eval
  --data      Model file for a classifier
  --target    Target label
  --model     Name of model file to test (if no model given, assuming evaluation of model trained in 'fit' subcommand)
  
./gradientboost predict
  --data      Model file for a classifier
  --model     Name of model file to test
  --target    Target label
  --output    Path to file with predictions
```
You can combine training and evaluation in one run. You also can use `--nthreads` option to control concurrency level (uses all cpu cores by default).

### Implementation details
This library (obviously) implements gradient boosting over decision trees. In current implementation we use oblivious decision trees with histograms for "fast" training.
Current code is tested with Higgs dataset.

### TODO:
- [x] Split implementation of decision tree and gradient boosting itself
- [ ] Implement custom loss functions
- [x] Test library with popular datasets
- [x] Introduce multithreaded training
- [x] Code quality improvements
- [x] Performance comparison with industrial solutions

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

Our code (multithreaded, 8 logical cores):

| Trees | Depth | Time elapsed | Train MSE | Test MSE |
|-------|-------|--------------|-----------|----------|
| 32    | 4     | 8            | 0.139818  | 0.101234 |
| 32    | 8     | 22           | 0.112873  | 0.112387 |
| 32    | 12    | 41           | 0.071294  | 0.152384 |
| 32    | 16    | 89           | 0.002887  | 0.183245 |

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

TODO: CatBoost benchmark
