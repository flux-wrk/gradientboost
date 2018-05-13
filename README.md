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

./gradientboost eval
  --data       	Model file for a classifier
  --target     	Target label
  --model 		Name of model file to test (if no model given, assuming evaluation of model trained in 'fit' subcommand)
```

Parameters explanation:
- `--train` - path to train dataset. Last column treated as categorial label for now.
- `--test` - path to test dataset in CSV format.
- `--tree-depth` or `--d` - depth of tree.
- `--learning-rate` or `--l` - learning rate. Make no effect now, but this will be changed soon.
- `--num-trees` or `--n` - count of trees in ensemble.

### Implementation details
This library (obviously) implements gradient boosting over decision trees. In current implementation we use oblivious decision trees with histograms for "fast" training.
Current code is tested with Higgs dataset.

### TODO:
- [x] Split implementation of decision tree and gradient boosting itself
- [ ] Implement custom loss functions
- [x] Test library with popular datasets
- [x] Introduce multithreaded training
- [x] Code quality and test coverage improvements
- [x] Performance comparison with industrial solutions

### Benchmark results

[benchmark results for Higgs dataset coming soon]
