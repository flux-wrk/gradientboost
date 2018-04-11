## GradientBoost
Gradient boosting library for YSDA LSML course project. 

### Building
Clone this repository, then build it using cmake:
```bash
git clone https://github.com/flux-wrk/gradientboost.git
cd gradientboost
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

### Running
Sample command line:
```bash
./gradientboost --train data/training.csv --test data/test.csv --n 10 --d 4 --l 0.2
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
- Add scripts for downloading test datasets
- Split implementation of decision tree and gradient boosting itself
- Test library with popular datasets - 
- Introduce multithreaded training
- Code quality and test coverage improvements
- Performance comparison with industrial solutions
