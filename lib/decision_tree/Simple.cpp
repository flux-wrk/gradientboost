#include "Simple.h"
#include "Tree.h"

#include <memory>

namespace NGradientBoost {

void SimpleDecisionTreeClassifier::Fit(
    const Dataset& /*dataset*/,
    const std::vector<Label>& /*labels*/
) {
    /*
    for (int split = 0; split < leaf_count_; ++split) {

        const auto best_split = FindBestSplit();
        if (best_split.GetGain() < 0) {
            break;
        }
        tree.Split(best_split);
    }
    */
}

std::vector<Label> SimpleDecisionTreeClassifier::Predict(const Dataset& /*dataset*/) const {
    return {};
}

} // namespace NGradientBoost

