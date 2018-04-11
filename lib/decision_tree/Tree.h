#pragma once

namespace NGradientBoost {

class Tree {
public:
    explicit Tree(int leaf_count)
        : leaf_count_(leaf_count)
    { }

    int GetLeafCount() {
        return leaf_count_;
    }

private:
    int leaf_count_;
};

}
