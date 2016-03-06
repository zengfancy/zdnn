#ifndef __ZDNN_LAYER__
#define __ZDNN_LAYER__

#include "activate.h"
#include "node.h"

namespace zdnn {

class Layer {
  public:
    void FeedForward(const Node* in, Node* out);
    void BackProp(Node* in, const Node* out);
    void UpdateDeltaWb();
  public:
    double* w_;
    double* b_;
    int i_dim_;
    int o_dim_;

    double* delta_w_;
    double* delta_b_;

    Activation* activate_;
    double learning_rate_;
};

}
#endif
