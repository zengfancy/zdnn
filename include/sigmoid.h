#ifndef __ZDNN_SIGMOID__
#define __ZDNN_SIGMOID__

#include "activate.h"

namespace zdnn {

class Sigmoid : public Activation {
  public:
  protected:
    virtual void act_impl(Node* node, int index) {
      node->out_value_[index] = sigmoid(node->in_value_[index]);
    }
    virtual void backprop_impl(Node* node, int index) {
      node->in_error_[index] = d_sigmoid(node->in_value_[index]) * node->out_error_[index];
    }
};

}
#endif
