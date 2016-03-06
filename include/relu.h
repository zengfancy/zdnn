#ifndef __ZDNN_RELU__
#define __ZDNN_RELU__

#include "activate.h"

namespace zdnn {

class Relu : public Activation {
  protected:
    virtual void act_impl(Node* node, int index) {
      if (node->in_value_[index] > 0) {
        node->out_value_[index] = node->in_value_[index];
      } else {
        node->out_value_[index] = 0;
      }
    }
    virtual void backprop_impl(Node* node, int index) {
      if (node->in_value_[index] > 0) {
        node->in_error_[index] = node->out_error_[index];
      } else {
        node->in_error_[index] = 0;
      }
    }
};


}

#endif
