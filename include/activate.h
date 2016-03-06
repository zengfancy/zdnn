#ifndef __ZDNN_ACTIVATE__
#define __ZDNN_ACTIVATE__

#include "node.h"

namespace zdnn {

class Activation {
  public:
    virtual void Activate(Node* node) {
      for (int row=0; row < node->dimension_; row++) {
        for (int col=0; col < node->batch_; col++) {
          int index = col * node->dimension_ + row;
          act_impl(node, index);
        }
      }
    }
    virtual void BackProp(Node* node) {
      for (int row=0; row < node->dimension_; row++) {
        for (int col=0; col < node->batch_; col++) {
          int index = col * node->dimension_ + row;
          backprop_impl(node, index);
        }
      }
    }

  protected:
    virtual void act_impl(Node* node, int index) = 0;
    virtual void backprop_impl(Node* node, int index) = 0;
};

}

#endif
