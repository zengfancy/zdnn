#ifndef __ZDNN_SOFTMAX__
#define __ZDNN_SOFTMAX__

#include "activate.h"

namespace zdnn {

class Softmax: public Activation {
  public:
    virtual void Activate(Node* node) {
      for (int col=0; col < node->batch_; col++) {
        // sum = sigma(exp(-x))
        double sum = 0;

        double* temp_value = new double[node->dimension_];
        for (int row=0; row < node->dimension_; row++) {
          int index = col * node->dimension_ + row;
          temp_value[row] = exp(-node->in_value_[index]);
          sum += temp_value[row];
        }

        // exp(-x) / sum
        for (int row=0; row < node->dimension_; row++) {
          int index = col * node->dimension_ + row;
          node->out_value_[index] = temp_value[row] / sum;
        }

        delete [] temp_value;
      }
    }
    virtual void BackProp(Node* node) {
      for (int col=0; col < node->batch_; col++) {
        for (int row=0; row < node->dimension_; row++) {
        }
      }
    }

  protected:
    virtual void act_impl(Node* node, int index) {
    }
    virtual void backprop_impl(Node* node, int index) {
    }
};

}
#endif
