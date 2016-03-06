#ifndef __ZDNN_ACTIVATE__
#define __ZDNN_ACTIVATE__

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


#endif
