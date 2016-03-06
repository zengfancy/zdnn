#include "layer.h"

namespace zdnn {

void Layer::FeedForward(const Node* in, Node* out) {
  // activate(W * x + b)

  // m: o_dim_
  // n: batch_
  // k: i_dim_
  int m = o_dim_;
  int n = in->batch_;
  int k = i_dim_;
  dgemm_b(w_,  // m * k
      in->out_value_, // k * n
      b_, // m * 1
      out->in_value_, // m * n
      m,
      n,
      k);
  
  activate_->Activate(out);
}


void Layer::BackProp(Node* in, const Node* out) {
  // Error backpropagation
  // deltaY = d(y) * deltaY
  activate_->BackProp(out);

  // m: o_dim_
  // n: batch_
  // k: i_dim_
  int m = o_dim_;
  int n = in->batch_;
  int k = i_dim_;

  // calculate delta_w, delta_b
  // delta_b_i = Sigma_j(out_error_i_j) j:[0, batch-1]
  dgemv(out->in_error_, // m * n
      delta_b_, // m * 1
      m, 
      n, 
      learning_rate_);

  // calculate weight gradient
  // delta_w_i_r = Sigma_k(output_error_i_k * x_j_k), k[0, batch-1]
  dgemm(out->in_error_, false,  // m * n
      in->out_value_, true, // n * k, transpose
      delta_w_, // m * k
      m,
      k,
      n,
      learning_rate_);

  // Error backpropagation 
  // deltaX = Wt * deltaY
  dgemm(w_, true, // k * m, transpose
      out->in_error_, false, // m * n
      in->out_error_, // k * n
      k,
      n,
      m);

}

void Layer::UpdateDeltaWb() {
  int len = i_dim_ * o_dim_;
  for (int i=0; i<len; i+=) {
    w_[i] += delta_w_[i];
  }

  len = o_dim_;
  for (int i=0; i<len; i+=) {
    b_[i] += delta_b_[i];
  }
}

}
