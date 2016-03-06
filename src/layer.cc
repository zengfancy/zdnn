#include "layer.h"

namespace zdnn {

void Layer::FeedForward(const Node* in, Node* out) {
  // activate(W * x + b)

  // m: o_dim_
  // n: batch_
  // k: i_dim_

  // copy b to out->in_value_ batch times
  for (int i=0; i<in->batch_; i++) {
    memcpy(out->in_value_ + i * o_dim_, b_, sizeof(double) * o_dim_);
  }

  int m = o_dim_;
  int n = in->batch_;
  int k = i_dim_;
  dgemm(w_, false,  // m * k
      in->out_value_, false, // k * n
      1.0, // alpha
//      b_, // m * 1
      out->in_value_, false, // m * n
      1.0, // beta
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
  double* identity = new double[n];
  dgemv(out->in_error_, false, // m * n
      identity, // n * 1
      learning_rate_, // alpha
      delta_b_, // m * 1
      1.0, // beta
      m,
      n);

  // calculate weight gradient
  // delta_w_i_r = Sigma_k(output_error_i_k * x_j_k), k[0, batch-1]
  dgemm(out->in_error_, false,  // m * n
      in->out_value_, true, // n * k, transpose
      learning_rate_, // alpha
      delta_w_, // m * k
      learning_rate_, // beta
      m,
      k,
      n);

  // Error backpropagation 
  // deltaX = Wt * deltaY
  dgemm(w_, true, // k * m, transpose
      out->in_error_, false, // m * n
      1.0, // alpha
      in->out_error_, false, // k * n
      1.0, // beta
      k,
      n,
      m);

}

void Layer::UpdateDeltaWb() {
  int len = i_dim_ * o_dim_;
  for (int i=0; i<len; i++) {
    w_[i] += delta_w_[i];
  }

  len = o_dim_;
  for (int i=0; i<len; i++) {
    b_[i] += delta_b_[i];
  }
}

}
