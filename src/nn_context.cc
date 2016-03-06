#include "nn_context.h"
#include "layer.h"

namespace zdnn {


NNContext::NNContext(Layer* layers, int layer_size, int batch) : 
  node_num_(layer_size) {
  input_node_ = new Node();
  input_node_->Init(layers[0].i_dim_, batch);
  output_nodes_ = new Node[node_num_];
  for (int i = 0; i < node_num_; i++) {
    output_nodes_[i].Init(layers[i].o_dim_, batch);
  }
}

NNContext::~NNContext() {
  delete input_node_;
  delete [] output_nodes_;

}

}
