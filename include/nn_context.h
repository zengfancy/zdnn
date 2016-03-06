#ifndef __ZDNN_CONTEXT__
#define __ZDNN_CONTEXT__

namespace zdnn {

class NNContext {
  public:
    NNContext(Layer* layers, int layer_size);

  public:
    Node* output_nodes_;
    Node* input_node_;
    int node_num_; // node_num == layer_size
};

}

#endif
