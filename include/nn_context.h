#ifndef __ZDNN_CONTEXT__
#define __ZDNN_CONTEXT__

#include "node.h"

namespace zdnn {

class Layer;

class NNContext {
  public:
    NNContext(Layer* layers, int layer_size, int batch);
    virtual ~NNContext();

  public:
    Node* output_nodes_;
    Node* input_node_;
    int node_num_; // node_num == layer_size
};

}

#endif
