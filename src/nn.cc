#include "nn.h"

namespace zdnn {


void Network::FeedForward(const double* i_value, 
    int dim, int batch,
    double* o_value) {
  NNContext* context = create_nn_context(layers_, layer_size_, batch);
  _feedForward(i_value, dim, batch, o_value, context);
  destroy_nn_context(context);
}

void Network::_feedForward(const double* i_value, 
    int dim, int batch,
    double* o_value, NNContext* context) {
  memcpy(context->input_node_->out_value_,
      i_value, dim * batch * sizeof(double));

  Node* i_node = context_input_node_;

  for (int i=0; i < layer_size_; i++) {
    Node* o_node = context->output_nodes_[i];
    layers_[i]->FeedForward(i_node, o_node);
    i_node = o_node;
  }

  // i_node is the last node now
  memcpy(o_value, i_node->out_value_, sizeof(double) * batch * i_node->dimension_);
}


void Network::Train(const double* i_value,
    int dim, int batch,
    const double* o_value) {
  NNContext* context = create_nn_context(layers_, layer_size_, batch);

  // output node
  Node* output_node = context->output_nodes_[context->node_num_ - 1];
  int output_dim = output_node->dimension_;
  double* predict_value = new double[batch * output_dim];
  _feedForward(i_value, dim, batch, predict_value, context);

  // init the output node error
  memcpy(output_node->out_error_, predict_value, sizeof(double) * batch * output_dim);
  /*for (int i=0; i < batch; i++) {
    output_node->set_output_error(0, i, predict_value[i]);
  }*/

  for (i = context->node_num_ - 1; i > 0; i--) {
    Layer* layer = layer_[i];
    Node* i_node = context->output_nodes_[i - 1];
    Node* o_node = context->output_nodes_[i];
    layer->BackProp(i_node, o_node);
    layer->UpdateDeltaWb();
  }

  layer_[0]->BackProp(context->input_node_, context->output_nodes_[0]);
  layer_[0]->UpdateDeltaWb();

  delete [] predict_value;
  destroy_nn_context(context);
}

}
