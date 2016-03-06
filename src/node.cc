#include "node.h"


namespace zdnn {

Node::Node(int dim, int batch) {
  Init(dim, batch);
}

Node::Node() {
  delete [] in_value_;
  delete [] out_value_;
  delete [] in_error_;
  delete [] out_error_;
}

void Node::Init(int dim, int batch) {
  dimension_ = dim;
  batch_ = batch;

  int len = dim * batch;
  in_value_ = new double[len];
  in_error_ = new double[len];
  out_value_ = new double[len];
  out_error_ = new double[len];
}

}
