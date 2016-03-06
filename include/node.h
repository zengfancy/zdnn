#ifndef __ZDNN_NODE__
#define __ZDNN_NODE__

namespace zdnn {

class Node {
  public:
    Node(int dim, int batch);
    Node();

    void Init(int dim, int batch);

    void set_output_error(int dim, int batch, double error) {
      out_error_[dim + batch * dimension_] = error;
    }
  public:
    double* in_value_;
    double* in_error_;
    double* out_value_;
    double* out_error_;

    int dimension_;
    int batch_;
};

}

#endif
