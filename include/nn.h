#ifndef __ZDNN_NN__
#define __ZDNN_NN__

namespace zdnn {

class Network {
  public:
    /**
     * @param i_value: column matrix, dim * batch
     * @param o_value: vector, len:batch
     */
     void FeedForward(const double* i_value, 
        int dim, int batch,
        double* o_value);
     void Train(const double* i_value,
        int dim, int batch,
        const double* o_value);

  protected:
     void _feedForward(const double* i_value, 
        int dim, int batch,
        double* o_value, NNContext* context);
    Layer* layers_;
    int layer_size_;
};

}

#endif
