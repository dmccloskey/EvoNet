/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LAYER_H
#define SMARTPEAK_LAYER_H

#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Link.h>

#include <vector>

namespace SmartPeak
{

  /**
    @brief An array of links that represents a layer in the execution graph.

    Foward propogation:
      1. f(source * weights) = sinks
      2. calculate the derivatives for back propogation

    Back propogation:
      1. source * weights . derivatives = sinks
      2. adjust the weights
  */
  class Layer
  {
public:
    Layer(); ///< Default constructor
    Layer(const int& id, const std::vector<Link>& links); ///< Explicit constructor  
    ~Layer(); ///< Default destructor

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setLinks(const std::vector<Link>& links); ///< links setter
    std::vector<Link> getLinks() const; ///< links getter

protected:
    int id_; ///< Layer ID
    std::vector<Link> links_; ///< Layer links
    std::vector<int> source_layer_dims_; ///< source layer dimensions (e.g., samples x source nodes)
    std::vector<int> weight_layer_dims_; ///< weight layer dimensions (e.g., source nodes x sink nodes)
    std::vector<int> sink_layer_dims_; ///< sink layer dimensions (e.g., samples x sink nodes)
    std::vector<int> derivative_layer_dims_; ///< derivative layer dimensions (same as sink_layer_dims)

  };
}

#endif //SMARTPEAK_LAYER_H