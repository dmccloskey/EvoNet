/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LINK_H
#define SMARTPEAK_LINK_H

#include <SmartPeak/ml/Node.h>

namespace SmartPeak
{

  /**
    @brief Directed Network Link
  */
  class Link
  {
public:
    Link(); ///< Default constructor
    Link(const int& id, SmartPeak::Node& source, SmartPeak::Node& sink, double& weight); ///< Explicit constructor  
    ~Link(); ///< Default destructor

    void setId(const double& id); ///< id setter
    double getId() const; ///< id getter

    void setSourceNode(const double& id); ///< id setter
    double getSourceNode() const; ///< id getter

    void setSinkNode(const double& id); ///< id setter
    double getSinkNode() const; ///< id getter

    void setWeight(const double& id); ///< id setter
    double getWeight() const; ///< id getter

private:
    int id_; ///< Link ID
    SmartPeak::Node source_node_; ///< Link source node
    SmartPeak::Node sink_node_; ///< Link sink node
    double weight_; ///< Link weight

  };
}

#endif //SMARTPEAK_LINK_H