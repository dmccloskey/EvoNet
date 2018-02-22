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
    Link(const int& id, const SmartPeak::Node& source_node,
      const SmartPeak::Node& sink_node); ///< Explicit constructor  
    ~Link(); ///< Default destructor

    void setId(const double& id); ///< id setter
    double getId() const; ///< id getter

    void setSourceNode(const SmartPeak::Node& source_node); ///< source_node setter
    SmartPeak::Node getSourceNode() const; ///< source_node getter

    void setSinkNode(const SmartPeak::Node& sink_node); ///< sink_node setter
    SmartPeak::Node getSinkNode() const; ///< sink_node getter

    void setWeight(const double& weight); ///< weight setter
    double getWeight() const; ///< weight getter

private:
    int id_; ///< Link ID
    SmartPeak::Node source_node_; ///< Link source node
    SmartPeak::Node sink_node_; ///< Link sink node
    double weight_ = 1.0; ///< Link weight

  };
}

#endif //SMARTPEAK_LINK_H