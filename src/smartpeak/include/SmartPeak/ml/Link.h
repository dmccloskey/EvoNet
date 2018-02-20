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
    Link(const int& id, Node& source, Node& sink, double& weight); ///< Explicit constructor  
    ~Link(); ///< Default destructor

    void setId(const double& id); ///< id setter
    double getId() const; ///< id getter


private:
    int id_; ///< Link ID
    Node source_; ///< Link Type
    Node sink_; ///< Link Output
    double weight_; ///< Link Error

  };
}

#endif //SMARTPEAK_LINK_H