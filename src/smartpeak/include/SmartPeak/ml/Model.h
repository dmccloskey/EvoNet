/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Layer.h>

#include <vector>

namespace SmartPeak
{

  /**
    @brief Directed Network Model
  */
  class Model
  {
public:
    Model(); ///< Default constructor
    Model(const int& id); ///< Explicit constructor  
    ~Model(); ///< Default destructor
 
    /**
      @brief A forward propogation step. Returns a vector of links where
        all sink output values are unknown (i.e. inactive),
        but all source node output values are known (i.e. active).

      If multiple vectors of links satisfy the above
        criteria, only the first vector of links will
        be returned.  All others will be returned
        on subsequent calls.

      @param[in] x_I Input value

      @returns layer vector of links
    */ 
    SmartPeak::Layer getNextInactiveLayer() const;
 
    /**
      @brief A back propogation step.  Returns a vector of links where
        all sink error values are unknown (i.e. active),
        but all source node error values are known (i.e. inactive).

      If multiple vectors of links satisfy the above
        criteria, only the first vector of links will
        be returned.  All others will be returned
        on subsequent calls.

      @param[in] x_I Input value

      @returns layer vector of links
    */ 
    SmartPeak::Layer getNextUncorrectedLayer() const;

    void setId(const double& id); ///< id setter
    double getId() const; ///< id getter

    void setError(const double& error); ///< error setter
    double getError() const; ///< error getter
 
    /**
      @brief Add new links to the model.

      @param[in] links Links to add to the model
    */ 
    void addLinks(const std::vector<Link>& links);

private:
    int id_; ///< Model ID
    std::vector<Link> links_; ///< Model links
    double error_; ///< Model error

  };
}

#endif //SMARTPEAK_MODEL_H