/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODEL_H
#define SMARTPEAK_MODEL_H

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>
#include <SmartPeak/ml/Layer.h>

#include <vector>
#include <map>

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

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setError(const double& error); ///< error setter
    double getError() const; ///< error getter
 
    /**
      @brief Add new links to the model.

      @param[in] links Links to add to the model
    */ 
    void addLinks(const std::vector<Link>& links);
 
    /**
      @brief Remove existing links from the model.

      @param[in] Link_ids Links to remove from the model
    */ 
    void removeLinks(const std::vector<int>& link_ids);
 
    /**
      @brief Add new nodes to the model.

      @param[in] nodes Nodes to add to the model
    */ 
    void addNodes(const std::vector<Node>& Nodes);
 
    /**
      @brief Remove existing nodes from the model.

      @param[in] node_ids Nodes to remove from the model
    */ 
    void removeNodes(const std::vector<int>& node_ids);
 
    /**
      @brief Removes nodes from the model that no longer
        have an associated link.
    */ 
    void pruneNodes();
 
    /**
      @brief Removes nodes from the model that no longer
        have an associated link.
    */ 
    void pruneLinks();

private:
    int id_; ///< Model ID
    std::map<Link> links_; ///< Model links
    std::map<Node> nodes_; ///< Model nodes
    double error_; ///< Model error

  };
}

#endif //SMARTPEAK_MODEL_H