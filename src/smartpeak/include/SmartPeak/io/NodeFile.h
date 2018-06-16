/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODEFILE_H
#define SMARTPEAK_NODEFILE_H

#include <SmartPeak/ml/Node.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <fstream>
#include <vector>

namespace SmartPeak
{

  /**
    @brief NodeFile
  */
  class NodeFile
  {
public:
    NodeFile(); ///< Default constructor
    ~NodeFile(); ///< Default destructor
 
    /**
      @brief Load nodes from file

      @param filename The name of the nodes file
      @param nodes The nodes to load data into

      @returns Status True on success, False if not
    */ 
    bool loadNodesBinary(const std::string& filename, std::vector<Node>& nodes);
    bool loadNodesCsv(const std::string& filename, std::vector<Node>& nodes);
 
    /**
      @brief Load nodes from file

      @param filename The name of the nodes file
      @param nodes The nodes to load data into

      @returns Status True on success, False if not
    */ 
    bool storeNodesBinary(const std::string& filename, const std::vector<Node>& nodes);
    bool storeNodesCsv(const std::string& filename, const std::vector<Node>& nodes);
  };
}

#endif //SMARTPEAK_NODEFILE_H