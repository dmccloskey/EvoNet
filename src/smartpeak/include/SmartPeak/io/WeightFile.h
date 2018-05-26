/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTFILE_H
#define SMARTPEAK_WEIGHTFILE_H

#include <SmartPeak/ml/Weight.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace SmartPeak
{

  /**
    @brief WeightFile
  */
  class WeightFile
  {
public:
    WeightFile(); ///< Default constructor
    ~WeightFile(); ///< Default destructor
 
    /**
      @brief Load weights from file

      @param filename The name of the weights file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */ 
    bool loadWeightsBinary(const std::string& filename, std::vector<Weight>& weights);
    bool loadWeightsCsv(const std::string& filename, std::vector<Weight>& weights);
 
    /**
      @brief Load weights from file

      @param filename The name of the weights file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */ 
    bool storeWeightsBinary(const std::string& filename, const std::vector<Weight>& weights);
    bool storeWeightsCsv(const std::string& filename, const std::vector<Weight>& weights);
  };
}

#endif //SMARTPEAK_WEIGHTFILE_H