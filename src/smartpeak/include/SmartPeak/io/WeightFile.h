/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTFILE_H
#define SMARTPEAK_WEIGHTFILE_H

#include <SmartPeak/ml/Weight.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

namespace SmartPeak
{

  /**
    @brief WeightFile
  */
	template<typename TensorT>
  class WeightFile
  {
public:
    WeightFile() = default; ///< Default constructor
    ~WeightFile() = default; ///< Default destructor
 
    /**
      @brief Load weights from binary file

      @param filename The name of the weights file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */ 
    bool loadWeightsBinary(const std::string& filename, std::vector<Weight<TensorT>>& weights);
 
    /**
      @brief Load weights from csv file

      @param filename The name of the weights file
      @param weights The weights to load data into

      @returns Status True on success, False if not
    */ 
    bool loadWeightsCsv(const std::string& filename, std::vector<Weight<TensorT>>& weights);
 
    /**
      @brief Stores weights from binary file

      @param filename The name of the weights file
      @param weights The weights to sore

      @returns Status True on success, False if not
    */ 
    bool storeWeightsBinary(const std::string& filename, const std::vector<Weight<TensorT>>& weights);
 
    /**
      @brief Stores weights from binary file

      @param filename The name of the weights file
      @param weights The weights to sore

      @returns Status True on success, False if not
    */ 
    bool storeWeightsCsv(const std::string& filename, const std::vector<Weight<TensorT>>& weights);

    std::map<std::string, TensorT> parseParameters(const std::string& parameters);
  };
}

#endif //SMARTPEAK_WEIGHTFILE_H