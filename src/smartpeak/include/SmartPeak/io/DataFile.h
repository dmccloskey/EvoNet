/**TODO:  Add copyright*/

#ifndef SMARTPEAK_DATAFILE_H
#define SMARTPEAK_DATAFILE_H

#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <fstream>
#include <vector>

namespace SmartPeak
{

  /**
    @brief DataFile

    based on the following:
      https://stackoverflow.com/questions/25389480/how-to-write-read-an-eigen-matrix-from-binary-file

    TODO copy over tests
  */
  class DataFile
  {
public:
    DataFile(); ///< Default constructor
    ~DataFile(); ///< Default destructor
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    template<typename T>
    bool loadDataBinary(const std::string& filename, T& data);
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    template<typename T>
    bool storeDataBinary(const std::string& filename, const T& data);
  };
}

#endif //SMARTPEAK_DATAFILE_H