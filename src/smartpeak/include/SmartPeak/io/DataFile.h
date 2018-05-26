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
      https://gist.github.com/infusion/43bd2aa421790d5b4582

    TODO copy over tests
  */
  template<typedef T>
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
    bool loadDataBinary(const std::string& filename, T& data)
    {
      std::ifstream in(filename, ios::in | std::ios::binary);
      Eigen::Index rows=0, cols=0;
      for (int i=0; i<data.NumDimensions; ++i) 
        in.read((char*) (&data.dimension(i)), sizeof(Eigen::Index));
      // data.resize(rows, cols);
      in.read( (char *) data.data(),  data.size()*sizeof(Eigen::Scalar));
      in.close();
    };
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    bool loadDataCsv(const std::string& filename, T& data);
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    bool storeDataBinary(const std::string& filename, const T& data)
    {
      std::ofstream out(filename, ios::out | ios::binary | ios::trunc);
      for (int i=0; i<data.NumDimensions; ++i) 
        out.write((char*) (&data.dimension(i)), sizeof(Eigen::Index));
      out.write((char*) data.data(), data.size()*sizeof(Eigen::Scalar) );
      out.close();
    };    
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    bool storeDataCsv(const std::string& filename, const T& data);
  };
}

#endif //SMARTPEAK_DATAFILE_H