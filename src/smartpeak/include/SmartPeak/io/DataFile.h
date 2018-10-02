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
    template<typename T, int R>
    bool loadDataBinary(const std::string& filename, Eigen::Tensor<T, R>& data)
    {
      try
      {
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        in.seekg(0);
        Eigen::array<Eigen::DenseIndex, R> dims;
        for (int i=0; i<R; ++i) 
        {
          char value_char[12];
          in.read((char*) (&value_char), sizeof(value_char));
          dims[i] = (int)std::stoi(value_char);
          // printf("dimension loaded: %d = %d\n", i, dims[i]); // DEBUGGING
        }
        data = Eigen::Tensor<T, R>(dims);
        in.read((char *) data.data(), sizeof(data.data()));
        in.close();
		    return true;
      }
      catch (std::exception& e)
      {
        printf("Exception: %s", e.what());
		    return false;
      }
      catch (...)
      {
        printf("Exception");
		    return false;
      }
    };
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    template<typename T, int R>
    bool storeDataBinary(const std::string& filename, const Eigen::Tensor<T, R>& data)
    {
      try
      {
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        for (int i=0; i<R; ++i) 
        {
          char value_char[12];
          // printf("dimension stored: %d = %d\n", i, data.dimension(i)); // DEBUGGING
          sprintf(value_char, "%d", data.dimension(i));
          out.write((char*) (&value_char), sizeof(value_char));
          // out.write((char*) (&data.dimension(i)), sizeof(typename Eigen::Tensor<T, R>::Index));
        }
        out.write((char*) data.data(), data.size()*sizeof(typename Eigen::Tensor<T, R>::Scalar));
        out.close();
		return true;
      }
      catch (std::exception& e)
      {
        printf("Exception: %s", e.what());
		return false;
      }
      catch (...)
      {
        printf("Exception");
		return false;
      }
    };
  };
}

#endif //SMARTPEAK_DATAFILE_H