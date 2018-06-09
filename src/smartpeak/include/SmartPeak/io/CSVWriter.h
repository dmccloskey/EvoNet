/**TODO:  Add copyright*/

#ifndef SMARTPEAK_CSVWRITER_H
#define SMARTPEAK_CSVWRITER_H

#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <fstream>
#include <vector>

namespace SmartPeak
{

  /**
    @brief CSVWriter

    based on the following:
        http://thispointer.com/how-to-write-data-in-a-csv-file-in-c/
  */
  class CSVWriter
  {
public:
    CSVWriter(); ///< Default constructor
    ~CSVWriter(); ///< Default destructor
    CSVWriter(std::string filename, std::string delm = ",") :
        fileName(filename), delimeter(delm), linesCount(0)
    {}    
 
    /**
      @brief This Function accepts a range and appends all the elements in the range
        to the last row, seperated by delimeter (Default is comma)

      @param filename The name of the data file
      @param data The data to load data into
    */ 
    template<typename T>
    void addDatainRow(T first, T last);
private:
    std::string fileName;
    std::string delimeter;
    int linesCount;
  };
}

#endif //SMARTPEAK_CSVWRITER_H