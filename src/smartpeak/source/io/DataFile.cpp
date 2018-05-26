/**TODO:  Add copyright*/

#include <SmartPeak/io/DataFile.h>

namespace SmartPeak
{

  DataFile::DataFile(){}
  DataFile::~DataFile(){}

  template<typename T>
  bool DataFile::loadDataBinary(const std::string& filename, T& data)
  {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    for (int i=0; i<data.NumDimensions; ++i) 
      in.read((char*) (&data.dimension(i)), sizeof(typename T::Index));
    // data.resize(rows, cols);
    in.read( (char *) data.data(),  data.size()*sizeof(typename T::Scalar));
    in.close();
  };

  template<typename T>
  bool DataFile::storeDataBinary(const std::string& filename, const T& data)
  {
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    for (int i=0; i<data.NumDimensions; ++i) 
      out.write((char*) (&data.dimension(i)), sizeof(typename T::Index));
    out.write((char*) data.data(), data.size()*sizeof(typename T::Scalar) );
    out.close();
  };
}