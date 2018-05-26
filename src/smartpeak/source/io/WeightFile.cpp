/**TODO:  Add copyright*/

#include <SmartPeak/io/WeightFile.h>
#include <SmartPeak/io/csv.h>

namespace SmartPeak
{

  WeightFile::WeightFile(){}
  WeightFile::~WeightFile(){}
 
  bool WeightFile::loadWeightBinary(const std::string& filename, std::vector<Weight>& weights){}
  bool WeightFile::loadWeightCsv(const std::string& filename, std::vector<Weight>& weights){}
  bool WeightFile::storeWeightBinary(const std::string& filename, const std::vector<Weight>& weights){}
  bool WeightFile::storeWeightCsv(const std::string& filename, const std::vector<Weight>& weights){}
}