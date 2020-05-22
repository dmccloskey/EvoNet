/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Parameters test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/io/Parameters.h>
#include <SmartPeak/test_config.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(Parameters)

BOOST_AUTO_TEST_CASE(sizeOfParametersTest)
{
  // Make the test tuple
  ID id("id", -1);
  DataDir data_dir("data_dir", std::string(""));
  BatchSize batch_size("batch_size", 32);
  MemorySize memory_size("memory_size", 64);
  auto parameters = std::make_tuple(id, data_dir, batch_size, memory_size);

  // Test the size
  size_t my_tuple_size = sizeOfParameters(parameters);
  BOOST_CHECK_EQUAL(my_tuple_size, 4);
}

BOOST_AUTO_TEST_CASE(loadParametersFromCsvTest)
{
  // Make the test tuple
  ID id("id", -1);
  DataDir data_dir("data_dir", std::string(""));
  BatchSize batch_size("batch_size", 0);
  MemorySize memory_size("memory_size", 0);
  auto parameters = std::make_tuple(id, data_dir, batch_size, memory_size);

  // Test reading in the parameters file
  const int id_int = -1;
  const std::string parameters_filename = SMARTPEAK_GET_TEST_DATA_PATH("Parameters.csv");
  LoadParametersFromCsv loadParametersFromCsv(id_int, parameters_filename);
  parameters = std::apply([&loadParametersFromCsv](auto&& ...args) { return loadParametersFromCsv(args...); }, parameters);
  BOOST_CHECK_EQUAL(std::get<BatchSize>(parameters).get(), 32);
  BOOST_CHECK_EQUAL(std::get<MemorySize>(parameters).get(), 64);
}

BOOST_AUTO_TEST_SUITE_END()