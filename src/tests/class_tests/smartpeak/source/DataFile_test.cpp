/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DataFile test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/io/DataFile.h>
// #include <filesystem> C++ 17

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(DataFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  DataFile* ptr = nullptr;
  DataFile* nullPointer = nullptr;
  ptr = new DataFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  DataFile* ptr = nullptr;
	ptr = new DataFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary) 
{
  DataFile data;

  // std::path data_path = std::current_path().replace_filename("data");  C++ 17
  // data_path /= "DataFileTest.dat";  C++ 17
  std::string filename = "../Data/DataFileTest.dat";

  Eigen::Tensor<float, 3> random_dat(2,2,2);
  random_dat.setRandom();
	data.storeDataBinary<Eigen::Tensor<float, 3>>(filename, random_dat);
	// data.storeDataBinary(data_path.string(), random_dat);  C++ 17

  Eigen::Tensor<float, 3> test_dat(2,2,2);
	data.loadDataBinary<Eigen::Tensor<float, 3>>(filename, test_dat);
	// data.loadDataBinary(data_path.string(), test_dat);  C++ 17

  BOOST_CHECK_CLOSE(test_dat(0, 0, 0), random_dat(0, 0, 0), 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()