/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Interpreter test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Interpreter.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(interpreter)

// BOOST_AUTO_TEST_CASE(constructor) 
// {
//   Interpreter* ptr = nullptr;
//   Interpreter* nullPointer = nullptr;
// 	ptr = new Interpreter();
//   BOOST_CHECK_NE(ptr, nullPointer);
// }

// BOOST_AUTO_TEST_CASE(destructor) 
// {
//   Interpreter* ptr = nullptr;
// 	ptr = new Interpreter();
//   delete ptr;
// }

BOOST_AUTO_TEST_CASE(modelTest) 
{
  //TODO  
}

BOOST_AUTO_TEST_SUITE_END()