/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Interpreter test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/Interpreter.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(interpreter)

BOOST_AUTO_TEST_CASE(constructor) 
{
  Interpreter* ptr = nullptr;
  Interpreter* nullPointer = nullptr;
	ptr = new Interpreter();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  Interpreter* ptr = nullptr;
	ptr = new Interpreter();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  Interpreter interpreter(1.0, 2.0, 3.0, 4.0);

  BOOST_CHECK_EQUAL(interpreter.getH(), 1.0);
  BOOST_CHECK_EQUAL(interpreter.getTau(), 2.0);
  BOOST_CHECK_EQUAL(interpreter.getMu(), 3.0);
  BOOST_CHECK_EQUAL(interpreter.getSigma(), 4.0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  Interpreter interpreter;
  interpreter.setH(1.0);
  interpreter.setTau(2.0);
  interpreter.setMu(3.0);
  interpreter.setSigma(4.0);

  BOOST_CHECK_EQUAL(interpreter.getH(), 1.0);
  BOOST_CHECK_EQUAL(interpreter.getTau(), 2.0);
  BOOST_CHECK_EQUAL(interpreter.getMu(), 3.0);
  BOOST_CHECK_EQUAL(interpreter.getSigma(), 4.0);
}

BOOST_AUTO_TEST_SUITE_END()