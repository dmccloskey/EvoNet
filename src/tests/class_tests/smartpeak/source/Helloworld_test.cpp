#include <boost/test/unit_test.hpp>
#include <SmartPeak/core/Helloworld.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(Helloworld)

Helloworld* ptr = nullptr;
Helloworld* nullPointer = nullptr;

BOOST_AUTO_TEST_CASE(constructor) 
{
	ptr = new helloworld();
  BOOST_CHECK_PREDICATE(std::not_equal_to<T>, (ptr)(nullPointer)); 
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  delete ptr;
}

BOOST_AUTO_TEST_CASE(addNumbers) 
{
  helloworld hw;
  double test = hw.addNumbers(2.0, 2.0);
  BOOST_CHECK_EQUAL(test, 4.0);
}

BOOST_AUTO_TEST_SUITE_END()