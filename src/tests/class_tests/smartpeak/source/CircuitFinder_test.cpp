/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE CircuitFinder test suite 
// #include <boost/test/unit_test.hpp> // changes every so often...
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/graph/CircuitFinder.h>

#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

#include <vector>
#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(circuitFinder)

BOOST_AUTO_TEST_CASE(constructor) 
{
  CircuitFinder* ptr = nullptr;
  CircuitFinder* nullPointer = nullptr;
	ptr = new CircuitFinder();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  CircuitFinder* ptr = nullptr;
	ptr = new CircuitFinder();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(test) 
{
	int A1[5][5] = {
		2, 0, 0, 0, 0,
		2, 3, 4, 0, 0,
		5, 0, 0, 0, 0,
		3, 0, 0, 0, 0,
		1, 0, 0, 0, 0
	};

	CircuitFinder<5> CF1(A1);
	CF1.run();

	int A2[6][6] = {
		2, 5, 0, 0, 0, 0,
		3, 0, 0, 0, 0, 0,
		1, 2, 4, 6, 0, 0,
		5, 0, 0, 0, 0, 0,
		2, 0, 0, 0, 0, 0,
		4, 0, 0, 0, 0, 0,
	};

	CircuitFinder<6> CF1(A2);
	CF2.run();
}

BOOST_AUTO_TEST_SUITE_END()