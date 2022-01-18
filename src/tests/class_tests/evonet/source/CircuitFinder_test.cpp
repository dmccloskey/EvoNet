/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE CircuitFinder test suite 
// #include <boost/test/unit_test.hpp> // changes every so often...
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/graph/CircuitFinder.h>

#include <EvoNet/ml/Link.h>
#include <EvoNet/ml/Node.h>

#include <vector>
#include <iostream>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(circuitFinder)

BOOST_AUTO_TEST_CASE(test)
{
	std::list<int>* A1;
	A1 = new std::list<int>[5];
	A1[0].push_back(2);
	A1[1].push_back(2); A1[1].push_back(3); A1[1].push_back(4);
	A1[2].push_back(5);
	A1[3].push_back(3);
	A1[4].push_back(1);

	CircuitFinder CF1(A1, 5);
	CF1.run();

	BOOST_CHECK_EQUAL(CF1.getCycles()[0].first, 5);
	BOOST_CHECK_EQUAL(CF1.getCycles()[1].first, 5);
	BOOST_CHECK_EQUAL(CF1.getCycles()[2].first, 2);
	BOOST_CHECK_EQUAL(CF1.getCycles()[0].second, 1);
	BOOST_CHECK_EQUAL(CF1.getCycles()[1].second, 1);
	BOOST_CHECK_EQUAL(CF1.getCycles()[2].second, 2);

	std::list<int>* A2;
	A2 = new std::list<int>[6];
	A2[0].push_back(2); A2[0].push_back(5);
	A2[1].push_back(3);
	A2[2].push_back(1); A2[2].push_back(2); A2[2].push_back(4); A2[2].push_back(6);
	A2[3].push_back(5);
	A2[4].push_back(2);
	A2[5].push_back(4);

	CircuitFinder CF2(A2, 6);
	CF2.run();

	BOOST_CHECK_EQUAL(CF2.getCycles()[0].first, 3);
	BOOST_CHECK_EQUAL(CF2.getCycles()[1].first, 3);
	BOOST_CHECK_EQUAL(CF2.getCycles()[2].first, 3);
	BOOST_CHECK_EQUAL(CF2.getCycles()[3].first, 5);
	BOOST_CHECK_EQUAL(CF2.getCycles()[4].first, 5);
	BOOST_CHECK_EQUAL(CF2.getCycles()[0].second, 1);
	BOOST_CHECK_EQUAL(CF2.getCycles()[1].second, 1);
	BOOST_CHECK_EQUAL(CF2.getCycles()[2].second, 2);
	BOOST_CHECK_EQUAL(CF2.getCycles()[3].second, 2);
	BOOST_CHECK_EQUAL(CF2.getCycles()[4].second, 2);
}

BOOST_AUTO_TEST_SUITE_END()