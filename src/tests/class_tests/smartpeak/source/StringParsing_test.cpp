/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE StringParsing test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/StringParsing.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(stringParsing)

BOOST_AUTO_TEST_CASE(SP_RemoveTokens)
{
	std::string test = RemoveTokens("{postgres list}", { "[\{\}]" });
	BOOST_CHECK_EQUAL(test, "postgres list");
}

BOOST_AUTO_TEST_CASE(SP_SplitString)
{
	std::vector<std::string> test = SplitString("a,b,c,d,e", ",");
	std::vector<std::string> check = { "a","b","c","d","e" };
	for (int i=0; i<check.size(); ++i)
		BOOST_CHECK_EQUAL(test[i], check[i]);
}

BOOST_AUTO_TEST_SUITE_END()