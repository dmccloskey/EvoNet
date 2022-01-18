/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE StringParsing test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/core/StringParsing.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(stringParsing)

BOOST_AUTO_TEST_CASE(SP_ReplaceTokens)
{
	std::string test = ReplaceTokens("{postgres list}", { "[\{\}]" }, "");
	BOOST_CHECK_EQUAL(test, "postgres list");
}

BOOST_AUTO_TEST_CASE(SP_SplitString)
{
	std::vector<std::string> test = SplitString("a,b,c,d,e", ",");
	std::vector<std::string> check = { "a","b","c","d","e" };
	for (int i=0; i<check.size(); ++i)
		BOOST_CHECK_EQUAL(test[i], check[i]);
}

BOOST_AUTO_TEST_CASE(SP_RemoveWhiteSpaces)
{
	std::string test = RemoveWhiteSpaces("A     string with \t\t\t a lot of     \n\n whitespace\n");
	BOOST_CHECK_EQUAL(test, "Astringwithalotofwhitespace");
}

BOOST_AUTO_TEST_SUITE_END()