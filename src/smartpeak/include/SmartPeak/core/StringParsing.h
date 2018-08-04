#ifndef SMARTPEAK_STRINGPARSING_H
#define SMARTPEAK_STRINGPARSING_H

#include <algorithm>>
#include <vector>
#include <string>>
#include <regex>

namespace SmartPeak
{
	/*
	@brief Methods for string parsing, tokenization, etc.
	*/


	/*
	@brief Remove tokens from a string

	Tests:
	std::string test = RemoveTokens("{postgres list}");
	BOOST_TEST_EQUAL(test, "postgres list");

	@param[in] string
	@param[in] tokens Vector of strings

	@returns string without tokens
	**/
	static std::string RemoveTokens(const std::string& str, const std::vector<std::string>& tokens)
	{
		std::string str_copy = str;
		for (const std::string& token : tokens)
			str_copy = std::regex_replace(str_copy, std::regex(token), "");
		return str_copy;
	}

	/*
	@brief Split string into a vector of substrings

	Tests:
	std::vector<std::string> test = SplitString("a,b,c,d,e");
	std::vector<std::string> check = {"a","b","c","d","e"};
	BOOST_TEST_EQUAL(test, check);

	@param[in] string
	@param[in] delimiter Token to use to split

	@returns a vector of strings
	**/
	static std::vector<std::string> SplitString(
		const std::string& str, 
		const std::string& delimiter)
	{
		std::vector<std::string> tokens;
		std::string str_copy = str;
		size_t pos = 0;
		while ((pos = str_copy.find(delimiter)) != std::string::npos) {
			std::string token = str_copy.substr(0, pos);
			tokens.push_back(token);
			str_copy.erase(0, pos + delimiter.length());
		}
		tokens.push_back(str_copy); // the last element
		return tokens;
	}

}

#endif //SMARTPEAK_STRINGPARSING_H