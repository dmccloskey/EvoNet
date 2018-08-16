/**TODO:  Add copyright*/

#include <SmartPeak/ml/ModelBuilder.h>
#include <SmartPeak/core/Preprocessing.h>

#include <random> // random number generator
#include <algorithm> // tokenizing
#include <regex> // tokenizing
#include <ctime> // time format
#include <chrono> // current time
#include <set>
#include "..\..\include\SmartPeak\ml\ModelBuilder.h"

namespace SmartPeak
{
	std::vector<std::string> ModelBuilder::addInputNodes(Model & model, const std::string & name, const int & n_nodes)
	{
		return std::vector<std::string>();
	}
}