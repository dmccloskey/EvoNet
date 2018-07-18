/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LINKFILE_H
#define SMARTPEAK_LINKFILE_H

#include <SmartPeak/ml/Link.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace SmartPeak
{

  /**
    @brief LinkFile
  */
  class LinkFile
  {
public:
    LinkFile(); ///< Default constructor
    ~LinkFile(); ///< Default destructor
 
    /**
      @brief Load links from file

      @param filename The name of the links file
      @param links The links to load data into

      @returns Status True on success, False if not
    */ 
    bool loadLinksBinary(const std::string& filename, std::vector<Link>& links);
    bool loadLinksCsv(const std::string& filename, std::vector<Link>& links);
 
    /**
      @brief save links to file

      @param filename The name of the links file
      @param links The links to load data into

      @returns Status True on success, False if not
    */ 
    bool storeLinksBinary(const std::string& filename, const std::vector<Link>& links);
    bool storeLinksCsv(const std::string& filename, const std::vector<Link>& links);

		/**
		@brief save network to file in dot format for visualization
			using e.g., GraphVIZ

		[TODO: move to GraphFile and take in the model as input
			to allow for the following
			1. coloring of nodes based on node type (i.e., input, hidden, bias, or output)
				e.g. node1 [shape=circle,style=filled,color=".7 .3 1.0"];
			2. annotation of links with the value of the weight
				e.g. node1 -> node2 [style=italic,label="weight = 10"];
		]

		@param filename The name of the links file (.gv extension)
		@param links The links to save to disk

		@returns Status True on success, False if not
		*/
		bool storeLinksDot(const std::string& filename, const std::vector<Link>& links);
  };
}

#endif //SMARTPEAK_LINKFILE_H