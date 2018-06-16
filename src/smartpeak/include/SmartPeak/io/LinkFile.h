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
      @brief Load links from file

      @param filename The name of the links file
      @param links The links to load data into

      @returns Status True on success, False if not
    */ 
    bool storeLinksBinary(const std::string& filename, const std::vector<Link>& links);
    bool storeLinksCsv(const std::string& filename, const std::vector<Link>& links);
  };
}

#endif //SMARTPEAK_LINKFILE_H