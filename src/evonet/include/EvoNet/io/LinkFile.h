/**TODO:  Add copyright*/

#ifndef EVONET_LINKFILE_H
#define EVONET_LINKFILE_H

#include <EvoNet/ml/Link.h>

#include <iostream>
#include <fstream>
#include <map>
#include <memory>

namespace EvoNet
{

  /**
    @brief LinkFile
  */
  class LinkFile
  {
public:
    LinkFile() = default; ///< Default constructor
    ~LinkFile() = default; ///< Default destructor
 
    /**
      @brief Load links from file

      @param filename The name of the links file
      @param links The links to load data into

      @returns Status True on success, False if not
    */ 
    bool loadLinksBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links);
    bool loadLinksCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links);
 
    /**
      @brief save links to file

      @param filename The name of the links file
      @param links The links to load data into

      @returns Status True on success, False if not
    */ 
    bool storeLinksBinary(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links);
    bool storeLinksCsv(const std::string& filename, std::map<std::string, std::shared_ptr<Link>>& links);
  };
}

#endif //EVONET_LINKFILE_H