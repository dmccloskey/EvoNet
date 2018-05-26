/**TODO:  Add copyright*/

#include <SmartPeak/io/LinkFile.h>
#include <SmartPeak/io/csv.h>

namespace SmartPeak
{

  LinkFile::LinkFile(){}
  LinkFile::~LinkFile(){}
 
  bool LinkFile::loadLinkBinary(const std::string& filename, std::vector<Link>& links){}
  bool LinkFile::loadLinkCsv(const std::string& filename, std::vector<Link>& links){}
  bool LinkFile::storeLinkBinary(const std::string& filename, const std::vector<Link>& links){}
  bool LinkFile::storeLinkCsv(const std::string& filename, const std::vector<Link>& links){}
}