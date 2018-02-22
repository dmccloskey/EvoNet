/**TODO:  Add copyright*/

#include <SmartPeak/ml/Layer.h>

#include <vector>

namespace SmartPeak
{
  Layer::Layer()
  {        
  }

  Layer::Layer(const int& id, const std::vector<Link>& links):
    id_(id), links_(links)
  {
  }

  Layer::~Layer()
  {
  }
  
  void Layer::setId(const int& id)
  {
    id_ = id;
  }
  int Layer::getId() const
  {
    return id_;
  }

  void Layer::setLinks(const std::vector<Link>& links)
  {
    links_ = links;
  }
  std::vector<Link> Layer::getLinks() const
  {
    return links_;
  }
}