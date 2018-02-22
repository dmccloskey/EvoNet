/**TODO:  Add copyright*/

#include <SmartPeak/ml/Model.h>

#include <vector>
#include <SmartPeak/ml/Link.h>

namespace SmartPeak
{
  Model::Model()
  {        
  }

  Model::Model(const int& id):
    id_(id)
  {
  }

  Model::~Model()
  {
  }
  
  void Model::setId(const double& id)
  {
    id_ = id;
  }
  double Model::getId() const
  {
    return id_;
  }

  void Model::setError(const double& error)
  {
    error_ = error;
  }
  double Model::getError() const
  {
    return error_;
  }

  void Model::addLinks(const std::vector<Link>& links)
  {
    links_.reserve(links_.size() + distance(links.begin(), links.end()));
    links_.insert(links_.end(), links.begin(), links.end());
  }
}