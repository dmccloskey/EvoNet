/**TODO:  Add copyright*/

#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/WeightInit.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

namespace SmartPeak
{
  Weight::Weight()
  {        
  }

  Weight::Weight(const int& id):
    id_(id)
  {
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }

  Weight::Weight(const std::string& name):
    name_(name)
  {
  }

  Weight::Weight(const int& id, std::shared_ptr<WeightInitOp>& weight_init, std::shared_ptr<SolverOp>& solver):
    id_(id)
  {
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
    setWeightInitOp(weight_init);
    setSolverOp(solver);
  }

  Weight::Weight(const std::string& name, std::shared_ptr<WeightInitOp>& weight_init, std::shared_ptr<SolverOp>& solver):
    name_(name)
  {
    setWeightInitOp(weight_init);
    setSolverOp(solver);
  }

  Weight::~Weight()
  {
  }
  
  void Weight::setId(const int& id)
  {
    id_ = id;
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }
  int Weight::getId() const
  {
    return id_;
  }
  
  void Weight::setName(const std::string& name)
  {
    name_ = name;    
  }
  std::string Weight::getName() const
  {
    return name_;
  }

  void Weight::setWeight(const float& weight)
  {
    weight_ = weight;
  }
  float Weight::getWeight() const
  {
    return weight_;
  }

  void Weight::setWeightInitOp(const std::shared_ptr<WeightInitOp>& weight_init)
  {
    weight_init_.reset();
    weight_init_ = std::move(weight_init);
  }
  WeightInitOp* Weight::getWeightInitOp() const
  {
    return weight_init_.get();
  }

  void Weight::setSolverOp(const std::shared_ptr<SolverOp>& solver)
  {
    solver_.reset();
    solver_ = std::move(solver);
  }
  SolverOp* Weight::getSolverOp() const
  {
    return solver_.get();
  }

  void Weight::initWeight()
  {
    // weight_ = weight_init_();
    weight_ = weight_init_->operator()();
  }

  void Weight::updateWeight(const float& error)
  {
    //TODO: implement gradient clipping
    // const float new_weight = solver_->operator()(weight_, error);
    // weight_ = solver_->clip_gradient(new_weight);   
    weight_ = solver_->operator()(weight_, error);
  }
}