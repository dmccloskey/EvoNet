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

  Weight::Weight(const Weight& other)
  {
    id_ = other.id_;
    name_ = other.name_;
    weight_ = other.weight_;
		module_id_ = other.module_id_;
		module_name_ = other.module_name_;
    weight_init_ = other.weight_init_;
    solver_ = other.solver_;
    weight_min_ = other.weight_min_;
    weight_max_ = other.weight_max_;
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
    checkWeight();
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

  void Weight::setWeightMin(const float& weight_min)
  {
    weight_min_ = weight_min;
  }
  void Weight::setWeightMax(const float& weight_max)
  {
    weight_max_ = weight_max;
  }

	void Weight::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	int Weight::getModuleId() const
	{
		return module_id_;
	}

	void Weight::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	std::string Weight::getModuleName() const
	{
		return module_name_;
	}

  void Weight::initWeight()
  {
    // weight_ = weight_init_();
    weight_ = weight_init_->operator()();
    checkWeight();
  }

  void Weight::updateWeight(const float& error)
  {
    //TEST: implement gradient clipping
    const float new_weight = solver_->operator()(weight_, error);
    weight_ = solver_->clipGradient(new_weight);   
    // weight_ = solver_->operator()(weight_, error);
    checkWeight();
  }

  void Weight::checkWeight()
  {
    if (weight_ < weight_min_)
      weight_ = weight_min_;
    else if (weight_ > weight_max_)
      weight_ = weight_max_;
  }
}