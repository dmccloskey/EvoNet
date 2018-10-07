/**TODO:  Add copyright*/

#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/WeightInit.h>
#include <SmartPeak/ml/SharedFunctions.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

namespace SmartPeak
{
	template<typename TensorT>
  Weight<TensorT>::Weight(const Weight<TensorT>& other)
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
		drop_probability_ = other.drop_probability_;
		drop_ = other.drop_;
  }

	template<typename TensorT>
  Weight<TensorT>::Weight(const int& id):
    id_(id)
  {
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }

	template<typename TensorT>
  Weight<TensorT>::Weight(const std::string& name):
    name_(name)
  {
  }

	template<typename TensorT>
  Weight<TensorT>::Weight(const int& id, const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver):
    id_(id)
  {
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
    setWeightInitOp(weight_init);
    setSolverOp(solver);
  }

	template<typename TensorT>
  Weight<TensorT>::Weight(const std::string& name, const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver):
    name_(name)
  {
    setWeightInitOp(weight_init);
    setSolverOp(solver);
  }

	template<typename TensorT>
  void Weight<TensorT>::setId(const int& id)
  {
    id_ = id;
    if (name_ == "")
    {
      name_ = std::to_string(id);
    }
  }
	template<typename TensorT>
  int Weight<TensorT>::getId() const
  {
    return id_;
  }

	template<typename TensorT>
  void Weight<TensorT>::setName(const std::string& name)
  {
    name_ = name;    
  }
	template<typename TensorT>
  std::string Weight<TensorT>::getName() const
  {
    return name_;
  }

	template<typename TensorT>
  void Weight<TensorT>::setWeight(const TensorT& weight)
  {
    weight_ = weight;
    checkWeight();
  }
	template<typename TensorT>
  TensorT Weight<TensorT>::getWeight() const
  {
    return weight_ * getDrop();
  }

	template<typename TensorT>
	TensorT* Weight<TensorT>::getWeightMutable()
	{
		return &weight_;
	}

	template<typename TensorT>
  void Weight<TensorT>::setWeightInitOp(const std::shared_ptr<WeightInitOp>& weight_init)
  {
    weight_init_.reset();
    weight_init_ = std::move(weight_init);
  }
	template<typename TensorT>
  WeightInitOp* Weight<TensorT>::getWeightInitOp() const
  {
    return weight_init_.get();
  }

	template<typename TensorT>
  void Weight<TensorT>::setSolverOp(const std::shared_ptr<SolverOp>& solver)
  {
    solver_.reset();
    solver_ = std::move(solver);
  }
	template<typename TensorT>
  SolverOp* Weight<TensorT>::getSolverOp() const
  {
    return solver_.get();
  }

	template<typename TensorT>
  void Weight<TensorT>::setWeightMin(const TensorT& weight_min)
  {
    weight_min_ = weight_min;
  }
	template<typename TensorT>
  void Weight<TensorT>::setWeightMax(const TensorT& weight_max)
  {
    weight_max_ = weight_max;
  }

	template<typename TensorT>
	void Weight<TensorT>::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	template<typename TensorT>
	int Weight<TensorT>::getModuleId() const
	{
		return module_id_;
	}

	template<typename TensorT>
	void Weight<TensorT>::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	template<typename TensorT>
	std::string Weight<TensorT>::getModuleName() const
	{
		return module_name_;
	}

	template<typename TensorT>
	void Weight<TensorT>::setDropProbability(const TensorT & drop_probability)
	{
		drop_probability_ = drop_probability;
		RandBinaryOp<TensorT> rand_bin(drop_probability_);
		setDrop(rand_bin(1.0f));
	}

	template<typename TensorT>
	TensorT Weight<TensorT>::getDropProbability() const
	{
		return drop_probability_;
	}

	template<typename TensorT>
	void Weight<TensorT>::setDrop(const TensorT & drop)
	{
		drop_ = drop;
	}

	template<typename TensorT>
	TensorT Weight<TensorT>::getDrop() const
	{
		return drop_;
	}

	template<typename TensorT>
  void Weight<TensorT>::initWeight()
  {
    // weight_ = weight_init_();
    weight_ = weight_init_->operator()();
    checkWeight();
  }

	template<typename TensorT>
  void Weight<TensorT>::updateWeight(const TensorT& error)
  {
		if (solver_->getName() == "DummySolverOp")
			return;
    const TensorT new_weight = solver_->operator()(weight_, getDrop()*error);
    weight_ = solver_->clipGradient(new_weight);
    checkWeight();
  }

	template<typename TensorT>
  void Weight<TensorT>::checkWeight()
  {
    if (weight_ < weight_min_)
      weight_ = weight_min_;
    else if (weight_ > weight_max_)
      weight_ = weight_max_;
  }
}