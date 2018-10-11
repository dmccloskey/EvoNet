/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHT_H
#define SMARTPEAK_WEIGHT_H

// .h
#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/WeightInit.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <tuple>
#include <string>

// .cpp
#include <SmartPeak/ml/SharedFunctions.h>
#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{

  /**
    @brief Directed Network Weight
  */
	template<typename TensorT>
  class Weight
  {
public:
    Weight() = default; ///< Default constructor
    Weight(const Weight& other); ///< Copy constructor // [TODO: add test]
    Weight(const int& id); ///< Explicit constructor 
    Weight(const std::string& name); ///< Explicit constructor 
    Weight(const int& id, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver); ///< Explicit constructor 
    Weight(const std::string& name, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver); ///< Explicit constructor 
    ~Weight() = default; ///< Default destructor

    inline bool operator==(const Weight& other) const
    {
      return
        std::tie(
          id_,
          name_,
					weight_,
					//weight_init_->getName(),
					//solver_->getName(),
					module_id_,
					module_name_
        ) == std::tie(
          other.id_,
          other.name_,
					other.weight_,
					//other.weight_init_->getName(),
					//other.solver_->getName(),
					other.module_id_,
					other.module_name_
        )
      ;
    }

    inline bool operator!=(const Weight& other) const
    {
      return !(*this == other);
    }

    inline Weight& operator=(const Weight& other)
    { // [TODO: add test]
      id_  = other.id_;
      name_  = other.name_;
			module_id_ = other.module_id_;
			module_name_ = other.module_name_;
      weight_  = other.weight_;
      weight_init_ = other.weight_init_;
      solver_ = other.solver_;
      weight_min_ = other.weight_min_;
      weight_max_ = other.weight_max_;
			drop_probability_ = other.drop_probability_;
			drop_ = other.drop_;
      return *this;
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< naem setter
    std::string getName() const; ///< name getter

    void setWeight(const TensorT& weight); ///< weight setter
    TensorT getWeight() const; ///< weight getter
		TensorT* getWeightMutable(); ///< weight getter

    void setWeightInitOp(const std::shared_ptr<WeightInitOp<TensorT>>& weight_init); ///< weight initialization operator setter
    WeightInitOp<TensorT>* getWeightInitOp() const; ///< weight initialization operator getter

    void setSolverOp(const std::shared_ptr<SolverOp<TensorT>>& solver); ///< weight update operator setter
    SolverOp<TensorT>* getSolverOp() const; ///< weight update operator getter

    void setWeightMin(const TensorT& weight_min); ///< min weight setter
    void setWeightMax(const TensorT& weight_max); ///< max weight setter

		void setModuleId(const int& module_id); ///< module id setter
		int getModuleId() const; ///< module id getter

		void setModuleName(const std::string& module_name); ///< module name setter
		std::string getModuleName() const; ///< module name getter

		void setDropProbability(const TensorT& drop_probability); ///< drop_probability setter
		TensorT getDropProbability() const; ///< drop_probability getter

		void setDrop(const TensorT& drop); ///< drop setter
		TensorT getDrop() const; ///< drop getter

    /**
      @brief Initializes the weight.  
    */ 
    void initWeight();
 
    /**
      @brief Update the weight.

      @param[in] errpr Weight error   
    */ 
    void updateWeight(const TensorT& error);
 
    /**
      @brief Check if the weight is within the min/max.  
    */ 
    void checkWeight();

private:
    int id_ = -1; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
		int module_id_ = -1; ///< Module ID
		std::string module_name_ = ""; ///<Module Name
    TensorT weight_ = 1.0; ///< Weight weight
    std::shared_ptr<WeightInitOp<TensorT>> weight_init_; ///< weight initialization operator
    std::shared_ptr<SolverOp<TensorT>> solver_; ///< weight update operator

    TensorT weight_min_ = -1.0e6;
    TensorT weight_max_ = 1.0e6;
		TensorT drop_probability_ = 0.0;
		TensorT drop_ = 1.0;
  };
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
	Weight<TensorT>::Weight(const int& id) :
		id_(id)
	{
		if (name_ == "")
		{
			name_ = std::to_string(id);
		}
	}

	template<typename TensorT>
	Weight<TensorT>::Weight(const std::string& name) :
		name_(name)
	{
	}

	template<typename TensorT>
	Weight<TensorT>::Weight(const int& id, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver) :
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
	Weight<TensorT>::Weight(const std::string& name, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver) :
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
	void Weight<TensorT>::setWeightInitOp(const std::shared_ptr<WeightInitOp<TensorT>>& weight_init)
	{
		weight_init_.reset();
		weight_init_ = std::move(weight_init);
	}
	template<typename TensorT>
	WeightInitOp<TensorT>* Weight<TensorT>::getWeightInitOp() const
	{
		return weight_init_.get();
	}

	template<typename TensorT>
	void Weight<TensorT>::setSolverOp(const std::shared_ptr<SolverOp<TensorT>>& solver)
	{
		solver_.reset();
		solver_ = std::move(solver);
	}
	template<typename TensorT>
	SolverOp<TensorT>* Weight<TensorT>::getSolverOp() const
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

#endif //SMARTPEAK_WEIGHT_H