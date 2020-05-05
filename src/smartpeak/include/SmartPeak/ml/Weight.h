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

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp> // std::pair
#include <cereal/types/vector.hpp>

// .cpp
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
					init_weight_,
					//weight_init_->getName(),
					//solver_->getName(),
					module_id_,
					module_name_,
					tensor_index_
        ) == std::tie(
          other.id_,
          other.name_,
					other.weight_,
					other.init_weight_,
					//other.weight_init_->getName(),
					//other.solver_->getName(),
					other.module_id_,
					other.module_name_,
					other.tensor_index_
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
			layer_name_ = other.layer_name_;
			tensor_index_ = other.tensor_index_;
			weight_ = other.weight_;
			init_weight_ = other.init_weight_;
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

    void setWeightInitOp(const std::shared_ptr<WeightInitOp<TensorT>>& weight_init); ///< weight initialization operator setter
    WeightInitOp<TensorT>* getWeightInitOp() const; ///< weight initialization operator getter

    void setSolverOp(const std::shared_ptr<SolverOp<TensorT>>& solver); ///< weight update operator setter
		std::shared_ptr<SolverOp<TensorT>> getSolverOpShared() const; ///< weight update operator getter
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

		void setInitWeight(const bool& drop); ///< init_weight setter
		bool getInitWeight() const; ///< init_weight getter

		void addTensorIndex(const std::tuple<int, int, int>& layer_id); ///< layer id setter
		std::vector<std::tuple<int, int, int>> getTensorIndex() const; ///< layer id getter
		void clearTensorIndex();

		void setLayerName(const std::string& layer_name); ///< layer name setter
		std::string getLayerName() const; ///< layer name getter

    /**
      @brief Initializes the weight.  
    */ 
    void initWeight();

private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(id_, name_, module_id_, module_name_, tensor_index_, layer_name_,
				weight_, init_weight_, weight_init_, solver_, weight_min_,
				weight_max_);
		}
    int id_ = -1; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
		int module_id_ = -1; ///< Module ID
		std::string module_name_ = ""; ///<Module Name
		std::string layer_name_ = ""; ///< Layer name
		std::vector<std::tuple<int, int, int>> tensor_index_; ///< Layer ID: tuple consisting of OperationsList index and source/sink Layer index(used internally by Model)
    std::shared_ptr<WeightInitOp<TensorT>> weight_init_; ///< weight initialization operator
    std::shared_ptr<SolverOp<TensorT>> solver_; ///< weight update operator
		TensorT weight_ = TensorT(1);
		bool init_weight_ = true; ///< whether to initialize the weight or use the provided value of `weight_`
    TensorT weight_min_ = TensorT(-1.0e6);
    TensorT weight_max_ = TensorT(1.0e6);
		TensorT drop_probability_ = TensorT(0.0);
		TensorT drop_ = TensorT(1);
  };

	template<typename TensorT>
	inline Weight<TensorT>::Weight(const Weight<TensorT>& other)
	{
		id_ = other.id_;
		name_ = other.name_;
		weight_ = other.weight_;
		init_weight_ = other.init_weight_;
		module_id_ = other.module_id_;
		module_name_ = other.module_name_;
		layer_name_ = other.layer_name_;
		tensor_index_ = other.tensor_index_;
    setWeightInitOp(std::shared_ptr<WeightInitOp<TensorT>>(other.weight_init_.get()->copy()));
    setSolverOp(std::shared_ptr<SolverOp<TensorT>>(other.solver_.get()->copy()));
		weight_min_ = other.weight_min_;
		weight_max_ = other.weight_max_;
		drop_probability_ = other.drop_probability_;
		drop_ = other.drop_;
	}

	template<typename TensorT>
	inline Weight<TensorT>::Weight(const int& id) :
		id_(id)
	{
		if (name_ == "")
		{
			name_ = std::to_string(id);
		}
	}

	template<typename TensorT>
	inline Weight<TensorT>::Weight(const std::string& name) :
		name_(name)
	{
	}

	template<typename TensorT>
	inline Weight<TensorT>::Weight(const int& id, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver) :
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
	inline Weight<TensorT>::Weight(const std::string& name, const std::shared_ptr<WeightInitOp<TensorT>>& weight_init, const std::shared_ptr<SolverOp<TensorT>>& solver) :
		name_(name)
	{
		setWeightInitOp(weight_init);
		setSolverOp(solver);
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setId(const int& id)
	{
		id_ = id;
		if (name_ == "")
		{
			name_ = std::to_string(id);
		}
	}
	template<typename TensorT>
	inline int Weight<TensorT>::getId() const
	{
		return id_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setName(const std::string& name)
	{
		name_ = name;
	}
	template<typename TensorT>
	inline std::string Weight<TensorT>::getName() const
	{
		return name_;
	}

	//template<typename TensorT>
	//void Weight<TensorT>::setWeight(const TensorT& weight)
	//{
	//	weight_data_->setWeight(weight);
	//}

	//template<typename TensorT>
	//TensorT Weight<TensorT>::getWeightView() const
	//{
	//	return weight_data_->getWeight()(0);
	//	//return weight_ * getDrop();
	//}

	//template<typename TensorT>
	//TensorT Weight<TensorT>::getWeight()
	//{
	//	return weight_data_->getWeight()(0);
	//	//return weight_ * getDrop();
	//}

	template<typename TensorT>
	inline void Weight<TensorT>::setWeight(const TensorT& weight)
	{
		weight_ = weight;
	}

	template<typename TensorT>
	inline TensorT Weight<TensorT>::getWeight() const
	{
		return weight_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setWeightInitOp(const std::shared_ptr<WeightInitOp<TensorT>>& weight_init)
	{
		weight_init_.reset();
		weight_init_ = std::move(weight_init);
	}
	template<typename TensorT>
	inline WeightInitOp<TensorT>* Weight<TensorT>::getWeightInitOp() const
	{
		return weight_init_.get();
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setSolverOp(const std::shared_ptr<SolverOp<TensorT>>& solver)
	{
		solver_.reset();
		solver_ = std::move(solver);
	}
	template<typename TensorT>
	inline std::shared_ptr<SolverOp<TensorT>> Weight<TensorT>::getSolverOpShared() const
	{
		return solver_;
	}
	template<typename TensorT>
	inline SolverOp<TensorT>* Weight<TensorT>::getSolverOp() const
	{
		return solver_.get();
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setWeightMin(const TensorT& weight_min)
	{
		weight_min_ = weight_min;
	}
	template<typename TensorT>
	inline void Weight<TensorT>::setWeightMax(const TensorT& weight_max)
	{
		weight_max_ = weight_max;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setModuleId(const int & module_id)
	{
		module_id_ = module_id;
	}

	template<typename TensorT>
	inline int Weight<TensorT>::getModuleId() const
	{
		return module_id_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setModuleName(const std::string & module_name)
	{
		module_name_ = module_name;
	}

	template<typename TensorT>
	inline std::string Weight<TensorT>::getModuleName() const
	{
		return module_name_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setLayerName(const std::string & layer_name)
	{
		layer_name_ = layer_name;
	}

	template<typename TensorT>
	inline std::string Weight<TensorT>::getLayerName() const
	{
		return layer_name_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setDropProbability(const TensorT & drop_probability)
	{
		//drop_probability_ = drop_probability;
		//RandBinaryOp<TensorT> rand_bin(drop_probability_);
		//setDrop(rand_bin((TensorT)1));
	}

	template<typename TensorT>
	inline TensorT Weight<TensorT>::getDropProbability() const
	{
		return drop_probability_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setDrop(const TensorT & drop)
	{
		drop_ = drop;
	}

	template<typename TensorT>
	inline TensorT Weight<TensorT>::getDrop() const
	{
		return drop_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::setInitWeight(const bool & init_weight)
	{
		init_weight_ = init_weight;
	}

	template<typename TensorT>
	inline bool Weight<TensorT>::getInitWeight() const
	{
		return init_weight_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::addTensorIndex(const std::tuple<int, int, int>& layer_id)
	{
		tensor_index_.push_back(layer_id);
	}

	template<typename TensorT>
	inline std::vector<std::tuple<int, int, int>> Weight<TensorT>::getTensorIndex() const
	{
		return tensor_index_;
	}

	template<typename TensorT>
	inline void Weight<TensorT>::clearTensorIndex()
	{
		tensor_index_.clear();
	}

	template<typename TensorT>
	inline void Weight<TensorT>::initWeight()
	{
		weight_ = weight_init_->operator()();
	}
}

#endif //SMARTPEAK_WEIGHT_H