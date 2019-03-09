/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTINIT_H
#define SMARTPEAK_WEIGHTINIT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Base class for all weight initialization functions
  */
	template<typename TensorT>
  class WeightInitOp
  {
	public: 
    WeightInitOp() = default; 
		~WeightInitOp() = default;
    virtual std::string getName() const = 0;
    virtual TensorT operator()() const = 0;
    virtual std::string getParamsAsStr() const = 0;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive){}
  };  

  /**
    @brief Random weight initialization based on the method of He, et al 2015

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
	template<typename TensorT>
  class RandWeightInitOp: public WeightInitOp<TensorT>
  {
public: 
    RandWeightInitOp(TensorT n = 1.0, TensorT f = 1.0): n_(n), f_(f){};
    std::string getName() const{return "RandWeightInitOp";};
    TensorT operator()() const {       
      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0.0, 1.0};
      return d(gen)*std::sqrt(f_/n_); 
    };
    TensorT getN() const { return n_; }
		TensorT getF() const { return f_; }
    std::string getParamsAsStr() const
    {
			std::string params = "n:" + std::to_string(getN()) +
				";f:" + std::to_string(getF());
      return params;
    }
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<WeightInitOp<TensorT>>(this), n_, f_);
		}
    TensorT n_ = 1.0; ///< the denominator (i.e., number of input nodes for He et al, or average input/output nodes for Xavior et al)
		TensorT f_ = 1.0; ///< the numerator (i.e., 2 for He et al, 1 for Xavior et al)
  };

  /**
    @brief Constant weight initialization.
  */
	template<typename TensorT>
  class ConstWeightInitOp: public WeightInitOp<TensorT>
  {
public: 
    ConstWeightInitOp(const TensorT& n):n_(n){};
    ConstWeightInitOp(){}; 
    ~ConstWeightInitOp(){};
    std::string getName() const{return "ConstWeightInitOp";};
    TensorT operator()() const { return n_; };
    TensorT getN() const {return n_;};
    std::string getParamsAsStr() const
    {
      std::string params = "n:" + std::to_string(getN());
      return params;
    }
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<WeightInitOp<TensorT>>(this), n_);
		}
    TensorT n_ = 1.0; ///< the constant to return
  };

  /**
    @brief Random weight initialization over a lower and upper bound of values

  */
  template<typename TensorT>
  class RangeWeightInitOp : public WeightInitOp<TensorT>
  {
  public:
    RangeWeightInitOp(TensorT lb = 0.0, TensorT ub = 1.0) : lb_(lb), ub_(ub) {};
    std::string getName() const { return "RangeWeightInitOp"; };
    TensorT operator()() const {
      std::random_device rd{};
      std::mt19937 gen{ rd() };
      std::uniform_real_distribution<> d(lb_, ub_); // define the range
      return d(gen);
    };
    TensorT getLB() const { return lb_; }
    TensorT getUB() const { return ub_; }
    std::string getParamsAsStr() const
    {
      std::string params = "lb:" + std::to_string(getLB()) +
        ";ub:" + std::to_string(getUB());
      return params;
    }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<WeightInitOp<TensorT>>(this), lb_, ub_);
    }
    TensorT lb_ = 0.0; ///< the lower bound
    TensorT ub_ = 1.0; ///< the upper bound
  };
}

CEREAL_REGISTER_TYPE(SmartPeak::RandWeightInitOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::ConstWeightInitOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::RandWeightInitOp<double>);
CEREAL_REGISTER_TYPE(SmartPeak::ConstWeightInitOp<double>);
CEREAL_REGISTER_TYPE(SmartPeak::RandWeightInitOp<int>);
CEREAL_REGISTER_TYPE(SmartPeak::ConstWeightInitOp<int>);
#endif //SMARTPEAK_WEIGHTINIT_H