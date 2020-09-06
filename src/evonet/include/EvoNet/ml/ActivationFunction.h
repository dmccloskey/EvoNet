/**TODO:  Add copyright*/

#ifndef EVONET_ACTIVATIONFUNCTION_H
#define EVONET_ACTIVATIONFUNCTION_H

#include <EvoNet/core/Preprocessing.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <iostream>
#include <limits>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace EvoNet
{
  /**
    @brief Base class for all activation functions.
  */
  template<typename TensorT>
  class ActivationOp
  {
  public: 
	  ActivationOp() = default;
    ActivationOp(const TensorT& eps, const TensorT& min, const TensorT& max) : eps_(eps), min_(min), max_(max) {};
		virtual ~ActivationOp() = default;
		void setEps(const TensorT& eps) { eps_ = eps; }
		void setMin(const TensorT& min) { min_ = min; }
		void setMax(const TensorT& max) { max_ = max; }
    TensorT getEps() const { return eps_; }
    TensorT getMin() const { return min_; }
    TensorT getMax() const { return max_; }
		virtual std::string getName() const = 0;
		virtual std::vector<TensorT> getParameters() const = 0;
    virtual ActivationOp<TensorT>* copy() const = 0;
//#endif // !EVONET_CUDA
	protected:
		TensorT eps_ = (TensorT)1e-6; ///< threshold to clip between min and max
		TensorT min_ =TensorT(-1e9);
		TensorT max_ = TensorT(1e9);
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(eps_, min_, max_);
		}
  };

  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT>
  class ReLUOp: public ActivationOp<TensorT>
  {
public: 
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "ReLUOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({this->getEps(), this->getMin(), this->getMax()}); }
    ActivationOp<TensorT>* copy() const { return new ReLUOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };

  /**
    @brief Rectified Linear Unit (ReLU) gradient

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT>
  class ReLUGradOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "ReLUGradOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new ReLUGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };

  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT>
  class ELUOp: public ActivationOp<TensorT>
  {
public:
		ELUOp() = default;
		~ELUOp() = default;
    ELUOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationOp(eps, min, max), alpha_(alpha) {};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUOp";};
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax(), alpha_ }); }
    ActivationOp<TensorT>* copy() const { return new ELUOp<TensorT>(*this); }
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this), alpha_);
		}
    TensorT alpha_ = 1;
  };

  /**
    @brief Exponential Linear Unit (ELU) gradient

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT>
  class ELUGradOp: public ActivationOp<TensorT>
  {
public:
		ELUGradOp() = default;
		~ELUGradOp() = default;
    ELUGradOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationOp(eps, min, max), alpha_(alpha) {};
    ELUGradOp(const TensorT& alpha): alpha_(alpha){}; 
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradOp";};
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax(), alpha_ }); }
    ActivationOp<TensorT>* copy() const { return new ELUGradOp<TensorT>(*this); }
private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this), alpha_);
		}
    TensorT alpha_ = 1;
  };

  /**
    @brief Sigmoid activation function
  */
  template<typename TensorT>
  class SigmoidOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "SigmoidOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new SigmoidOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };

  /**
    @brief Sigmoid gradient
  */
  template<typename TensorT>
  class SigmoidGradOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "SigmoidGradOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new SigmoidGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };
  
  /**
    @brief Hyperbolic Tangent activation function
  */
  template<typename TensorT>
  class TanHOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "TanHOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new TanHOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename TensorT>
  class TanHGradOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "TanHGradOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new TanHGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };
  
  /**
    @brief Rectified Hyperbolic Tangent activation function
  */
  template<typename TensorT>
  class ReTanHOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "ReTanHOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new ReTanHOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };

  /**
    @brief Rectified Hyperbolic Tangent gradient
  */
  template<typename TensorT>
  class ReTanHGradOp: public ActivationOp<TensorT>
  {
public:
		using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const{return "ReTanHGradOp";};
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new ReTanHGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
  };

	/**
	@brief Linear activation function
	*/
	template<typename TensorT>
	class LinearOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "LinearOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new LinearOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Linear gradient
	*/
	template<typename TensorT>
	class LinearGradOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "LinearGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new LinearGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Inverse activation function
	*/
	template<typename TensorT>
	class InverseOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "InverseOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new InverseOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Inverse gradient
	*/
	template<typename TensorT>
	class InverseGradOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "InverseGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new InverseGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Exponential activation function
	*/
	template<typename TensorT>
	class ExponentialOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "ExponentialOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new ExponentialOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Exponential gradient
	*/
	template<typename TensorT>
	class ExponentialGradOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "ExponentialGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new ExponentialGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Log activation function
	*/
	template<typename TensorT>
	class LogOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "LogOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new LogOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Log gradient
	*/
	template<typename TensorT>
	class LogGradOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "LogGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new LogGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Pow activation function
	*/
	template<typename TensorT>
	class PowOp : public ActivationOp<TensorT>
	{
	public:
		PowOp() = default;
		~PowOp() = default;
    PowOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& base) : ActivationOp(eps, min, max), base_(base) {};
    PowOp(const TensorT& base) : base_(base) {};
		std::string getName() const { return "PowOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax(), base_ }); }
    ActivationOp<TensorT>* copy() const { return new PowOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this), base_);
		}
		TensorT base_;
	};

	/**
	@brief Pow gradient
	*/
	template<typename TensorT>
	class PowGradOp : public ActivationOp<TensorT>
	{
	public:
		PowGradOp() = default;
		~PowGradOp() = default;
    PowGradOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& base) : ActivationOp(eps, min, max), base_(base) {};
		PowGradOp(const TensorT& base) : base_(base) {};
		std::string getName() const { return "PowGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax(), base_ }); }
    ActivationOp<TensorT>* copy() const { return new PowGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this), base_);
		}
		TensorT base_;
	};

	/**
		@brief LeakyReLU activation function

		default alpha = 1e-2
	*/
	template<typename TensorT>
	class LeakyReLUOp : public ActivationOp<TensorT>
	{
	public:
		LeakyReLUOp() = default;
		~LeakyReLUOp() = default;
    LeakyReLUOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationOp(eps, min, max), alpha_(alpha) {};
		LeakyReLUOp(const TensorT& alpha) : alpha_(alpha) {};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax(), alpha_ }); }
    ActivationOp<TensorT>* copy() const { return new LeakyReLUOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this), alpha_);
		}
		TensorT alpha_ = 1e-2;
	};

	/**
		@brief LeakyReLU gradient
	*/
	template<typename TensorT>
	class LeakyReLUGradOp : public ActivationOp<TensorT>
	{
	public:
		LeakyReLUGradOp() = default;
		~LeakyReLUGradOp() = default;
    LeakyReLUGradOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationOp(eps, min, max), alpha_(alpha) {};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUGradOp"; };
		std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax(), alpha_ }); }
    ActivationOp<TensorT>* copy() const { return new LeakyReLUGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this), alpha_);
		}
		TensorT alpha_ = 1e-2;
	};

	/**
	@brief Sin activation function
	*/
	template<typename TensorT>
	class SinOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "SinOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new SinOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Sin gradient
	*/
	template<typename TensorT>
	class SinGradOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "SinGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new SinGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Cos activation function
	*/
	template<typename TensorT>
	class CosOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "CosOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new CosOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

	/**
	@brief Cos gradient
	*/
	template<typename TensorT>
	class CosGradOp : public ActivationOp<TensorT>
	{
	public:
		using ActivationOp<TensorT>::ActivationOp;
		std::string getName() const { return "CosGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new CosGradOp<TensorT>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ActivationOp<TensorT>>(this));
		}
	};

  /**
  @brief BatchNorm activation function
  */
  template<typename TensorT>
  class BatchNormOp : public ActivationOp<TensorT>
  {
  public:
    using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const { return "BatchNormOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new BatchNormOp<TensorT>(*this); }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<ActivationOp<TensorT>>(this));
    }
  };

  /**
  @brief BatchNorm gradient
  */
  template<typename TensorT>
  class BatchNormGradOp : public ActivationOp<TensorT>
  {
  public:
    using ActivationOp<TensorT>::ActivationOp;
    std::string getName() const { return "BatchNormGradOp"; };
    std::vector<TensorT> getParameters() const { return std::vector<TensorT>({ this->getEps(), this->getMin(), this->getMax() }); }
    ActivationOp<TensorT>* copy() const { return new BatchNormGradOp<TensorT>(*this); }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<ActivationOp<TensorT>>(this));
    }
  };
}

CEREAL_REGISTER_TYPE(EvoNet::ReLUOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ReLUGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ELUOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ELUGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::SigmoidOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::SigmoidGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::TanHOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::TanHGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ReTanHOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ReTanHGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::LinearOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::LinearGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::InverseOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::InverseGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ExponentialOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::ExponentialGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::LogOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::LogGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::PowOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::PowGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::LeakyReLUOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::LeakyReLUGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::SinOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::SinGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::CosOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::CosGradOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::BatchNormOp<float>);
CEREAL_REGISTER_TYPE(EvoNet::BatchNormGradOp<float>);

//CEREAL_REGISTER_TYPE(EvoNet::ReLUOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ReLUGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ELUOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ELUGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::SigmoidOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::SigmoidGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::TanHOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::TanHGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ReTanHOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ReTanHGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::LinearOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::LinearGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::InverseOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::InverseGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ExponentialOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::ExponentialGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::LogOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::LogGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::PowOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::PowGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::LeakyReLUOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::LeakyReLUGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::SinOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::SinGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::CosOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::CosGradOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::BatchNormOp<double>);
//CEREAL_REGISTER_TYPE(EvoNet::BatchNormGradOp<double>);
//
//CEREAL_REGISTER_TYPE(EvoNet::ReLUOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ReLUGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ELUOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ELUGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::SigmoidOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::SigmoidGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::TanHOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::TanHGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ReTanHOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ReTanHGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::LinearOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::LinearGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::InverseOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::InverseGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ExponentialOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::ExponentialGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::LogOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::LogGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::PowOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::PowGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::LeakyReLUOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::LeakyReLUGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::SinOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::SinGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::CosOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::CosGradOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::BatchNormOp<int>);
//CEREAL_REGISTER_TYPE(EvoNet::BatchNormGradOp<int>);

#endif //EVONET_ACTIVATIONFUNCTION_H