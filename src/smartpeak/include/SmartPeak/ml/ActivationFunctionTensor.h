/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONTENSORFUNCTION_H
#define SMARTPEAK_ACTIVATIONTENSORFUNCTION_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>


//#include <cereal/access.hpp>  // serialiation of private members
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Base class for all activation function wrappers.
  */
  template<typename TensorT, typename DeviceT>
  class ActivationTensorOp
  {
  public: 
		ActivationTensorOp() {};
		~ActivationTensorOp() {};
		virtual std::string getName() const = 0;
		virtual void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {}
  };

  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT, typename DeviceT>
  class ReLUTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
	public: 
    ReLUTensorOp(){}; 
    ~ReLUTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReLUOp<TensorT>());
			auto result = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(x.chip(time_step, 1), x.chip(time_step, 1).constant(0));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    std::string getName() const{return "ReLUTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Rectified Linear Unit (ReLU) gradient

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT, typename DeviceT>
  class ReLUGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    ReLUGradTensorOp(){}; 
    ~ReLUGradTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReLUGradOp<TensorT>());
			auto result = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(x.chip(time_step, 1).constant(1), x.chip(time_step, 1).constant(0));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    std::string getName() const{return "ReLUGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT, typename DeviceT>
  class ELUTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    ELUTensorOp(){}; 
    ELUTensorOp(const TensorT& alpha): alpha_(alpha){}; 
    ~ELUTensorOp(){};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ELUOp<TensorT>(alpha_));
			auto result = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1), 
				x.chip(time_step, 1).constant(alpha_) * (x.chip(time_step, 1).exp() - x.chip(time_step, 1).constant(1)));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUTensorOp";};
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
    TensorT alpha_ = 1;
  };

  /**
    @brief Exponential Linear Unit (ELU) gradient

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT, typename DeviceT>
  class ELUGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    ELUGradTensorOp(){}; 
    ELUGradTensorOp(const TensorT& alpha): alpha_(alpha){}; 
    ~ELUGradTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ELUGradOp<TensorT>(alpha_));
			auto result = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(1),
				(x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(x.chip(time_step, 1), x.chip(time_step, 1).constant(alpha_) * (x.chip(time_step, 1).exp() - x.chip(time_step, 1).constant(1))) + x.chip(time_step, 1).constant(alpha_));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradTensorOp";};
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
    TensorT alpha_ = 1;
  };

  /**
    @brief Sigmoid activation function
  */
  template<typename TensorT, typename DeviceT>
  class SigmoidTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    SigmoidTensorOp(){}; 
    ~SigmoidTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(SigmoidOp<TensorT>());
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(1)/(x.chip(time_step, 1).constant(1) + (-x.chip(time_step, 1).exp()));
			auto result = x.chip(time_step, 1).sigmoid();
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    std::string getName() const{return "SigmoidTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Sigmoid gradient
  */
  template<typename TensorT, typename DeviceT>
  class SigmoidGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    SigmoidGradTensorOp(){}; 
    ~SigmoidGradTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(SigmoidGradOp<TensorT>());
			auto result = x.chip(time_step, 1).sigmoid() * (x.chip(time_step, 1).constant(1) - x.chip(time_step, 1).sigmoid());
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    std::string getName() const{return "SigmoidGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };
  
  /**
    @brief Hyperbolic Tangent activation function
  */
  template<typename TensorT, typename DeviceT>
  class TanHTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    TanHTensorOp(){}; 
    ~TanHTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(TanHOp<TensorT>());
			auto result = x.chip(time_step, 1).tanh();
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    std::string getName() const{return "TanHTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename TensorT, typename DeviceT>
  class TanHGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    TanHGradTensorOp(){}; 
    ~TanHGradTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(TanHGradOp<TensorT>());
			auto result = x.chip(time_step, 1).constant(1) - (x.chip(time_step, 1).tanh()).pow(2); 
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
    std::string getName() const{return "TanHGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };
  
  /**
    @brief Rectified Hyperbolic Tangent activation function
  */
  template<typename TensorT, typename DeviceT>
  class ReTanHTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    ReTanHTensorOp(){}; 
    ~ReTanHTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReTanHOp<TensorT>());
			//out.chip(time_step, 1).device(device) = [TODO]
		};
    std::string getName() const{return "ReTanHTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Rectified Hyperbolic Tangent gradient
  */
  template<typename TensorT, typename DeviceT>
  class ReTanHGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
public: 
    ReTanHGradTensorOp(){}; 
    ~ReTanHGradTensorOp(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReTanHGradOp<TensorT>());
			//out.chip(time_step, 1).device(device) = [TODO]
		};
    std::string getName() const{return "ReTanHGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

	/**
	@brief Linear activation function
	*/
	template<typename TensorT, typename DeviceT>
	class LinearTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		LinearTensorOp() {};
		~LinearTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LinearOp<TensorT>());
			auto result = x.chip(time_step, 1); 
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "LinearTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Linear gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LinearGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		LinearGradTensorOp() {};
		~LinearGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LinearGradOp<TensorT>());
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(1);
		};
		std::string getName() const { return "LinearGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Inverse activation function
	*/
	template<typename TensorT, typename DeviceT>
	class InverseTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		InverseTensorOp() {};
		~InverseTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(InverseOp<TensorT>());
			auto result = (x.chip(time_step, 1) != x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(1)/ x.chip(time_step, 1), x.chip(time_step, 1).constant(0));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "InverseTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Inverse gradient
	*/
	template<typename TensorT, typename DeviceT>
	class InverseGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		InverseGradTensorOp() {};
		~InverseGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(InverseGradOp<TensorT>());
			auto result = (x.chip(time_step, 1) != x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(-1) / x.chip(time_step, 1).pow(2), x.chip(time_step, 1).constant(0));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "InverseGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Exponential activation function
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		ExponentialTensorOp() {};
		~ExponentialTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ExponentialOp<TensorT>());
			auto result = x.chip(time_step, 1).exp(); 
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "ExponentialTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Exponential gradient
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		ExponentialGradTensorOp() {};
		~ExponentialGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ExponentialGradOp<TensorT>());
			auto result = x.chip(time_step, 1).exp();
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "ExponentialGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Log activation function
	*/
	template<typename TensorT, typename DeviceT>
	class LogTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		LogTensorOp() {};
		~LogTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LogOp<TensorT>());
			auto result = x.chip(time_step, 1).log();
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "LogTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Log gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LogGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		LogGradTensorOp() {};
		~LogGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LogGradOp<TensorT>());
			auto result = x.chip(time_step, 1).constant(1) / x.chip(time_step, 1);
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "LogGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Pow activation function
	*/
	template<typename TensorT, typename DeviceT>
	class PowTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		PowTensorOp() {};
		PowTensorOp(const TensorT& base): base_(base){};
		~PowTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(PowOp<TensorT>(base_));
			auto result = x.chip(time_step, 1).pow(base_);
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "PowTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), base_);
		//}
		TensorT base_;
	};

	/**
	@brief Pow gradient
	*/
	template<typename TensorT, typename DeviceT>
	class PowGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		PowGradTensorOp() {};
		PowGradTensorOp(const TensorT& base) : base_(base) {};
		~PowGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(PowGradOp<TensorT>(base_));
			auto result = x.chip(time_step, 1).constant(base_) * x.chip(time_step, 1).pow(base_ - 1);
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "PowGradTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), base_);
		//}
		TensorT base_;
	};

	/**
		@brief LeakyReLU activation function

		default alpha = 1e-2
	*/
	template<typename TensorT, typename DeviceT>
	class LeakyReLUTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		LeakyReLUTensorOp() {};
		LeakyReLUTensorOp(const TensorT& alpha) : alpha_(alpha) {};
		~LeakyReLUTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LeakyReLUOp<TensorT>(alpha_));
			auto result = (x.chip(time_step, 1) >= x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1), x.chip(time_step, 1) * x.chip(time_step, 1).constant(alpha_));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
		TensorT alpha_ = 1e-2;
	};

	/**
		@brief LeakyReLU gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LeakyReLUGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		LeakyReLUGradTensorOp() {};
		LeakyReLUGradTensorOp(const TensorT& alpha) : alpha_(alpha) {};
		~LeakyReLUGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LeakyReLUGradOp<TensorT>(alpha_));
			auto result = (x.chip(time_step, 1) >= x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(1), x.chip(time_step, 1).constant(alpha_));
			out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUGradTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
		TensorT alpha_ = 1e-2;
	};

	/**
	@brief Sin activation function
	*/
	template<typename TensorT, typename DeviceT>
	class SinTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		SinTensorOp() {};
		~SinTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = x.chip(time_step, 1).sin();
			//out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "SinTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

	/**
	@brief Sin gradient
	*/
	template<typename TensorT, typename DeviceT>
	class SinGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		SinGradTensorOp() {};
		~SinGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = x.chip(time_step, 1).cos();
			//out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "SinGradTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

	/**
	@brief Cos activation function
	*/
	template<typename TensorT, typename DeviceT>
	class CosTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		CosTensorOp() {};
		~CosTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = x.chip(time_step, 1).cos();
			//out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "CosTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

	/**
	@brief Cos gradient
	*/
	template<typename TensorT, typename DeviceT>
	class CosGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		CosGradTensorOp() {};
		~CosGradTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = -x.chip(time_step, 1).sin();
			//out.chip(time_step, 1).device(device) = result.clip(-1e9, 1e9);
		};
		std::string getName() const { return "CosGradTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};
}
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosGradTensorOp<float, Eigen::DefaultDevice>);
//
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosGradTensorOp<float, Eigen::GpuDevice>);
//#endif
//
//// TODO: double, int, etc.,

#endif //SMARTPEAK_ACTIVATIONTENSORFUNCTION_H