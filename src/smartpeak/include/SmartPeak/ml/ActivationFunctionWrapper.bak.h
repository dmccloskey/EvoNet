/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONFUNCTIONWRAPPER_H
#define SMARTPEAK_ACTIVATIONFUNCTIONWRAPPER_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  /**
    @brief Base class for all activation function wrappers.
  */
 template<typename TensorT, typename DeviceT>
  class ActivationOpWrapper
  {
public: 
	ActivationOpWrapper() {};
		~ActivationOpWrapper() {};
		virtual std::string getName() const = 0;
		virtual void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const = 0;
  };

  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT, typename DeviceT>
  class ReLUOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    ReLUOpWrapper(){}; 
    ~ReLUOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReLUOp<TensorT>());
		};
    std::string getName() const{return "ReLUOpWrapper";};
  };

  /**
    @brief Rectified Linear Unit (ReLU) gradient

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT, typename DeviceT>
  class ReLUGradOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    ReLUGradOpWrapper(){}; 
    ~ReLUGradOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReLUGradOp<TensorT>());
		};
    std::string getName() const{return "ReLUGradOpWrapper";};
  };

  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT, typename DeviceT>
  class ELUOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    ELUOpWrapper(){}; 
    ELUOpWrapper(const TensorT& alpha): alpha_(alpha){}; 
    ~ELUOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ELUOp<TensorT>(alpha_));
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUOpWrapper";};
private:
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
  class ELUGradOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    ELUGradOpWrapper(){}; 
    ELUGradOpWrapper(const TensorT& alpha): alpha_(alpha){}; 
    ~ELUGradOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ELUGradOp<TensorT>(alpha_));
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradOpWrapper";};
private:
    TensorT alpha_ = 1;
  };

  /**
    @brief Sigmoid activation function
  */
  template<typename TensorT, typename DeviceT>
  class SigmoidOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    SigmoidOpWrapper(){}; 
    ~SigmoidOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(SigmoidOp<TensorT>());
		};
    std::string getName() const{return "SigmoidOpWrapper";};
  };

  /**
    @brief Sigmoid gradient
  */
  template<typename TensorT, typename DeviceT>
  class SigmoidGradOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    SigmoidGradOpWrapper(){}; 
    ~SigmoidGradOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(SigmoidGradOp<TensorT>());
		};
    std::string getName() const{return "SigmoidGradOpWrapper";};
  };
  
  /**
    @brief Hyperbolic Tangent activation function
  */
  template<typename TensorT, typename DeviceT>
  class TanHOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    TanHOpWrapper(){}; 
    ~TanHOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(TanHOp<TensorT>());
		};
    std::string getName() const{return "TanHOpWrapper";};
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename TensorT, typename DeviceT>
  class TanHGradOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    TanHGradOpWrapper(){}; 
    ~TanHGradOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(TanHGradOp<TensorT>());
		};
    std::string getName() const{return "TanHGradOpWrapper";};
  };
  
  /**
    @brief Rectified Hyperbolic Tangent activation function
  */
  template<typename TensorT, typename DeviceT>
  class ReTanHOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    ReTanHOpWrapper(){}; 
    ~ReTanHOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReTanHOp<TensorT>());
		};
    std::string getName() const{return "ReTanHOpWrapper";};
  };

  /**
    @brief Rectified Hyperbolic Tangent gradient
  */
  template<typename TensorT, typename DeviceT>
  class ReTanHGradOpWrapper: public ActivationOpWrapper<TensorT, DeviceT>
  {
public: 
    ReTanHGradOpWrapper(){}; 
    ~ReTanHGradOpWrapper(){};
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReTanHGradOp<TensorT>());
		};
    std::string getName() const{return "ReTanHGradOpWrapper";};
  };

	/**
	@brief Linear activation function
	*/
	template<typename TensorT, typename DeviceT>
	class LinearOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		LinearOpWrapper() {};
		~LinearOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LinearOp<TensorT>());
		};
		std::string getName() const { return "LinearOpWrapper"; };
	};

	/**
	@brief Linear gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LinearGradOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		LinearGradOpWrapper() {};
		~LinearGradOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LinearGradOp<TensorT>());
		};
		std::string getName() const { return "LinearGradOpWrapper"; };
	};

	/**
	@brief Inverse activation function
	*/
	template<typename TensorT, typename DeviceT>
	class InverseOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		InverseOpWrapper() {};
		~InverseOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(InverseOp<TensorT>());
		};
		std::string getName() const { return "InverseOpWrapper"; };
	};

	/**
	@brief Inverse gradient
	*/
	template<typename TensorT, typename DeviceT>
	class InverseGradOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		InverseGradOpWrapper() {};
		~InverseGradOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(InverseGradOp<TensorT>());
		};
		std::string getName() const { return "InverseGradOpWrapper"; };
	};

	/**
	@brief Exponential activation function
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		ExponentialOpWrapper() {};
		ExponentialOpWrapper(const TensorT& eps, const TensorT& min, const TensorT& max) {
			this->setEps(eps);
			this->setMin(min);
			this->setMax(max);
		};
		~ExponentialOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ExponentialOp<TensorT>());
		};
		std::string getName() const { return "ExponentialOpWrapper"; };
	};

	/**
	@brief Exponential gradient
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialGradOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		ExponentialGradOpWrapper() {};
		~ExponentialGradOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ExponentialGradOp<TensorT>());
		};
		std::string getName() const { return "ExponentialGradOpWrapper"; };
	};

	/**
	@brief Log activation function
	*/
	template<typename TensorT, typename DeviceT>
	class LogOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		LogOpWrapper() {};
		~LogOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LogOp<TensorT>());
		};
		std::string getName() const { return "LogOpWrapper"; };
	};

	/**
	@brief Log gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LogGradOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		LogGradOpWrapper() {};
		~LogGradOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LogGradOp<TensorT>());
		};
		std::string getName() const { return "LogGradOpWrapper"; };
	};

	/**
	@brief Pow activation function
	*/
	template<typename TensorT, typename DeviceT>
	class PowOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		PowOpWrapper(const TensorT& base): base_(base){};
		~PowOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(PowOp<TensorT>(base_));
		};
		std::string getName() const { return "PowOpWrapper"; };
	private:
		TensorT base_;
	};

	/**
	@brief Pow gradient
	*/
	template<typename TensorT, typename DeviceT>
	class PowGradOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		PowGradOpWrapper(const TensorT& base) : base_(base) {};
		~PowGradOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(PowGradOp<TensorT>(base_));
		};
		std::string getName() const { return "PowGradOpWrapper"; };
	private:
		TensorT base_;
	};

	/**
		@brief LeakyReLU activation function

		default alpha = 1e-2
	*/
	template<typename TensorT, typename DeviceT>
	class LeakyReLUOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		LeakyReLUOpWrapper() {};
		LeakyReLUOpWrapper(const TensorT& alpha) : alpha_(alpha) {};
		~LeakyReLUOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LeakyReLUOp<TensorT>(alpha_));
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUOpWrapper"; };
	private:
		TensorT alpha_ = 1e-2;
	};

	/**
		@brief LeakyReLU gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LeakyReLUGradOpWrapper : public ActivationOpWrapper<TensorT, DeviceT>
	{
	public:
		LeakyReLUGradOpWrapper() {};
		LeakyReLUGradOpWrapper(const TensorT& alpha) : alpha_(alpha) {};
		~LeakyReLUGradOpWrapper() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x(x_I, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> out(x_O, batch_size, memory_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(LeakyReLUGradOp<TensorT>(alpha_));
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUGradOpWrapper"; };
	private:
		TensorT alpha_ = 1e-2;
	};
}
#endif //SMARTPEAK_ACTIVATIONFUNCTIONWRAPPER_H