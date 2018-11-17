/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONTENSORFUNCTION_H
#define SMARTPEAK_ACTIVATIONTENSORFUNCTION_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <unsupported/Eigen/CXX11/Tensor>

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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(x.chip(time_step, 1), x.chip(time_step, 1).constant(0));
		};
    std::string getName() const{return "ReLUTensorOp";};
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(x.chip(time_step, 1).constant(1), x.chip(time_step, 1).constant(0));
		};
    std::string getName() const{return "ReLUGradTensorOp";};
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1), 
				x.chip(time_step, 1).constant(alpha_) * (x.chip(time_step, 1).exp() - x.chip(time_step, 1).constant(1)));
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUTensorOp";};
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(1),
				(x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(x.chip(time_step, 1), x.chip(time_step, 1).constant(alpha_) * (x.chip(time_step, 1).exp() - x.chip(time_step, 1).constant(1))) + x.chip(time_step, 1).constant(alpha_));
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradTensorOp";};
private:
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(1)/(x.chip(time_step, 1).constant(1) + (-x.chip(time_step, 1).exp()));
		};
    std::string getName() const{return "SigmoidTensorOp";};
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
			auto sigmoid = x.chip(time_step, 1).constant(1) / (x.chip(time_step, 1).constant(1) + (-x.chip(time_step, 1).exp()));
			out.chip(time_step, 1).device(device) = sigmoid * (x.chip(time_step, 1).constant(1) - sigmoid);
		};
    std::string getName() const{return "SigmoidGradTensorOp";};
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).tanh();
		};
    std::string getName() const{return "TanHTensorOp";};
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(1) - (x.chip(time_step, 1).tanh()).pow(2);
		};
    std::string getName() const{return "TanHGradTensorOp";};
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1);
		};
		std::string getName() const { return "LinearTensorOp"; };
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) != x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(1)/ x.chip(time_step, 1), x.chip(time_step, 1).constant(0));
		};
		std::string getName() const { return "InverseTensorOp"; };
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) != x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(-1) / x.chip(time_step, 1).pow(2), x.chip(time_step, 1).constant(0));
		};
		std::string getName() const { return "InverseGradTensorOp"; };
	};

	/**
	@brief Exponential activation function
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
	public:
		ExponentialTensorOp() {};
		ExponentialTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max) {
			this->setEps(eps);
			this->setMin(min);
			this->setMax(max);
		};
		~ExponentialTensorOp() {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ExponentialOp<TensorT>());
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).exp();
		};
		std::string getName() const { return "ExponentialTensorOp"; };
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).exp();
		};
		std::string getName() const { return "ExponentialGradTensorOp"; };
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).log();
		};
		std::string getName() const { return "LogTensorOp"; };
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(1) / x.chip(time_step, 1);
		};
		std::string getName() const { return "LogGradTensorOp"; };
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).pow(base_);
		};
		std::string getName() const { return "PowTensorOp"; };
	private:
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
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(base_) * x.chip(time_step, 1).pow(base_ - 1);
		};
		std::string getName() const { return "PowGradTensorOp"; };
	private:
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1), x.chip(time_step, 1) * x.chip(time_step, 1).constant(alpha_));
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUTensorOp"; };
	private:
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
			out.chip(time_step, 1).device(device) = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(0)).select(
				x.chip(time_step, 1).constant(1), x.chip(time_step, 1).constant(alpha_));
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUGradTensorOp"; };
	private:
		TensorT alpha_ = 1e-2;
	};
}
#endif //SMARTPEAK_ACTIVATIONTENSORFUNCTION_H