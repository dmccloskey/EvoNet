/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTEGRATIONFUNCTION_H
#define SMARTPEAK_INTEGRATIONFUNCTION_H

#include <SmartPeak/ml/SharedFunctions.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  /**
    @brief Base class for all integration functions.
  */
	template<typename TensorT, typename KernalT>
  class IntegrationOp
  {
public: 
    IntegrationOp() = default;
		IntegrationOp(const TensorT& eps) : eps_(eps) {};
    ~IntegrationOp() = default;
    virtual std::string getName() const = 0;
    virtual void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, KernalT* kernal) = 0;
	protected:
		TensorT eps_ = 1e-9;
  };

  /**
    @brief Sum integration function
  */
  template<typename TensorT>
  class SumOp: public IntegrationOp<TensorT>
  {
public: 
		SumOp(){};
    ~SumOp(){};
    void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, KernalT* kernal) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input(sink_input, batch_size, memory_size);
			sink_input.device(kernal->getDevice()).chip(sink_time_step, 1).setConstant(0);
			for (int i = 0; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
				sink_input.device(kernal->getDevice()).chip(sink_time_step, 1) += source_output.chip(source_time_steps[i], 1) * weight;
			}
		};
    std::string getName() const{return "SumOp";};
  };

	/**
	@brief Product integration function
	*/
	template<typename TensorT>
	class ProdOp : public IntegrationOp<TensorT>
	{
	public:
		ProdOp() {};
		~ProdOp() {};
		void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, KernalT* kernal) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input(sink_input, batch_size, memory_size);
			sink_input.device(kernal->getDevice()).chip(sink_time_step, 1).setConstant(1);
			for (int i = 0; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
				sink_input.device(kernal->getDevice()).chip(sink_time_step, 1) *= source_output.chip(source_time_steps[i], 1) * weight;
			}
		};
		std::string getName() const { return "ProdOp"; };
	};

	///**
	//@brief Max integration function
	//*/
	//template<typename TensorT>
	//class MaxOp : public IntegrationOp<TensorT>
	//{
	//public:
	//	MaxOp() {};
	//	void initNetNodeInput(const int& batch_size) {
	//		Eigen::Tensor<TensorT, 1> net_node_input(batch_size);
	//		net_node_input.setConstant(-1e12);
	//		this->setNetNodeInput(net_node_input);
	//		this->n_ = 0;
	//	}
	//	~MaxOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>&source_output) { 
	//		this->net_node_input_ = this->net_node_input_.cwiseMax(weight * source_output);
	//		this->n_ += 1;
	//	};
	//	std::string getName() const { return "MaxOp"; };
	//};

	///**
	//	@brief Mean integration function
	//*/
	//template<typename TensorT>
	//class MeanOp : public IntegrationOp<TensorT>
	//{
	//public:
	//	MeanOp() {};
	//	void initNetNodeInput(const int& batch_size) {
	//		Eigen::Tensor<TensorT, 1> net_node_input(batch_size);
	//		net_node_input.setConstant(0);
	//		this->setNetNodeInput(net_node_input);
	//		this->n_ = 0;
	//	}
	//	~MeanOp() {};
	//	Eigen::Tensor<TensorT, 1> getNetNodeInput() const {
	//		Eigen::Tensor<TensorT, 1> n(this->net_node_input_.dimension(0));
	//		n.setConstant(this->n_);
	//		return this->net_node_input_/n; 
	//	}
	//	void operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>&source_output) { 
	//		this->net_node_input_ += weight * source_output;
	//		this->n_ += 1;
	//	};
	//	std::string getName() const { return "MeanOp"; };
	//};

	///**
	//	@brief Variance integration function

	//	References:
	//	TensorT.F.Chan, G.H. Golub and R.J. LeVeque (1983). ""Algorithms for computing the sample variance: Analysis and recommendations", The American Statistician, 37": 242–247.
	//*/
	//template<typename TensorT>
	//class VarianceOp : public IntegrationOp<TensorT>
	//{
	//public:
	//	VarianceOp() {};
	//	void initNetNodeInput(const int& batch_size) {
	//		Eigen::Tensor<TensorT, 1> net_node_input(batch_size);
	//		net_node_input.setConstant(0);
	//		this->setNetNodeInput(net_node_input);
	//		this->n_ = 0;
	//	}
	//	~VarianceOp() {};
	//	Eigen::Tensor<TensorT, 1> getNetNodeInput() const { 
	//		Eigen::Tensor<TensorT, 1> n(this->net_node_input_.dimension(0));
	//		n.setConstant(this->n_); 
	//		return (this->net_node_input_  - (ex_ * ex_)/ n)/n; }
	//	void operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>&source_output) {
	//		auto input = weight * source_output;
	//		if (this->n_ == 0)
	//			k_ = input;
	//		auto input_k = input - k_;
	//		ex_ += input_k;
	//		this->n_ += 1;
	//		this->net_node_input_ += (input_k * input_k);
	//	};
	//	std::string getName() const { return "VarianceOp"; };
	//private:
	//	Eigen::Tensor<TensorT, 1> k_ = 0;
	//	Eigen::Tensor<TensorT, 1> ex_ = 0;
	//};

	///**
	//	@brief VarMod integration function

	//	Modified variance integration function: 1/n Sum[0 to n](Xi)^2
	//	where Xi = xi - u (u: mean, xi: single sample)
	//*/
	//template<typename TensorT>
	//class VarModOp : public IntegrationOp<TensorT>
	//{
	//public:
	//	VarModOp() {};
	//	void initNetNodeInput(const int& batch_size) {
	//		Eigen::Tensor<TensorT, 1> net_node_input(batch_size);
	//		net_node_input.setConstant(0);
	//		this->setNetNodeInput(net_node_input);
	//		this->n_ = 0;
	//	}
	//	~VarModOp() {};
	//	Eigen::Tensor<TensorT, 1> getNetNodeInput() const {
	//		Eigen::Tensor<TensorT, 1> n(this->net_node_input_.dimension(0));
	//		n.setConstant(this->n_);
	//		return this->net_node_input_ / n; }
	//	void operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>&source_output) {
	//		auto input = weight * source_output;
	//		this->n_ += 1;
	//		this->net_node_input_ += (input * input);
	//	};
	//	std::string getName() const { return "VarModOp"; };
	//};

	///**
	//	@brief Count integration function
	//*/
	//template<typename TensorT>
	//class CountOp : public IntegrationOp<TensorT>
	//{
	//public:
	//	CountOp() {};
	//	void initNetNodeInput(const int& batch_size) {
	//		Eigen::Tensor<TensorT, 1> net_node_input(batch_size);
	//		net_node_input.setConstant(0);
	//		this->setNetNodeInput(net_node_input);
	//	}
	//	~CountOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>&source_output) { 
	//		Eigen::Tensor<TensorT, 1> one(source_output.dimension(0));
	//		one.setConstant(1.0f);
	//		this->net_node_input_ += one; 
	//		++(this->n_);
	//	};
	//	std::string getName() const { return "CountOp"; };
	//};

	///**
	//@brief Base class for all integration error functions.
	//*/
	//template<typename TensorT>
	//class IntegrationErrorOp
	//{
	//public:
	//	IntegrationErrorOp() = default;
	//	IntegrationErrorOp(const TensorT& eps) : eps_(eps) {};
	//	~IntegrationErrorOp() = default;
	//	virtual std::string getName() const = 0;
	//	/*
	//	@brief Sum integration error void operator

	//	@param[in] x1 The weight tensor
	//	@param[in] x2 The source error tensor
	//	@param[in] x3 The source net input tensor
	//	@param[in] x4 The sink output tensor
	//	@param[in] x5 The number of inputs
	//	*/
	//	virtual Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) = 0;
	//protected:
	//	TensorT eps_ = 1e-6;
	//};

	///**
	//@brief Sum integration error function
	//*/
	//template<typename TensorT>
	//class SumErrorOp : public IntegrationErrorOp<TensorT>
	//{
	//public:
	//	SumErrorOp() {};
	//	~SumErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		return weight * source_error;
	//	};
	//	std::string getName() const { return "SumErrorOp"; };
	//};

	///**
	//@brief Product integration error function
	//*/
	//template<typename TensorT>
	//class ProdErrorOp : public IntegrationErrorOp<TensorT>
	//{
	//public:
	//	ProdErrorOp() {};
	//	~ProdErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> eps(weight.dimension(0));
	//		eps.setConstant(this->eps_);
	//		return (source_net_input * source_error / (sink_output + eps)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9)); // .unaryExpr(std::ptr_fun(checkNan<TensorT>));
	//		//return (source_net_input * source_error / sink_output).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9)).unaryExpr(std::ptr_fun(checkNan<TensorT>));
	//	};
	//	std::string getName() const { return "ProdErrorOp"; };
	//};

	///**
	//@brief Max integration error function
	//*/
	//template<typename TensorT>
	//class MaxErrorOp : public IntegrationErrorOp<TensorT>
	//{
	//public:
	//	MaxErrorOp() {};
	//	~MaxErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n)
	//	{
	//		auto perc_max_tensor = (sink_output / source_net_input).unaryExpr([](const TensorT& v) {
	//			if (v < 1 - 1e-6) 
	//				return 0;
	//			else 
	//				return 1;
	//		});
	//		return weight * source_error * perc_max_tensor;
	//	};
	//	std::string getName() const { return "MaxErrorOp"; };
	//};

	///**
	//@brief Mean integration error function
	//*/
	//template<typename TensorT>
	//class MeanErrorOp : public IntegrationErrorOp<TensorT>
	//{
	//public:
	//	MeanErrorOp() {};
	//	~MeanErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		return weight * source_error / n;
	//	};
	//	std::string getName() const { return "MeanErrorOp"; };
	//};

	///**
	//@brief VarMod integration error function
	//*/
	//template<typename TensorT>
	//class VarModErrorOp : public IntegrationErrorOp<TensorT>
	//{
	//public:
	//	VarModErrorOp() {};
	//	~VarModErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(2);
	//		return weight * source_error * constant / n;
	//	};
	//	std::string getName() const { return "VarModErrorOp"; };
	//};

	///**
	//@brief Count integration error function
	//*/
	//template<typename TensorT>
	//class CountErrorOp : public IntegrationErrorOp<TensorT>
	//{
	//public:
	//	CountErrorOp() {};
	//	~CountErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(0);
	//		return constant;
	//	};
	//	std::string getName() const { return "CountErrorOp"; };
	//};

	///**
	//@brief Base class for all integration error functions.
	//*/
	//template<typename TensorT>
	//class IntegrationWeightGradOp
	//{
	//public:
	//	IntegrationWeightGradOp() = default;
	//	IntegrationWeightGradOp(const TensorT& eps) : eps_(eps) {};
	//	~IntegrationWeightGradOp() = default;
	//	void initNetWeightError() { net_weight_error_ = 0; }
	//	void setNetWeightError(const TensorT& net_weight_error) { net_weight_error_ = net_weight_error; }
	//	TensorT getNetWeightError() const { return net_weight_error_; }
	//	virtual std::string getName() const = 0;
	//	virtual void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) = 0;
	//protected:
	//	TensorT net_weight_error_ = 0; ///< 
	//	TensorT eps_ = 1e-6;
	//	//std::atomic<TensorT> net_weight_error_ = 0; ///< 
	//};

	///**
	//@brief Sum integration error function
	//*/
	//template<typename TensorT>
	//class SumWeightGradOp : public IntegrationWeightGradOp<TensorT>
	//{
	//public:
	//	SumWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~SumWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "SumWeightGradOp"; };
	//};

	///**
	//@brief Product integration error function
	//*/
	//template<typename TensorT>
	//class ProdWeightGradOp : public IntegrationWeightGradOp<TensorT>
	//{
	//public:
	//	ProdWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~ProdWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = ((-sink_error * source_net_input / weight).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9))).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "ProdWeightGradOp"; };
	//};

	///**
	//@brief Max integration error function
	//*/
	//template<typename TensorT>
	//class MaxWeightGradOp : public IntegrationWeightGradOp<TensorT>
	//{
	//public:
	//	MaxWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~MaxWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "MaxWeightGradOp"; };
	//};

	///**
	//@brief Count integration error function
	//*/
	//template<typename TensorT>
	//class CountWeightGradOp : public IntegrationWeightGradOp<TensorT>
	//{
	//public:
	//	CountWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~CountWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		this->net_weight_error_ += 0;
	//	};
	//	std::string getName() const { return "CountWeightGradOp"; };
	//};

	///**
	//@brief Mean integration error function
	//*/
	//template<typename TensorT>
	//class MeanWeightGradOp : public IntegrationWeightGradOp<TensorT>
	//{
	//public:
	//	MeanWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~MeanWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output / n).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "MeanWeightGradOp"; };
	//};

	///**
	//@brief VarMod integration error function
	//*/
	//template<typename TensorT>
	//class VarModWeightGradOp : public IntegrationWeightGradOp<TensorT>
	//{
	//public:
	//	VarModWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~VarModWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(2);
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output * constant / n).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "VarModWeightGradOp"; };
	//};
}
#endif //SMARTPEAK_INTEGRATIONFUNCTION_H