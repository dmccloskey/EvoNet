/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTEGRATIONFUNCTION_H
#define SMARTPEAK_INTEGRATIONFUNCTION_H

#include <SmartPeak/ml/SharedFunctions.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <atomic>

namespace SmartPeak
{

  /**
    @brief Base class for all integration functions.
  */
	template<typename T>
  class IntegrationOp
  {
public: 
    IntegrationOp() = default;
		IntegrationOp(const T& eps) : eps_(eps) {};
    ~IntegrationOp() = default;
		virtual void initNetNodeInput(const int& batch_size) = 0;
		void setNetNodeInput(const Eigen::Tensor<T, 1>& net_node_input) { net_node_input_ = net_node_input; }
		Eigen::Tensor<T, 1> getNetNodeInput() const { return net_node_input_; }
    virtual std::string getName() const = 0;
		T getN() { return n_; };
    virtual void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) = 0;
	protected:
		Eigen::Tensor<T, 1> net_node_input_; ///<
		T n_ = 0;
		T eps_ = 1e-6;
		//std::atomic<Eigen::Tensor<T, 1>> net_node_input_; ///< 
  };

  /**
    @brief Sum integration function
  */
  template<typename T>
  class SumOp: public IntegrationOp<T>
  {
public: 
		SumOp(){};
		void initNetNodeInput(const int& batch_size){
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
			this->n_ = 0;
		}
    ~SumOp(){};
    void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { 
			this->net_node_input_ += weight * source_output; 
			this->n_ += 1;
		};
    std::string getName() const{return "SumOp";};
  };

	/**
	@brief Product integration function
	*/
	template<typename T>
	class ProdOp : public IntegrationOp<T>
	{
	public:
		ProdOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(1);
			this->setNetNodeInput(net_node_input);
			this->n_ = 0;
		}
		~ProdOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { 
			this->net_node_input_ *= weight * source_output;
			this->n_ += 1;
		};
		std::string getName() const { return "ProdOp"; };
	};

	/**
	@brief Max integration function
	*/
	template<typename T>
	class MaxOp : public IntegrationOp<T>
	{
	public:
		MaxOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(-1e12);
			this->setNetNodeInput(net_node_input);
			this->n_ = 0;
		}
		~MaxOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { 
			this->net_node_input_ = this->net_node_input_.cwiseMax(weight * source_output);
			this->n_ += 1;
		};
		std::string getName() const { return "MaxOp"; };
	};

	/**
		@brief Mean integration function
	*/
	template<typename T>
	class MeanOp : public IntegrationOp<T>
	{
	public:
		MeanOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
			this->n_ = 0;
		}
		~MeanOp() {};
		Eigen::Tensor<T, 1> getNetNodeInput() const {
			Eigen::Tensor<T, 1> n(this->net_node_input_.dimension(0));
			n.setConstant(this->n_);
			return this->net_node_input_/n; 
		}
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { 
			this->net_node_input_ += weight * source_output;
			this->n_ += 1;
		};
		std::string getName() const { return "MeanOp"; };
	};

	/**
		@brief Variance integration function

		References:
		T.F.Chan, G.H. Golub and R.J. LeVeque (1983). ""Algorithms for computing the sample variance: Analysis and recommendations", The American Statistician, 37": 242–247.
	*/
	template<typename T>
	class VarianceOp : public IntegrationOp<T>
	{
	public:
		VarianceOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
			this->n_ = 0;
		}
		~VarianceOp() {};
		Eigen::Tensor<T, 1> getNetNodeInput() const { 
			Eigen::Tensor<T, 1> n(this->net_node_input_.dimension(0));
			n.setConstant(this->n_); 
			return (this->net_node_input_  - (ex_ * ex_)/ n)/n; }
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) {
			auto input = weight * source_output;
			if (this->n_ == 0)
				k_ = input;
			auto input_k = input - k_;
			ex_ += input_k;
			this->n_ += 1;
			this->net_node_input_ += (input_k * input_k);
		};
		std::string getName() const { return "VarianceOp"; };
	private:
		Eigen::Tensor<T, 1> k_ = 0;
		Eigen::Tensor<T, 1> ex_ = 0;
	};

	/**
		@brief VarMod integration function

		Modified variance integration function: 1/n Sum[0 to n](Xi)^2
		where Xi = xi - u (u: mean, xi: single sample)
	*/
	template<typename T>
	class VarModOp : public IntegrationOp<T>
	{
	public:
		VarModOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
			this->n_ = 0;
		}
		~VarModOp() {};
		Eigen::Tensor<T, 1> getNetNodeInput() const {
			Eigen::Tensor<T, 1> n(this->net_node_input_.dimension(0));
			n.setConstant(this->n_);
			return this->net_node_input_ / n; }
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) {
			auto input = weight * source_output;
			this->n_ += 1;
			this->net_node_input_ += (input * input);
		};
		std::string getName() const { return "VarModOp"; };
	};

	/**
		@brief Count integration function
	*/
	template<typename T>
	class CountOp : public IntegrationOp<T>
	{
	public:
		CountOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_input(batch_size);
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
		}
		~CountOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { 
			Eigen::Tensor<T, 1> one(source_output.dimension(0));
			one.setConstant(1.0f);
			this->net_node_input_ += one; 
			++(this->n_);
		};
		std::string getName() const { return "CountOp"; };
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename T>
	class IntegrationErrorOp
	{
	public:
		IntegrationErrorOp() = default;
		IntegrationErrorOp(const T& eps) : eps_(eps) {};
		~IntegrationErrorOp() = default;
		virtual std::string getName() const = 0;
		/*
		@brief Sum integration error void operator

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		@param[in] x5 The number of inputs
		*/
		virtual Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n) = 0;
	protected:
		T eps_ = 1e-6;
	};

	/**
	@brief Sum integration error function
	*/
	template<typename T>
	class SumErrorOp : public IntegrationErrorOp<T>
	{
	public:
		SumErrorOp() {};
		~SumErrorOp() {};
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n) {
			return weight * source_error;
		};
		std::string getName() const { return "SumErrorOp"; };
	};

	/**
	@brief Product integration error function
	*/
	template<typename T>
	class ProdErrorOp : public IntegrationErrorOp<T>
	{
	public:
		ProdErrorOp() {};
		~ProdErrorOp() {};
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 1> eps(weight.dimension(0));
			eps.setConstant(this->eps_);
			return (source_net_input * source_error / (sink_output + eps)).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9)); // .unaryExpr(std::ptr_fun(checkNan<T>));
		};
		std::string getName() const { return "ProdErrorOp"; };
	};

	/**
	@brief Max integration error function
	*/
	template<typename T>
	class MaxErrorOp : public IntegrationErrorOp<T>
	{
	public:
		MaxErrorOp() {};
		~MaxErrorOp() {};
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n)
		{
			auto perc_max_tensor = (sink_output / source_net_input).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9)).unaryExpr([](const T& v) {
				if (v < 1 - 1e-3) 
					return 0;
				else 
					return 1;
			});
			return weight * source_error * perc_max_tensor;
		};
		std::string getName() const { return "MaxErrorOp"; };
	};

	/**
	@brief Mean integration error function
	*/
	template<typename T>
	class MeanErrorOp : public IntegrationErrorOp<T>
	{
	public:
		MeanErrorOp() {};
		~MeanErrorOp() {};
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n) {
			return weight * source_error / n;
		};
		std::string getName() const { return "MeanErrorOp"; };
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename T>
	class VarModErrorOp : public IntegrationErrorOp<T>
	{
	public:
		VarModErrorOp() {};
		~VarModErrorOp() {};
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 1> constant(weight.dimension(0));
			constant.setConstant(2);
			return weight * source_error * constant / n;
		};
		std::string getName() const { return "VarModErrorOp"; };
	};

	/**
	@brief Count integration error function
	*/
	template<typename T>
	class CountErrorOp : public IntegrationErrorOp<T>
	{
	public:
		CountErrorOp() {};
		~CountErrorOp() {};
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 1> constant(weight.dimension(0));
			constant.setConstant(0);
			return constant;
		};
		std::string getName() const { return "CountErrorOp"; };
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename T>
	class IntegrationWeightGradOp
	{
	public:
		IntegrationWeightGradOp() = default;
		IntegrationWeightGradOp(const T& eps) : eps_(eps) {};
		~IntegrationWeightGradOp() = default;
		void initNetWeightError() { net_weight_error_ = 0; }
		void setNetWeightError(const T& net_weight_error) { net_weight_error_ = net_weight_error; }
		T getNetWeightError() const { return net_weight_error_; }
		virtual std::string getName() const = 0;
		virtual void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) = 0;
	protected:
		T net_weight_error_ = 0; ///< 
		T eps_ = 1e-6;
		//std::atomic<T> net_weight_error_ = 0; ///< 
	};

	/**
	@brief Sum integration error function
	*/
	template<typename T>
	class SumWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		SumWeightGradOp() { this->setNetWeightError(T(0)); };
		~SumWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
			this->net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "SumWeightGradOp"; };
	};

	/**
	@brief Product integration error function
	*/
	template<typename T>
	class ProdWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		ProdWeightGradOp() { this->setNetWeightError(T(0)); };
		~ProdWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = ((-sink_error * source_net_input / weight).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9))).mean(); // average derivative
			this->net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "ProdWeightGradOp"; };
	};

	/**
	@brief Max integration error function
	*/
	template<typename T>
	class MaxWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		MaxWeightGradOp() { this->setNetWeightError(T(0)); };
		~MaxWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
			this->net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "MaxWeightGradOp"; };
	};

	/**
	@brief Count integration error function
	*/
	template<typename T>
	class CountWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		CountWeightGradOp() { this->setNetWeightError(T(0)); };
		~CountWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) {
			this->net_weight_error_ += 0;
		};
		std::string getName() const { return "CountWeightGradOp"; };
	};

	/**
	@brief Mean integration error function
	*/
	template<typename T>
	class MeanWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		MeanWeightGradOp() { this->setNetWeightError(T(0)); };
		~MeanWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output / n).mean(); // average derivative
			this->net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "MeanWeightGradOp"; };
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename T>
	class VarModWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		VarModWeightGradOp() { this->setNetWeightError(T(0)); };
		~VarModWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& n) {
			Eigen::Tensor<T, 1> constant(weight.dimension(0));
			constant.setConstant(2);
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output * constant / n).mean(); // average derivative
			this->net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "VarModWeightGradOp"; };
	};
}
#endif //SMARTPEAK_INTEGRATIONFUNCTION_H