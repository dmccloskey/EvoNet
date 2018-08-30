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
    IntegrationOp(){};  
    ~IntegrationOp(){};
		virtual void initNetNodeInput(const int& batch_size) = 0;
		void setNetNodeInput(const Eigen::Tensor<T, 1>& net_node_input) { net_node_input_ = net_node_input; }
		Eigen::Tensor<T, 1> getNetNodeInput() const { return net_node_input_; }
    virtual std::string getName() const = 0;
    virtual void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) = 0;
	protected:
		Eigen::Tensor<T, 1> net_node_input_; ///< 
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
		}
    ~SumOp(){};
    void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { this->net_node_input_ += weight * source_output; };
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
		}
		~ProdOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { this->net_node_input_ *= weight * source_output; };
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
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
		}
		~MaxOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { this->net_node_input_ = this->net_node_input_.cwiseMax(weight * source_output); };
		std::string getName() const { return "MaxOp"; };
	};

	/**
		@brief Mean integration function

		[TODO: add tests]
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
			n_ = 0;
		}
		~MeanOp() {};
		Eigen::Tensor<T, 1> getNetNodeInput() const { return this->net_node_input_/n_; }
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { 
			this->net_node_input_ += weight * source_output; 
			++n_;
		};
		std::string getName() const { return "MeanOp"; };
	private:
		int n_ = 0;
	};

	/**
		@brief Count integration function

		[TODO: add tests and associated integration classes]
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
		IntegrationErrorOp() {};
		~IntegrationErrorOp() {};
		virtual std::string getName() const = 0;
		virtual Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output) = 0;
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
		/*
		@brief Sum integration error void operator		

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		*/
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output) {
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
		/*
		@brief Sum integration error void operator

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		*/
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output) {
			return (source_net_input * source_error / sink_output).unaryExpr(std::ptr_fun(substituteNanInf<T>)); // Note: was checkNanInf
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
		/*
		@brief Sum integration error void operator

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		*/
		Eigen::Tensor<T, 1> operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output){
			//std::cout << "Source net input: " << source_net_input << std::endl;
			//std::cout << "Sink output: " << sink_output << std::endl;
			auto perc_max_tensor = (sink_output / source_net_input).unaryExpr(std::ptr_fun(checkNanInf<T>)).unaryExpr([](const T& v) {
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
	@brief Base class for all integration error functions.
	*/
	template<typename T>
	class IntegrationWeightGradOp
	{
	public:
		IntegrationWeightGradOp() {};
		~IntegrationWeightGradOp() {};
		void initNetWeightError() { net_weight_error_ = 0; }
		void setNetWeightError(const T& net_weight_error) { net_weight_error_ = net_weight_error; }
		T getNetWeightError() const { return net_weight_error_; }
		virtual std::string getName() const = 0;
		virtual void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) = 0;
	protected:
		T net_weight_error_ = 0; ///< 
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
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) {
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
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = ((-sink_error * source_net_input / weight).unaryExpr(std::ptr_fun(substituteNanInf<T>))).mean(); // average derivative
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
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
			this->net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "MaxWeightGradOp"; };
	};
}
#endif //SMARTPEAK_INTEGRATIONFUNCTION_H