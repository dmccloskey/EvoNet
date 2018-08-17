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
		void setMaxNode(const int& max_node) { max_node_ = max_node; }
		int getMaxNode() const { return max_node_; }
		virtual void initNetNodeInput(const int& batch_size) = 0;
		void setNetNodeInput(const Eigen::Tensor<T, 1>& net_node_input) { net_node_input_ = net_node_input; }
		Eigen::Tensor<T, 1> getNetNodeInput() const { return net_node_input_; }
    virtual std::string getName() const = 0;
    virtual void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) = 0;
	protected:
		int max_node_ = -1; ///< node that is the max
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
			setNetNodeInput(net_node_input);
		}
    ~SumOp(){};
    void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { net_node_input_ += weight * source_output; };
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
			setNetNodeInput(net_node_input);
		}
		~ProdOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { net_node_input_ *= weight * source_output; };
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
			setNetNodeInput(net_node_input);
		}
		~MaxOp() {};
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) { net_node_input_ = net_node_input_.cwiseMax(weight * source_output); };
		std::string getName() const { return "MaxOp"; };
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
		void setMaxNode(const int& max_node) { max_node_ = max_node; }
		int getMaxNode() const { return max_node_; }
		virtual void initNetNodeError(const int& batch_size) = 0;
		void setNetNodeError(const Eigen::Tensor<T, 1>& net_node_error) { net_node_error_ = net_node_error; }
		Eigen::Tensor<T, 1> getNetNodeError() const { return net_node_error_; }
		virtual std::string getName() const = 0;
		virtual void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output) = 0;
	protected:
		int max_node_ = -1; ///< node that is the max
		Eigen::Tensor<T, 1> net_node_error_; ///< 
		//std::atomic<Eigen::Tensor<T, 1>> net_node_error_; ///< 
	};

	/**
	@brief Sum integration error function
	*/
	template<typename T>
	class SumErrorOp : public IntegrationErrorOp<T>
	{
	public:
		SumErrorOp() {};
		void initNetNodeError(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_error(batch_size);
			net_node_error.setConstant(0);
			setNetNodeError(net_node_error);
		}
		~SumErrorOp() {};
		/*
		@brief Sum integration error void operator		

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		*/
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output) {
			net_node_error_ += weight * source_error;
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
		void initNetNodeError(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_error(batch_size);
			net_node_error.setConstant(0);
			setNetNodeError(net_node_error);
		}
		~ProdErrorOp() {};
		/*
		@brief Sum integration error void operator

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		*/
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output) {
			net_node_error_ += (source_net_input * source_error / sink_output).unaryExpr(std::ptr_fun(substituteNanInf<T>)); // Note: was checkNanInf
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
		void initNetNodeError(const int& batch_size) {
			Eigen::Tensor<T, 1> net_node_error(batch_size);
			net_node_error.setConstant(0);
			setNetNodeError(net_node_error);
		}
		~MaxErrorOp() {};
		/*
		@brief Sum integration error void operator

		@param[in] x1 The weight tensor
		@param[in] x2 The source error tensor
		@param[in] x3 The source net input tensor
		@param[in] x4 The sink output tensor
		*/
		void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_error, const Eigen::Tensor<T, 1>& source_net_input, const Eigen::Tensor<T, 1>& sink_output){
			auto max_tensor = sink_output.cwiseMax(source_net_input);
			auto perc_max_tensor = (sink_output / max_tensor).unaryExpr(std::ptr_fun(checkNanInf<T>)).unaryExpr([](const T& v) {
				if (v < 1) return 0;
				else return 1;
			});
			net_node_error_ += weight * source_error * perc_max_tensor;
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
		SumWeightGradOp() { setNetWeightError(T(0)); };
		~SumWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
			net_weight_error_ += derivative_mean_tensor(0);
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
		ProdWeightGradOp() { setNetWeightError(T(0)); };
		~ProdWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = ((-sink_error * source_net_input / weight).unaryExpr(std::ptr_fun(substituteNanInf<T>))).mean(); // average derivative
			net_weight_error_ += derivative_mean_tensor(0);
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
		MaxWeightGradOp() { setNetWeightError(T(0)); };
		~MaxWeightGradOp() {};
		void operator()(const Eigen::Tensor<T, 1>& sink_error, const Eigen::Tensor<T, 1>& source_output, const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>& source_net_input) {
			Eigen::Tensor<T, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
			net_weight_error_ += derivative_mean_tensor(0);
		};
		std::string getName() const { return "MaxWeightGradOp"; };
	};
}
#endif //SMARTPEAK_INTEGRATIONFUNCTION_H