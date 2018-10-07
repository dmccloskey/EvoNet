/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODEDATA_H
#define SMARTPEAK_NODEDATA_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace SmartPeak
{
  /**
    @brief Network NodeData
  */
	template<typename TensorT>
  class NodeData
  {
public:
    NodeData() = default; ///< Default constructor
    NodeData(const NodeData& other)
		{
			input_ = other.input_;
			output_ = other.output_;
			error_ = other.error_;
			derivative_ = other.derivative_;
			dt_ = other.dt_;
			h_input_ = other.h_input_;
			h_output_ = other.h_output_;
			h_error_ = other.h_error_;
			h_derivative_ = other.h_derivative_;
			h_dt_ = other.h_dt_;
			d_input_ = other.d_input_;
			d_output_ = other.d_output_;
			d_error_ = other.d_error_;
			d_derivative_ = other.d_derivative_;
			d_dt_ = other.d_dt_;
		};
    ~NodeData() = default; ///< Default destructor

    inline bool operator==(const NodeData& other) const
    {
      return
        std::tie(
       
        ) == std::tie(

        )
      ;
    }

    inline bool operator!=(const NodeData& other) const
    {
      return !(*this == other);
    }

    inline NodeData& operator=(const NodeData& other)
    { 
			input_ = other.input_;
      output_ = other.output_;
      error_ = other.error_;
      derivative_ = other.derivative_;
      dt_ = other.dt_;
			h_input_ = other.h_input_;
			h_output_ = other.h_output_;
			h_error_ = other.h_error_;
			h_derivative_ = other.h_derivative_;
			h_dt_ = other.h_dt_;
			d_input_ = other.d_input_;
			d_output_ = other.d_output_;
			d_error_ = other.d_error_;
			d_derivative_ = other.d_derivative_;
			d_dt_ = other.d_dt_;
      return *this;
    }

		void setBatchSize(const size_t& batch_size) { batch_size_ = batch_size; }
		void setMemorySize(const size_t& memory_size) { memory_size_ = memory_size; }
		size_t getBatchSize() const { return batch_size_; }
		size_t getMemorySize() const	{ return memory_size_; }

		virtual void setInput(TensorT* input) = 0; ///< input setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getInput() const { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> input(h_output_.get(), batch_size_, memory_size_); return input; }; ///< input copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getInputMutable() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> input(h_output_.get(), batch_size_, memory_size_); return input; }; ///< input copy getter
		std::shared_ptr<TensorT> getHInputPointer() { return h_input_; }; ///< input pointer getter
		std::shared_ptr<TensorT> getDInputPointer() { return d_input_; }; ///< input pointer getter

    virtual void setOutput(TensorT* output) = 0; ///< output setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getOutput() const { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> output(h_output_.get(), batch_size_, memory_size_); return output; }; ///< output copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getOutputMutable() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> output(h_output_.get(), batch_size_, memory_size_); return output; }; ///< output copy getter
		std::shared_ptr<TensorT> getHOutputPointer() { return h_output_; }; ///< output pointer getter
		std::shared_ptr<TensorT> getDOutputPointer() { return d_output_; }; ///< output pointer getter

    virtual void setError(TensorT* error) = 0; ///< error setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getError() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error(h_error_.get(), batch_size_, memory_size_); return error; }; ///< error copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getErrorMutable() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error(h_error_.get(), batch_size_, memory_size_); return error; }; ///< error copy getter
		std::shared_ptr<TensorT> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<TensorT> getDErrorPointer() { return d_error_; }; ///< error pointer getter

    virtual void setDerivative(TensorT* derivative) = 0; ///< derivative setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getDerivative() const; ///< derivative copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getDerivativeMutable() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> derivative(h_derivative_, batch_size_, memory_size_); return derivative; }; ///< derivative copy getter
		std::shared_ptr<TensorT> getHDerivativePointer() { return h_derivative_; }; ///< derivative pointer getter
		std::shared_ptr<TensorT> getDDerivativePointer() { return d_derivative_; }; ///< derivative pointer getter

    virtual void setDt(TensorT* dt) = 0; ///< dt setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getDt() const; ///< dt copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getDtMutable() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> dt(h_dt_, batch_size_, memory_size_); return dt;	}; ///< dt copy getter
		std::shared_ptr<TensorT> getHDtPointer() { return h_dt_; }; ///< dt pointer getter
		std::shared_ptr<TensorT> getDDtPointer() { return d_dt_; }; ///< dt pointer getter

protected:
		size_t batch_size_ = 1; ///< Mini batch size
		size_t memory_size_ = 2; ///< Memory size
    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */		
		std::shared_ptr<TensorT> h_input_;
		std::shared_ptr<TensorT> h_output_;
		std::shared_ptr<TensorT> h_error_;
		std::shared_ptr<TensorT> h_derivative_;
		std::shared_ptr<TensorT> h_dt_;
		std::shared_ptr<TensorT> d_input_;
		std::shared_ptr<TensorT> d_output_;
		std::shared_ptr<TensorT> d_error_;
		std::shared_ptr<TensorT> d_derivative_;
		std::shared_ptr<TensorT> d_dt_;
  };
}

#endif //SMARTPEAK_NODEDATA_H