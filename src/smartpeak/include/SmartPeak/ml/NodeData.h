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
	template<typename HDelT, typename DDelT, typename TensorT>
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

		void setInput(const Eigen::Tensor<TensorT, 2>& input); ///< input setter
		Eigen::Tensor<TensorT, 2> getInput() const { Eigen::Tensor<TensorT, 2> input = input_; return input; }; ///< input copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getInputMutable() { return input_; }; ///< input copy getter
		std::shared_ptr<HDelT> getHInputPointer() { return h_input_; }; ///< input pointer getter
		std::shared_ptr<DDelT> getDInputPointer() { return d_input_; }; ///< input pointer getter

    void setOutput(const Eigen::Tensor<TensorT, 2>& output); ///< output setter
		Eigen::Tensor<TensorT, 2> getOutput() const { Eigen::Tensor<TensorT, 2> output = output_; return output; }; ///< output copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getOutputMutable() { return output_; }; ///< output copy getter
		std::shared_ptr<HDelT> getHOutputPointer() { return h_output_; }; ///< output pointer getter
		std::shared_ptr<DDelT> getDOutputPointer() { return d_output_; }; ///< output pointer getter

    void setError(const Eigen::Tensor<TensorT, 2>& error); ///< error setter
		Eigen::Tensor<TensorT, 2> getError() const { Eigen::Tensor<TensorT, 2> error = error_; return error; }; ///< error copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getErrorMutable() { return error_; }; ///< error copy getter
		std::shared_ptr<HDelT> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<DDelT> getDErrorPointer() { return d_error_; }; ///< error pointer getter

    void setDerivative(const Eigen::Tensor<TensorT, 2>& derivative); ///< derivative setter
    Eigen::Tensor<TensorT, 2> getDerivative() const; ///< derivative copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getDerivativeMutable() { return derivative_; }; ///< derivative copy getter
		std::shared_ptr<HDelT> getHDerivativePointer() { return h_derivative_; }; ///< derivative pointer getter
		std::shared_ptr<DDelT> getDDerivativePointer() { return d_derivative_; }; ///< derivative pointer getter

    void setDt(const Eigen::Tensor<TensorT, 2>& dt); ///< dt setter
    Eigen::Tensor<TensorT, 2> getDt() const; ///< dt copy getter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& getDtMutable() { return dt_;	}; ///< dt copy getter
		std::shared_ptr<HDelT> getHDtPointer() { return h_dt_; }; ///< dt pointer getter
		std::shared_ptr<DDelT> getDDtPointer() { return d_dt_; }; ///< dt pointer getter

		void setOutputMin(const TensorT& min_output); ///< min output setter
		void setOutputMax(const TensorT& output_max); ///< max output setter

private:
    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& input_; ///< NodeData Net Input (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& output_; ///< NodeData Output (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& error_; ///< NodeData Error (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& derivative_; ///< NodeData Error (rows: # of samples, cols: # of time steps)
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>>& dt_; ///< Resolution of each time-step (rows: # of samples, cols: # of time steps)		
		std::shared_ptr<HDelT> h_input_;
		std::shared_ptr<HDelT> h_output_;
		std::shared_ptr<HDelT> h_error_;
		std::shared_ptr<HDelT> h_derivative_;
		std::shared_ptr<HDelT> h_dt_;
		std::shared_ptr<DDelT> d_input_;
		std::shared_ptr<DDelT> d_output_;
		std::shared_ptr<DDelT> d_error_;
		std::shared_ptr<DDelT> d_derivative_;
		std::shared_ptr<DDelT> d_dt_;

		TensorT output_min_ = -1.0e6; ///< Min Node output
		TensorT output_max_ = 1.0e6; ///< Max Node output
  };
}

#endif //SMARTPEAK_NODEDATA_H