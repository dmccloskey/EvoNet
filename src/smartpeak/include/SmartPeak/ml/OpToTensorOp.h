/**TODO:  Add copyright*/

#ifndef SMARTPEAK_OPTOTENSOROP_H
#define SMARTPEAK_OPTOTENSOROP_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <SmartPeak/ml/IntegrationFunction.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/SolverTensor.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  /**
    @brief Base class for all conversions from ...Op to ...TensorOp.
  */
	template<typename TensorT, typename DeviceT, typename OperatorT, typename OperatorTensorT>
  class OpToTensorOp
  {
	public: 
		OpToTensorOp() {};
		~OpToTensorOp() {};
		virtual void operator()(OperatorT<TensorT>* op_class, OperatorTensorT<TensorT, DeviceT>* op_tensor_class, std::vector<TensorT>& op_params) const = 0;
		virtual OperatorTensorT<TensorT, DeviceT>* convertOpToTensorOp(OperatorT<TensorT>* op_class) const = 0;
		virtual std::vector<TensorT> getTensorParams(OperatorT<TensorT>* op_class) const = 0;
  };

	template<typename TensorT, typename DeviceT, typename OperatorT, typename OperatorTensorT>
	class ActivationOpToActivationTensorOp: public OpToTensorOp<ReLUOp<TensorT>, ReLUTensorOp<TensorT,DeviceT>>
	{
	public:
		void operator()(ActivationOp<TensorT>* op_class, ActivationTensorOp<TensorT, DeviceT>* op_tensor_class, std::vector<TensorT>& op_params) const {
			//TODO
		}
		OperatorTensorT<TensorT, DeviceT>* convertOpToTensorOp(OperatorT<TensorT>* op_class) const {
			//TODO
		}
		std::vector<TensorT> getTensorParams(OperatorT<TensorT>* op_class) const {
			//TODO
		}
	};

}
#endif //SMARTPEAK_OPTOTENSOROP_H