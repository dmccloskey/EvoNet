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
		virtual OperatorTensorT* convertOpToTensorOp(OperatorT* op_class) const = 0;
		virtual std::vector<TensorT> getTensorParams(OperatorT* op_class) const = 0;
		void operator()(OperatorT* op_class, OperatorTensorT* op_tensor_class, std::vector<TensorT>& op_params) const {
			op_tensor_class = convertOpToTensorOp(op_class);
			op_params = getTensorParams(op_class);
		}
  };

	template<typename TensorT, typename DeviceT>
	class ActivationOpToActivationTensorOp: public OpToTensorOp<TensorT, DeviceT, ActivationOp<TensorT>, ActivationTensorOp<TensorT,DeviceT>>
	{
	public:
		ActivationTensorOp<TensorT, DeviceT>* convertOpToTensorOp(ActivationOp<TensorT>* op_class) const {
			if (op_class->getName() == "ReLUOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReLUTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReLUGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReLUGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			//else if (op_class->getName() == "ELUOp") {
			//	return &ELUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
			//}
			//else if (op_class->getName() == "ELUGradOp") {
			//	return &ELUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
			//}
			//else if (op_class->getName() == "SigmoidOp") {
			//	return &SigmoidTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "SigmoidGradOp") {
			//	return &SigmoidGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "TanHOp") {
			//	return &TanHTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "TanHGradOp") {
			//	return &TanHGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "ReTanHOp") {
			//	return &ReTanHTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "ReTanHGradOp") {
			//	return &ReTanHGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "LinearOp") {
			//	return &LinearTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "LinearGradOp") {
			//	return &LinearGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "InverseOp") {
			//	return &InverseTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "InverseGradOp") {
			//	return &InverseGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "ExponentialOp") {
			//	return &ExponentialTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "ExponentialGradOp") {
			//	return &ExponentialGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "LogOp") {
			//	return &LogTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "LogGradOp") {
			//	return &LogGradTensorOp<TensorT, DeviceT>();
			//}
			//else if (op_class->getName() == "PowOp") {
			//	return &PowTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
			//}
			//else if (op_class->getName() == "PowGradOp") {
			//	return &PowGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
			//}
			//else if (op_class->getName() == "LeakyReLUOp") {
			//	return &LeakyReLUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
			//}
			//else if (op_class->getName() == "LeakyReLUGradOp") {
			//	return &LeakyReLUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
			//}
			//else {
			//	std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
			//	return &LinearTensorOp<TensorT, DeviceT>();
			//}
		}
		std::vector<TensorT> getTensorParams(ActivationOp<TensorT>* op_class) const {	return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class SolverOpToSolverTensorOp : public OpToTensorOp<TensorT, DeviceT, SolverOp<TensorT>, SolverTensorOp<TensorT, DeviceT>>
	{
	public:
		SolverTensorOp<TensorT, DeviceT>* convertOpToTensorOp(SolverOp<TensorT>* op_class) const {
			if (op_class->getName() == "SGDOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new SGDTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "AdamOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new AdamTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "DummySolverOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new DummySolverTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "SGDNoiseOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new SGDNoiseTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new DummySolverTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(SolverOp<TensorT>* op_class) const {	return op_class->getParameters();	}
	};

}
#endif //SMARTPEAK_OPTOTENSOROP_H