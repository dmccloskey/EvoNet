/**TODO:  Add copyright*/

#ifndef SMARTPEAK_OPTOTENSOROP_H
#define SMARTPEAK_OPTOTENSOROP_H

#include <SmartPeak/ml/ActivationFunction.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <SmartPeak/ml/IntegrationFunction.h>
#include <SmartPeak/ml/IntegrationFunctionTensor.h>
#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/SolverTensor.h>
#include <SmartPeak/ml/LossFunction.h>
#include <SmartPeak/ml/LossFunctionTensor.h>
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
			else if (op_class->getName() == "ELUOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ELUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ELUGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ELUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "SigmoidOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new SigmoidTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "SigmoidGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new SigmoidGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "TanHOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new TanHTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "TanHGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new TanHGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReTanHOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReTanHTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReTanHGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReTanHGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "LinearOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LinearTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "LinearGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LinearGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "InverseOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new InverseTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "InverseGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new InverseGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ExponentialOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ExponentialTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ExponentialGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ExponentialGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "LogOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LogTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "LogGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LogGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "PowOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new PowTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "PowGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new PowGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LeakyReLUOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LeakyReLUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LeakyReLUGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LeakyReLUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0]);
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LinearTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
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

	template<typename TensorT, typename DeviceT>
	class LossFunctionOpToLossFunctionTensorOp : public OpToTensorOp<TensorT, DeviceT, LossFunctionOp<TensorT>, LossFunctionTensorOp<TensorT, DeviceT>>
	{
	public:
		LossFunctionTensorOp<TensorT, DeviceT>* convertOpToTensorOp(LossFunctionOp<TensorT>* op_class) const {
			if (op_class->getName() == "EuclideanDistanceOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new EuclideanDistanceTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new L2NormTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new BCETensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new NegativeLogLikelihoodTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSEOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSETensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceMuOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceMuTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceLogVarTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEWithLogitsTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSETensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(SolverOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class LossFunctionGradOpToLossFunctionGradTensorOp : public OpToTensorOp<TensorT, DeviceT, LossFunctionGradOp<TensorT>, LossFunctionGradTensorOp<TensorT, DeviceT>>
	{
	public:
		LossFunctionGradTensorOp<TensorT, DeviceT>* convertOpToTensorOp(LossFunctionOp<TensorT>* op_class) const {
			if (op_class->getName() == "EuclideanDistanceGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new EuclideanDistanceGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new L2NormGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new NegativeLogLikelihoodGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSEGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSEGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceMuGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceMuGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceLogVarGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEWithLogitsGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSEGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(SolverOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

}
#endif //SMARTPEAK_OPTOTENSOROP_H