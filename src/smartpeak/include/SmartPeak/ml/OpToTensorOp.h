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
		OpToTensorOp() = default;
		virtual ~OpToTensorOp() = default;
		virtual OperatorTensorT* convertOpToTensorOp(OperatorT* op_class) const = 0;
		virtual std::vector<TensorT> getTensorParams(OperatorT* op_class) const = 0;
		void operator()(OperatorT* op_class, OperatorTensorT*& op_tensor_class, std::vector<TensorT>& op_params) const {
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
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReLUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReLUGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReLUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ELUOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ELUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ELUGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ELUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "SigmoidOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new SigmoidTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "SigmoidGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new SigmoidGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "TanHOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new TanHTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "TanHGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new TanHGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReTanHOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReTanHTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReTanHGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ReTanHGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LinearOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LinearTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LinearGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LinearGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "InverseOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new InverseTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "InverseGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new InverseGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ExponentialOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ExponentialTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "ExponentialGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new ExponentialGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LogOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LogTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LogGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LogGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "PowOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new PowTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "PowGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new PowGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LeakyReLUOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LeakyReLUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "LeakyReLUGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new LeakyReLUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "SinOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new SinTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "SinGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new SinGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "CosOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new CosTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "CosGradOp") {
				ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new CosGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
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
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new SGDTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma());
				return op_tensor_class;
			}
			else if (op_class->getName() == "AdamOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new AdamTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma());
				return op_tensor_class;
			}
			else if (op_class->getName() == "DummySolverOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new DummySolverTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma());
				return op_tensor_class;
			}
			else if (op_class->getName() == "SGDNoiseOp") {
				SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new SGDNoiseTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma());
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
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new EuclideanDistanceTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new L2NormTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new BCETensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new NegativeLogLikelihoodTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSEOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSETensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceMuOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceMuTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceLogVarTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEWithLogitsTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsOp") {
				CrossEntropyWithLogitsTensorOp<TensorT, DeviceT>* op_tensor_class = new CrossEntropyWithLogitsTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeUBOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeUBTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeLBOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeLBTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSETensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(LossFunctionOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class LossFunctionGradOpToLossFunctionGradTensorOp : public OpToTensorOp<TensorT, DeviceT, LossFunctionGradOp<TensorT>, LossFunctionGradTensorOp<TensorT, DeviceT>>
	{
	public:
		LossFunctionGradTensorOp<TensorT, DeviceT>* convertOpToTensorOp(LossFunctionGradOp<TensorT>* op_class) const {
			if (op_class->getName() == "EuclideanDistanceGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new EuclideanDistanceGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new L2NormGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new NegativeLogLikelihoodGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSEGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSEGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceMuGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceMuGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceLogVarGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEWithLogitsGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsGradOp") {
				CrossEntropyWithLogitsGradTensorOp<TensorT, DeviceT>* op_tensor_class = new CrossEntropyWithLogitsGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeLBGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeLBGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeUBGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeUBGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSEGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(LossFunctionGradOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class IntegrationOpToIntegrationTensorOp : public OpToTensorOp<TensorT, DeviceT, IntegrationOp<TensorT>, IntegrationTensorOp<TensorT, DeviceT>>
	{
	public:
		IntegrationTensorOp<TensorT, DeviceT>* convertOpToTensorOp(IntegrationOp<TensorT>* op_class) const {
			if (op_class->getName() == "SumOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new SumTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new ProdTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "ProdSCOp") {
        IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new ProdSCTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "MeanOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new MeanTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new MaxTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinOp") {
        IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new MinTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new VarModTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new VarTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new CountTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = new SumTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(IntegrationOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class IntegrationErrorOpToIntegrationErrorTensorOp : public OpToTensorOp<TensorT, DeviceT, IntegrationErrorOp<TensorT>, IntegrationErrorTensorOp<TensorT, DeviceT>>
	{
	public:
		IntegrationErrorTensorOp<TensorT, DeviceT>* convertOpToTensorOp(IntegrationErrorOp<TensorT>* op_class) const {
			if (op_class->getName() == "SumErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new SumErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new ProdErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MeanErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new MeanErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new MaxErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinErrorOp") {
        IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new MinErrorTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new VarModErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarErrorOp") {// [TODO: ]
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new VarErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new CountErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = new SumErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(IntegrationErrorOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class IntegrationWeightGradOpToIntegrationWeightGradTensorOp : public OpToTensorOp<TensorT, DeviceT, IntegrationWeightGradOp<TensorT>, IntegrationWeightGradTensorOp<TensorT, DeviceT>>
	{
	public:
		IntegrationWeightGradTensorOp<TensorT, DeviceT>* convertOpToTensorOp(IntegrationWeightGradOp<TensorT>* op_class) const {
			if (op_class->getName() == "SumWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new SumWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new ProdWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MeanWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MeanWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MaxWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinWeightGradOp") {
        IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MinWeightGradTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new VarModWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new VarWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new CountWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = new SumWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(IntegrationWeightGradOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
	};

}
#endif //SMARTPEAK_OPTOTENSOROP_H