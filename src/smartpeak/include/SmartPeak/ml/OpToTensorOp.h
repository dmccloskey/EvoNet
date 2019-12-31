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
#include <SmartPeak/ml/MetricFunction.h>
#include <SmartPeak/ml/MetricFunctionTensor.h>
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
      else if (op_class->getName() == "BatchNormOp") {
      ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new BatchNormTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
      return op_tensor_class;
      }
      else if (op_class->getName() == "BatchNormGradOp") {
      ActivationTensorOp<TensorT, DeviceT>* op_tensor_class = new BatchNormGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
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
      else if (op_class->getName() == "SSDOp") {
        SolverTensorOp<TensorT, DeviceT>* op_tensor_class = new SSDTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma());
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
			if (op_class->getName() == "ManhattanDistanceLossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new ManhattanDistanceLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormLossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new L2NormLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCELossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new BCELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodLossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new NegativeLogLikelihoodLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSELossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
      else if (op_class->getName() == "MAELossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MAELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MRSELossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MRSELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MLELossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MLELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
			else if (op_class->getName() == "KLDivergenceMuLossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceMuLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarLossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceLogVarLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsLossOp") {
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEWithLogitsLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsLossOp") {
				CrossEntropyWithLogitsLossTensorOp<TensorT, DeviceT>* op_tensor_class = new CrossEntropyWithLogitsLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeUBLossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeUBLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeLBLossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeLBLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "KLDivergenceCatLossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceCatLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAPELossOp") {
        LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MAPELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MSELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
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
			if (op_class->getName() == "ManhattanDistanceLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new ManhattanDistanceLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new L2NormLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCELossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new BCELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new NegativeLogLikelihoodLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSELossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
      else if (op_class->getName() == "MAELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MAELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MRSELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MRSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MLELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MLELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
			else if (op_class->getName() == "KLDivergenceMuLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceMuLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceLogVarLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new BCEWithLogitsLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsLossGradOp") {
				CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>* op_tensor_class = new CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeLBLossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeLBLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeUBLossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSERangeUBLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "KLDivergenceCatLossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new KLDivergenceCatLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]);
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAPELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MAPELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = new MSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]);
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

  template<typename TensorT, typename DeviceT>
  class MetricFunctionOpToMetricFunctionTensorOp : public OpToTensorOp<TensorT, DeviceT, MetricFunctionOp<TensorT>, MetricFunctionTensorOp<TensorT, DeviceT>>
  {
  public:
    MetricFunctionTensorOp<TensorT, DeviceT>* convertOpToTensorOp(MetricFunctionOp<TensorT>* op_class) const {
      if (op_class->getName() == "AccuracyBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new AccuracyBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "AccuracyMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new AccuracyMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "AccuracyMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new AccuracyMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new PrecisionBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new PrecisionMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new PrecisionMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new RecallBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new RecallMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new RecallMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new F1ScoreBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new F1ScoreMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new F1ScoreMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAEOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MAETensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "CosineSimilarityOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new CosineSimilarityTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "PearsonROp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new PearsonRTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else {
        std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = new MAETensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
    }
    std::vector<TensorT> getTensorParams(MetricFunctionOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
  };
}
#endif //SMARTPEAK_OPTOTENSOROP_H