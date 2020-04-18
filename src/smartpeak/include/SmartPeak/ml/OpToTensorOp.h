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
		virtual std::shared_ptr<OperatorTensorT> convertOpToTensorOp(std::shared_ptr<OperatorT>& op_class) const = 0;
		virtual std::vector<TensorT> getTensorParams(std::shared_ptr<OperatorT>& op_class) const = 0;
		void operator()(std::shared_ptr<OperatorT>& op_class, std::shared_ptr<OperatorTensorT>& op_tensor_class, std::vector<TensorT>& op_params) const {
			op_tensor_class = convertOpToTensorOp(op_class);
			op_params = getTensorParams(op_class);
		}
  };

	template<typename TensorT, typename DeviceT>
	class ActivationOpToActivationTensorOp: public OpToTensorOp<TensorT, DeviceT, ActivationOp<TensorT>, ActivationTensorOp<TensorT,DeviceT>>
	{
	public:
		std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<ActivationOp<TensorT>>& op_class) const {
			if (op_class->getName() == "ReLUOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ReLUTensorOp<TensorT, DeviceT>>(ReLUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReLUGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ReLUGradTensorOp<TensorT, DeviceT>>(ReLUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ELUOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ELUTensorOp<TensorT, DeviceT>>( ELUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ELUGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ELUGradTensorOp<TensorT, DeviceT>>( ELUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "SigmoidOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SigmoidTensorOp<TensorT, DeviceT>>( SigmoidTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "SigmoidGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SigmoidGradTensorOp<TensorT, DeviceT>>( SigmoidGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "TanHOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<TanHTensorOp<TensorT, DeviceT>>( TanHTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "TanHGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<TanHGradTensorOp<TensorT, DeviceT>>( TanHGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReTanHOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ReTanHTensorOp<TensorT, DeviceT>>( ReTanHTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ReTanHGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ReTanHGradTensorOp<TensorT, DeviceT>>( ReTanHGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "LinearOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LinearTensorOp<TensorT, DeviceT>>( LinearTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "LinearGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LinearGradTensorOp<TensorT, DeviceT>>( LinearGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "InverseOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<InverseTensorOp<TensorT, DeviceT>>( InverseTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "InverseGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<InverseGradTensorOp<TensorT, DeviceT>>( InverseGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ExponentialOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ExponentialTensorOp<TensorT, DeviceT>>( ExponentialTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "ExponentialGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ExponentialGradTensorOp<TensorT, DeviceT>>( ExponentialGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "LogOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LogTensorOp<TensorT, DeviceT>>( LogTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "LogGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LogGradTensorOp<TensorT, DeviceT>>( LogGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "PowOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PowTensorOp<TensorT, DeviceT>>( PowTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "PowGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PowGradTensorOp<TensorT, DeviceT>>( PowGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "LeakyReLUOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LeakyReLUTensorOp<TensorT, DeviceT>>( LeakyReLUTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "LeakyReLUGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LeakyReLUGradTensorOp<TensorT, DeviceT>>( LeakyReLUGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2], op_class->getParameters()[3]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "SinOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SinTensorOp<TensorT, DeviceT>>( SinTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "SinGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SinGradTensorOp<TensorT, DeviceT>>( SinGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "CosOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<CosTensorOp<TensorT, DeviceT>>( CosTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "CosGradOp") {
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<CosGradTensorOp<TensorT, DeviceT>>( CosGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "BatchNormOp") {
      std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<BatchNormTensorOp<TensorT, DeviceT>>( BatchNormTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
      return op_tensor_class;
      }
      else if (op_class->getName() == "BatchNormGradOp") {
      std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<BatchNormGradTensorOp<TensorT, DeviceT>>( BatchNormGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
      return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared< LinearTensorOp<TensorT, DeviceT>>( LinearTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<ActivationOp<TensorT>>& op_class) const {	return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class SolverOpToSolverTensorOp : public OpToTensorOp<TensorT, DeviceT, SolverOp<TensorT>, SolverTensorOp<TensorT, DeviceT>>
	{
	public:
		std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<SolverOp<TensorT>>& op_class) const {
			if (op_class->getName() == "SGDOp") {
				std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SGDTensorOp<TensorT, DeviceT>>( SGDTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma(), op_class->getGradientNoiseGamma()));
				return op_tensor_class;
			}
      else if (op_class->getName() == "SSDOp") {
        std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SSDTensorOp<TensorT, DeviceT>>( SSDTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma(), op_class->getGradientNoiseGamma()));
        return op_tensor_class;
      }
			else if (op_class->getName() == "AdamOp") {
				std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<AdamTensorOp<TensorT, DeviceT>>( AdamTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma(), op_class->getGradientNoiseGamma()));
				return op_tensor_class;
			}
      else if (op_class->getName() == "SVAGOp") {
        std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SVAGTensorOp<TensorT, DeviceT>>( SVAGTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma(), op_class->getGradientNoiseGamma()));
        return op_tensor_class;
      }
			else if (op_class->getName() == "DummySolverOp") {
				std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<DummySolverTensorOp<TensorT, DeviceT>>( DummySolverTensorOp<TensorT, DeviceT>(op_class->getGradientThreshold(), op_class->getGradientNoiseSigma(), op_class->getGradientNoiseGamma()));
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<SolverTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<DummySolverTensorOp<TensorT, DeviceT>>( DummySolverTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<SolverOp<TensorT>>& op_class) const {	return op_class->getParameters();	}
	};

	template<typename TensorT, typename DeviceT>
	class LossFunctionOpToLossFunctionTensorOp : public OpToTensorOp<TensorT, DeviceT, LossFunctionOp<TensorT>, LossFunctionTensorOp<TensorT, DeviceT>>
	{
	public:
		std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<LossFunctionOp<TensorT>>& op_class) const {
			if (op_class->getName() == "ManhattanDistanceLossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ManhattanDistanceLossTensorOp<TensorT, DeviceT>>( ManhattanDistanceLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormLossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<L2NormLossTensorOp<TensorT, DeviceT>>( L2NormLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCELossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<BCELossTensorOp<TensorT, DeviceT>>( BCELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodLossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<NegativeLogLikelihoodLossTensorOp<TensorT, DeviceT>>( NegativeLogLikelihoodLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSELossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSELossTensorOp<TensorT, DeviceT>>( MSELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "MAELossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MAELossTensorOp<TensorT, DeviceT>>( MAELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MRSELossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MRSELossTensorOp<TensorT, DeviceT>>( MRSELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MLELossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MLELossTensorOp<TensorT, DeviceT>>( MLELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
			else if (op_class->getName() == "KLDivergenceMuLossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<KLDivergenceMuLossTensorOp<TensorT, DeviceT>>( KLDivergenceMuLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarLossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<KLDivergenceLogVarLossTensorOp<TensorT, DeviceT>>( KLDivergenceLogVarLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsLossOp") {
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<BCEWithLogitsLossTensorOp<TensorT, DeviceT>>( BCEWithLogitsLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsLossOp") {
				CrossEntropyWithLogitsLossTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<CrossEntropyWithLogitsLossTensorOp<TensorT, DeviceT>>( CrossEntropyWithLogitsLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeUBLossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSERangeUBLossTensorOp<TensorT, DeviceT>>( MSERangeUBLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeLBLossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSERangeLBLossTensorOp<TensorT, DeviceT>>( MSERangeLBLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "KLDivergenceCatLossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<KLDivergenceCatLossTensorOp<TensorT, DeviceT>>( KLDivergenceCatLossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAPELossOp") {
        std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MAPELossTensorOp<TensorT, DeviceT>>( MAPELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<LossFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSELossTensorOp<TensorT, DeviceT>>( MSELossTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<LossFunctionOp<TensorT>>& op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class LossFunctionGradOpToLossFunctionGradTensorOp : public OpToTensorOp<TensorT, DeviceT, LossFunctionGradOp<TensorT>, LossFunctionGradTensorOp<TensorT, DeviceT>>
	{
	public:
		LossFunctionGradTensorOp<TensorT, DeviceT>* convertOpToTensorOp(LossFunctionGradOp<TensorT>* op_class) const {
			if (op_class->getName() == "ManhattanDistanceLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( ManhattanDistanceLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( L2NormLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCELossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( BCELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( NegativeLogLikelihoodLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSELossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "MAELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MAELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MRSELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MRSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MLELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MLELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
			else if (op_class->getName() == "KLDivergenceMuLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( KLDivergenceMuLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( KLDivergenceLogVarLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsLossGradOp") {
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( BCEWithLogitsLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsLossGradOp") {
				CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeLBLossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MSERangeLBLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeUBLossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MSERangeUBLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "KLDivergenceCatLossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( KLDivergenceCatLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAPELossGradOp") {
        LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MAPELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				LossFunctionGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
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
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( SumTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( ProdTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "ProdSCOp") {
        IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( ProdSCTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "MeanOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MeanTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MaxTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinOp") {
        IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MinTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( VarModTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( VarTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountOp") {
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( CountTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				IntegrationTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( SumTensorOp<TensorT, DeviceT>();
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
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( SumErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( ProdErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MeanErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MeanErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MaxErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinErrorOp") {
        IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MinErrorTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( VarModErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarErrorOp") {// [TODO: ]
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( VarErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountErrorOp") {
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( CountErrorTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				IntegrationErrorTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( SumErrorTensorOp<TensorT, DeviceT>();
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
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( SumWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( ProdWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MeanWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MeanWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MaxWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinWeightGradOp") {
        IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MinWeightGradTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( VarModWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( VarWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountWeightGradOp") {
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( CountWeightGradTensorOp<TensorT, DeviceT>();
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				IntegrationWeightGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( SumWeightGradTensorOp<TensorT, DeviceT>();
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
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( AccuracyBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "AccuracyMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( AccuracyMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "AccuracyMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( AccuracyMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( PrecisionBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( PrecisionMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( PrecisionMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( RecallBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( RecallMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( RecallMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreBCOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( F1ScoreBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0));
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreMCMicroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( F1ScoreMCMicroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreMCMacroOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( F1ScoreMCMacroTensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAEOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MAETensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "CosineSimilarityOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( CosineSimilarityTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "PearsonROp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( PearsonRTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "EuclideanDistOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( EuclideanDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "ManhattanDistOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( ManhattanDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "JeffreysAndMatusitaDistOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( JeffreysAndMatusitaDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "LogarithmicDistOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( LogarithmicDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else if (op_class->getName() == "PercentDifferenceOp") {
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( PercentDifferenceTensorOp<TensorT, DeviceT>(op_class->getReductionFunc());
        return op_tensor_class;
      }
      else {
        std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
        MetricFunctionTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<>( MAETensorOp<TensorT, DeviceT>();
        return op_tensor_class;
      }
    }
    std::vector<TensorT> getTensorParams(MetricFunctionOp<TensorT>* op_class) const { return std::vector<TensorT>(); }
  };
}
#endif //SMARTPEAK_OPTOTENSOROP_H