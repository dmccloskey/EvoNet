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
		std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<LossFunctionGradOp<TensorT>>& op_class) const {
			if (op_class->getName() == "ManhattanDistanceLossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ManhattanDistanceLossGradTensorOp<TensorT, DeviceT>>( ManhattanDistanceLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "L2NormLossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<L2NormLossGradTensorOp<TensorT, DeviceT>>( L2NormLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCELossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<BCELossGradTensorOp<TensorT, DeviceT>>( BCELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "NegativeLogLikelihoodLossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<NegativeLogLikelihoodLossGradTensorOp<TensorT, DeviceT>>( NegativeLogLikelihoodLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "MSELossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSELossGradTensorOp<TensorT, DeviceT>>( MSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "MAELossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MAELossGradTensorOp<TensorT, DeviceT>>( MAELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MRSELossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MRSELossGradTensorOp<TensorT, DeviceT>>( MRSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MLELossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MLELossGradTensorOp<TensorT, DeviceT>>( MLELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
			else if (op_class->getName() == "KLDivergenceMuLossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<KLDivergenceMuLossGradTensorOp<TensorT, DeviceT>>( KLDivergenceMuLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "KLDivergenceLogVarLossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<KLDivergenceLogVarLossGradTensorOp<TensorT, DeviceT>>( KLDivergenceLogVarLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "BCEWithLogitsLossGradOp") {
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<BCEWithLogitsLossGradTensorOp<TensorT, DeviceT>>( BCEWithLogitsLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
			else if (op_class->getName() == "CrossEntropyWithLogitsLossGradOp") {
				CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>* op_tensor_class = std::make_shared<CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>>( CrossEntropyWithLogitsLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
      else if (op_class->getName() == "MSERangeLBLossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSERangeLBLossGradTensorOp<TensorT, DeviceT>>( MSERangeLBLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MSERangeUBLossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSERangeUBLossGradTensorOp<TensorT, DeviceT>>( MSERangeUBLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "KLDivergenceCatLossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<KLDivergenceCatLossGradTensorOp<TensorT, DeviceT>>( KLDivergenceCatLossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1], op_class->getParameters()[2]));
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAPELossGradOp") {
        std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MAPELossGradTensorOp<TensorT, DeviceT>>( MAPELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
        return op_tensor_class;
      }
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<LossFunctionGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MSELossGradTensorOp<TensorT, DeviceT>>( MSELossGradTensorOp<TensorT, DeviceT>(op_class->getParameters()[0], op_class->getParameters()[1]));
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<LossFunctionGradOp<TensorT>>& op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class IntegrationOpToIntegrationTensorOp : public OpToTensorOp<TensorT, DeviceT, IntegrationOp<TensorT>, IntegrationTensorOp<TensorT, DeviceT>>
	{
	public:
		std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<IntegrationOp<TensorT>>& op_class) const {
			if (op_class->getName() == "SumOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SumTensorOp<TensorT, DeviceT>>( SumTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ProdTensorOp<TensorT, DeviceT>>( ProdTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
      else if (op_class->getName() == "ProdSCOp") {
        std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ProdSCTensorOp<TensorT, DeviceT>>( ProdSCTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
			else if (op_class->getName() == "MeanOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MeanTensorOp<TensorT, DeviceT>>( MeanTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MaxTensorOp<TensorT, DeviceT>>( MaxTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinOp") {
        std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MinTensorOp<TensorT, DeviceT>>( MinTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<VarModTensorOp<TensorT, DeviceT>>( VarModTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<VarTensorOp<TensorT, DeviceT>>( VarTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountOp") {
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<CountTensorOp<TensorT, DeviceT>>( CountTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<IntegrationTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SumTensorOp<TensorT, DeviceT>>( SumTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<IntegrationOp<TensorT>>& op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class IntegrationErrorOpToIntegrationErrorTensorOp : public OpToTensorOp<TensorT, DeviceT, IntegrationErrorOp<TensorT>, IntegrationErrorTensorOp<TensorT, DeviceT>>
	{
	public:
		std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<IntegrationErrorOp<TensorT>>& op_class) const {
			if (op_class->getName() == "SumErrorOp") {
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SumErrorTensorOp<TensorT, DeviceT>>( SumErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdErrorOp") {
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ProdErrorTensorOp<TensorT, DeviceT>>( ProdErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "MeanErrorOp") {
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MeanErrorTensorOp<TensorT, DeviceT>>( MeanErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxErrorOp") {
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MaxErrorTensorOp<TensorT, DeviceT>>( MaxErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinErrorOp") {
        std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MinErrorTensorOp<TensorT, DeviceT>>( MinErrorTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModErrorOp") {
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<VarModErrorTensorOp<TensorT, DeviceT>>( VarModErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarErrorOp") {// [TODO: ]
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<VarErrorTensorOp<TensorT, DeviceT>>( VarErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountErrorOp") {
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<CountErrorTensorOp<TensorT, DeviceT>>( CountErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<IntegrationErrorTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SumErrorTensorOp<TensorT, DeviceT>>( SumErrorTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<IntegrationErrorOp<TensorT>>& op_class) const { return std::vector<TensorT>(); }
	};

	template<typename TensorT, typename DeviceT>
	class IntegrationWeightGradOpToIntegrationWeightGradTensorOp : public OpToTensorOp<TensorT, DeviceT, IntegrationWeightGradOp<TensorT>, IntegrationWeightGradTensorOp<TensorT, DeviceT>>
	{
	public:
		std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<IntegrationWeightGradOp<TensorT>>& op_class) const {
			if (op_class->getName() == "SumWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SumWeightGradTensorOp<TensorT, DeviceT>>( SumWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "ProdWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ProdWeightGradTensorOp<TensorT, DeviceT>>( ProdWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "MeanWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MeanWeightGradTensorOp<TensorT, DeviceT>>( MeanWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "MaxWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MaxWeightGradTensorOp<TensorT, DeviceT>>( MaxWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
      else if (op_class->getName() == "MinWeightGradOp") {
        std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MinWeightGradTensorOp<TensorT, DeviceT>>( MinWeightGradTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
			else if (op_class->getName() == "VarModWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<VarModWeightGradTensorOp<TensorT, DeviceT>>( VarModWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "VarWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<VarWeightGradTensorOp<TensorT, DeviceT>>( VarWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else if (op_class->getName() == "CountWeightGradOp") {
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<CountWeightGradTensorOp<TensorT, DeviceT>>( CountWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
			else {
				std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
				std::shared_ptr<IntegrationWeightGradTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<SumWeightGradTensorOp<TensorT, DeviceT>>( SumWeightGradTensorOp<TensorT, DeviceT>());
				return op_tensor_class;
			}
		}
		std::vector<TensorT> getTensorParams(std::shared_ptr<IntegrationWeightGradOp<TensorT>>& op_class) const { return std::vector<TensorT>(); }
	};

  template<typename TensorT, typename DeviceT>
  class MetricFunctionOpToMetricFunctionTensorOp : public OpToTensorOp<TensorT, DeviceT, MetricFunctionOp<TensorT>, MetricFunctionTensorOp<TensorT, DeviceT>>
  {
  public:
    std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> convertOpToTensorOp(std::shared_ptr<MetricFunctionOp<TensorT>>& op_class) const {
      if (op_class->getName() == "AccuracyBCOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<AccuracyBCTensorOp<TensorT, DeviceT>>( AccuracyBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0)));
        return op_tensor_class;
      }
      else if (op_class->getName() == "AccuracyMCMicroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<AccuracyMCMicroTensorOp<TensorT, DeviceT>>( AccuracyMCMicroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "AccuracyMCMacroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<AccuracyMCMacroTensorOp<TensorT, DeviceT>>( AccuracyMCMacroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionBCOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PrecisionBCTensorOp<TensorT, DeviceT>>( PrecisionBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0)));
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionMCMicroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PrecisionMCMicroTensorOp<TensorT, DeviceT>>( PrecisionMCMicroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "PrecisionMCMacroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PrecisionMCMacroTensorOp<TensorT, DeviceT>>( PrecisionMCMacroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallBCOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<RecallBCTensorOp<TensorT, DeviceT>>( RecallBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0)));
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallMCMicroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<RecallMCMicroTensorOp<TensorT, DeviceT>>( RecallMCMicroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "RecallMCMacroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<RecallMCMacroTensorOp<TensorT, DeviceT>>( RecallMCMacroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreBCOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<F1ScoreBCTensorOp<TensorT, DeviceT>>( F1ScoreBCTensorOp<TensorT, DeviceT>(op_class->getParameters().at(0)));
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreMCMicroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<F1ScoreMCMicroTensorOp<TensorT, DeviceT>>( F1ScoreMCMicroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "F1ScoreMCMacroOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<F1ScoreMCMacroTensorOp<TensorT, DeviceT>>( F1ScoreMCMacroTensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
      else if (op_class->getName() == "MAEOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MAETensorOp<TensorT, DeviceT>>( MAETensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "CosineSimilarityOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<CosineSimilarityTensorOp<TensorT, DeviceT>>( CosineSimilarityTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "PearsonROp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PearsonRTensorOp<TensorT, DeviceT>>( PearsonRTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "EuclideanDistOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<EuclideanDistTensorOp<TensorT, DeviceT>>( EuclideanDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "ManhattanDistOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<ManhattanDistTensorOp<TensorT, DeviceT>>( ManhattanDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "JeffreysAndMatusitaDistOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<JeffreysAndMatusitaDistTensorOp<TensorT, DeviceT>>( JeffreysAndMatusitaDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "LogarithmicDistOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<LogarithmicDistTensorOp<TensorT, DeviceT>>( LogarithmicDistTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else if (op_class->getName() == "PercentDifferenceOp") {
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<PercentDifferenceTensorOp<TensorT, DeviceT>>( PercentDifferenceTensorOp<TensorT, DeviceT>(op_class->getReductionFunc()));
        return op_tensor_class;
      }
      else {
        std::cout << "No conversion available for " << op_class->getName() << "." << std::endl;
        std::shared_ptr<MetricFunctionTensorOp<TensorT, DeviceT>> op_tensor_class = std::make_shared<MAETensorOp<TensorT, DeviceT>>( MAETensorOp<TensorT, DeviceT>());
        return op_tensor_class;
      }
    }
    std::vector<TensorT> getTensorParams(std::shared_ptr<MetricFunctionOp<TensorT>>& op_class) const { return std::vector<TensorT>(); }
  };
}
#endif //SMARTPEAK_OPTOTENSOROP_H