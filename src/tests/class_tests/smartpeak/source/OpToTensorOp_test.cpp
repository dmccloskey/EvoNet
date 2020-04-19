/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE OpToTensorOp test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/OpToTensorOp.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(OpToTensorOp1)

BOOST_AUTO_TEST_CASE(constructorActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<ActivationOp<float>> op_class;
	std::shared_ptr<ActivationTensorOp<float, Eigen::DefaultDevice>> op_tensor_class;

	op_class = std::make_shared<ReLUOp<float>>(ReLUOp<float>(1, 2, 3));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReLUTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReLUGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ELUOp<float>>(ELUOp<float>(1.0f, 2.0f, 3.0f, 4.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ELUTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ELUGradOp<float>>(ELUGradOp<float>(1.0f, 2.0f, 3.0f, 4.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ELUGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<SigmoidOp<float>>(SigmoidOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SigmoidTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SigmoidGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<TanHOp<float>>(TanHOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "TanHTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<TanHGradOp<float>>(TanHGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "TanHGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ReTanHOp<float>>(ReTanHOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReTanHTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ReTanHGradOp<float>>(ReTanHGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReTanHGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<LinearOp<float>>(LinearOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LinearTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<LinearGradOp<float>>(LinearGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LinearGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<InverseOp<float>>(InverseOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "InverseTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<InverseGradOp<float>>(InverseGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "InverseGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ExponentialOp<float>>(ExponentialOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ExponentialTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<ExponentialGradOp<float>>(ExponentialGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ExponentialGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<LogOp<float>>(LogOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<LogGradOp<float>>(LogGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<PowOp<float>>(PowOp<float>(1.0f,2.0f,3.0f,2.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PowTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<PowGradOp<float>>(PowGradOp<float>(1.0f, 2.0f, 3.0f, 2.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PowGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<LeakyReLUOp<float>>(LeakyReLUOp<float>(1.0f, 2.0f, 3.0f, 4.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LeakyReLUTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<LeakyReLUGradOp<float>>(LeakyReLUGradOp<float>(1.0f, 2.0f, 3.0f, 4.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LeakyReLUGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<SinOp<float>>(SinOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SinTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<SinGradOp<float>>(SinGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SinGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<CosOp<float>>(CosOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CosTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = std::make_shared<CosGradOp<float>>(CosGradOp<float>(1.0f, 2.0f, 3.0f));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CosGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

  op_class = std::make_shared<BatchNormOp<float>>(BatchNormOp<float>(1.0f, 2.0f, 3.0f));
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "BatchNormTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

  op_class = std::make_shared<BatchNormGradOp<float>>(BatchNormGradOp<float>(1.0f, 2.0f, 3.0f));
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "BatchNormGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);
}

BOOST_AUTO_TEST_CASE(getTensorParamsActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<ActivationOp<float>> op_class = std::make_shared<ReLUOp<float>>(ReLUOp<float>());
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ReLUGradOp<float>>(ReLUGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ELUOp<float>>(ELUOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ELUGradOp<float>>(ELUGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<SigmoidOp<float>>(SigmoidOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<SigmoidGradOp<float>>(SigmoidGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<TanHOp<float>>(TanHOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<TanHGradOp<float>>(TanHGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ReTanHOp<float>>(ReTanHOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ReTanHGradOp<float>>(ReTanHGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<LinearOp<float>>(LinearOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<LinearGradOp<float>>(LinearGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<InverseOp<float>>(InverseOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<InverseGradOp<float>>(InverseGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ExponentialOp<float>>(ExponentialOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ExponentialGradOp<float>>(ExponentialGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<LogOp<float>>(LogOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<LogGradOp<float>>(LogGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<PowOp<float>>(PowOp<float>(2));
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<PowGradOp<float>>(PowGradOp<float>(2));
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<LeakyReLUOp<float>>(LeakyReLUOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<LeakyReLUGradOp<float>>(LeakyReLUGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<SinOp<float>>(SinOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<SinGradOp<float>>(SinGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<CosOp<float>>(CosOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<CosGradOp<float>>(CosGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<BatchNormOp<float>>(BatchNormOp<float>());
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<BatchNormGradOp<float>>(BatchNormGradOp<float>());
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_CASE(constructorSolverOpToSolverTensorOp)
{
	SolverOpToSolverTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	SolverOpToSolverTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new SolverOpToSolverTensorOp<float, Eigen::DefaultDevice>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorSolverOpToSolverTensorOp)
{
	SolverOpToSolverTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new SolverOpToSolverTensorOp<float, Eigen::DefaultDevice>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpSolverOpToSolverTensorOp)
{
	SolverOpToSolverTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<SolverOp<float>> op_class;
	std::shared_ptr<SolverTensorOp<float, Eigen::DefaultDevice>> op_tensor_class;

	op_class = std::make_shared<SGDOp<float>>(SGDOp<float>(0.1, 0.9, 10.0, 1.0, 0.55));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SGDTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);
  BOOST_CHECK_CLOSE(op_tensor_class->getGradientNoiseGamma(), 0.55, 1e-4);

  op_class = std::make_shared<SSDOp<float>>(SSDOp<float>(0.1, 0.9, 10.0, 1.0, 0.55));
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SSDTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);
  BOOST_CHECK_CLOSE(op_tensor_class->getGradientNoiseGamma(), 0.55, 1e-4);

	op_class = std::make_shared<AdamOp<float>>(AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10.0, 1.0, 0.55));
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "AdamTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);
  BOOST_CHECK_CLOSE(op_tensor_class->getGradientNoiseGamma(), 0.55, 1e-4);

  op_class = std::make_shared<SVAGOp<float>>(SVAGOp<float>(0.001, 0.9,10.0, 1.0, 0.55));
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SVAGTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);
  BOOST_CHECK_CLOSE(op_tensor_class->getGradientNoiseGamma(), 0.55, 1e-4);

	op_class = std::make_shared<DummySolverOp<float>>(DummySolverOp<float>());
  op_class->setGradientThreshold(10);
  op_class->setGradientNoiseSigma(1);
  op_class->setGradientNoiseGamma(0.55);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "DummySolverTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);
  BOOST_CHECK_CLOSE(op_tensor_class->getGradientNoiseGamma(), 0.55, 1e-4);
}

BOOST_AUTO_TEST_CASE(getTensorParamsSolverOpToSolverTensorOp)
{
	SolverOpToSolverTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	std::shared_ptr<SolverOp<float>> op_class = std::make_shared<SGDOp<float>>(SGDOp<float>(1, 2));
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 3);
	BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 0);

  op_class = std::make_shared<SSDOp<float>>(SSDOp<float>(1, 2);
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 3);
  BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 0);

	op_class = std::make_shared<AdamOp<float>>(AdamOp<float>(1, 2, 3, 4));
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 6);
	BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 3); 
	BOOST_CHECK_EQUAL(params[3], 4); BOOST_CHECK_EQUAL(params[4], 0); BOOST_CHECK_EQUAL(params[5], 0);

  op_class = std::make_shared<SVAGOp<float>>(SVAGOp<float>(1, 2));
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 4);
  BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 0);
  BOOST_CHECK_EQUAL(params[3], 0);

	op_class = std::make_shared<DummySolverOp<float>>(DummySolverOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_CASE(constructorIntegrationOpToIntegrationTensorOp)
{
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorIntegrationOpToIntegrationTensorOp)
{
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpIntegrationOpToIntegrationTensorOp)
{
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<IntegrationOp<float>> op_class;
	std::shared_ptr<IntegrationTensorOp<float, Eigen::DefaultDevice>> op_tensor_class;

	op_class = std::make_shared<SumOp<float>>(SumOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SumTensorOp");

	op_class = std::make_shared<ProdOp<float>>(ProdOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdTensorOp");

  op_class = std::make_shared<ProdSCOp<float>>(ProdSCOp<float>());
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdSCTensorOp");

	op_class = std::make_shared<MaxOp<float>>(MaxOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxTensorOp");

  op_class = std::make_shared<MinOp<float>>(MinOp<float>());
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MinTensorOp");

	//op_class = std::make_shared<VarOp<float>>(VarOp<float>()); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarTensorOp");

	op_class = std::make_shared<CountOp<float>>(CountOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountTensorOp");

	op_class = std::make_shared<VarModOp<float>>(VarModOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationOpToIntegrationTensorOp)
{
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	std::shared_ptr<IntegrationOp<float>> op_class = std::make_shared<SumOp<float>>(SumOp<float>());
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ProdOp<float>>(ProdOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<ProdSCOp<float>>(ProdSCOp<float>());
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MaxOp<float>>(MaxOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<MinOp<float>>(MinOp<float>());
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MeanOp<float>>(MeanOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = std::make_shared<VarOp<float>>(VarOp<float>());
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<CountOp<float>>(CountOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<VarModOp<float>>(VarModOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_CASE(constructorIntegrationErrorOpToIntegrationErrorTensorOp)
{
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorIntegrationErrorOpToIntegrationErrorTensorOp)
{
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpIntegrationErrorOpToIntegrationErrorTensorOp)
{
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<IntegrationErrorOp<float>> op_class;
	std::shared_ptr<IntegrationErrorTensorOp<float, Eigen::DefaultDevice>> op_tensor_class;

	op_class = std::make_shared<SumErrorOp<float>>(SumErrorOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SumErrorTensorOp");

	op_class = std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdErrorTensorOp");

	op_class = std::make_shared<MaxErrorOp<float>>(MaxErrorOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxErrorTensorOp");

  op_class = std::make_shared<MinErrorOp<float>>(MinErrorOp<float>());
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MinErrorTensorOp");

	op_class = std::make_shared<MeanErrorOp<float>>(MeanErrorOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanErrorTensorOp");

	//op_class = std::make_shared<VarErrorOp<float>>(VarErrorOp<float>()); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarErrorTensorOp");

	op_class = std::make_shared<CountErrorOp<float>>(CountErrorOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountErrorTensorOp");

	op_class = std::make_shared<VarModErrorOp<float>>(VarModErrorOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModErrorTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationErrorOpToIntegrationErrorTensorOp)
{
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	std::shared_ptr<IntegrationErrorOp<float>> op_class = std::make_shared<SumErrorOp<float>>(SumErrorOp<float>());
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ProdErrorOp<float>>(ProdErrorOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MaxErrorOp<float>>(MaxErrorOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<MinErrorOp<float>>(MinErrorOp<float>());
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MeanErrorOp<float>>(MeanErrorOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = std::make_shared<VarErrorOp<float>>(VarErrorOp<float>());
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<CountErrorOp<float>>(CountErrorOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<VarModErrorOp<float>>(VarModErrorOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_CASE(constructorIntegrationWeightGradOpToIntegrationWeightGradTensorOp)
{
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorIntegrationWeightGradOpToIntegrationWeightGradTensorOp)
{
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpIntegrationWeightGradOpToIntegrationWeightGradTensorOp)
{
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<IntegrationWeightGradOp<float>> op_class;
	std::shared_ptr<IntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>> op_tensor_class;

	op_class = std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SumWeightGradTensorOp");

	op_class = std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdWeightGradTensorOp");

	op_class = std::make_shared<MaxWeightGradOp<float>>(MaxWeightGradOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxWeightGradTensorOp");

  op_class = std::make_shared<MinWeightGradOp<float>>(MinWeightGradOp<float>());
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MinWeightGradTensorOp");

	op_class = std::make_shared<MeanWeightGradOp<float>>(MeanWeightGradOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanWeightGradTensorOp");

	//op_class = std::make_shared<VarWeightGradOp<float>>(VarWeightGradOp<float>()); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarWeightGradTensorOp");

	op_class = std::make_shared<CountWeightGradOp<float>>(CountWeightGradOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountWeightGradTensorOp");

	op_class = std::make_shared<VarModWeightGradOp<float>>(VarModWeightGradOp<float>());
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModWeightGradTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationWeightGradOpToIntegrationWeightGradTensorOp)
{
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	std::shared_ptr<IntegrationWeightGradOp<float>> op_class = std::make_shared<SumWeightGradOp<float>>(SumWeightGradOp<float>());
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<ProdWeightGradOp<float>>(ProdWeightGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MaxWeightGradOp<float>>(MaxWeightGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<MinWeightGradOp<float>>(MinWeightGradOp<float>());
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MeanWeightGradOp<float>>(MeanWeightGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = std::make_shared<VarWeightGradOp<float>>(VarWeightGradOp<float>());
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<CountWeightGradOp<float>>(CountWeightGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<VarModWeightGradOp<float>>(VarModWeightGradOp<float>());
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_CASE(constructorLossFunctionGradOpToLossFunctionGradTensorOp)
{
	LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorLossFunctionGradOpToLossFunctionGradTensorOp)
{
	LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpLossFunctionGradOpToLossFunctionGradTensorOp)
{
	LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	std::shared_ptr<LossFunctionGradOp<float>> op_class;
	std::shared_ptr<LossFunctionGradTensorOp<float, Eigen::DefaultDevice>> op_tensor_class;

	op_class = std::make_shared<ManhattanDistanceLossGradOp<float>>(ManhattanDistanceLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ManhattanDistanceLossGradTensorOp");

	op_class = std::make_shared<L2NormLossGradOp<float>>(L2NormLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormLossGradTensorOp");

	op_class = std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "NegativeLogLikelihoodLossGradTensorOp");

	op_class = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSELossGradTensorOp");

  op_class = std::make_shared<MAELossGradOp<float>>(MAELossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MAELossGradTensorOp");

  op_class = std::make_shared<MRSELossGradOp<float>>(MRSELossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MRSELossGradTensorOp");

  op_class = std::make_shared<MLELossGradOp<float>>(MLELossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MLELossGradTensorOp");

	op_class = new KLDivergenceMuLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceMuLossGradTensorOp");

	op_class = new KLDivergenceLogVarLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceLogVarLossGradTensorOp");

	op_class = new BCEWithLogitsLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "BCEWithLogitsLossGradTensorOp");

	op_class = new CrossEntropyWithLogitsLossGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CrossEntropyWithLogitsLossGradTensorOp");

  op_class = new MSERangeLBLossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeLBLossGradTensorOp");

  op_class = new MSERangeUBLossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeUBLossGradTensorOp");

  op_class = new KLDivergenceCatLossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceCatLossGradTensorOp");

  op_class = new MAPELossGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MAPELossGradTensorOp");

  delete op_class;
  delete op_tensor_class;
}

BOOST_AUTO_TEST_CASE(getTensorParamsLossFunctionGradOpToLossFunctionGradTensorOp)
{
	LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	LossFunctionGradOp<float>* op_class = nullptr;
	std::vector<float> params;

	op_class = std::make_shared<ManhattanDistanceLossGradOp<float>>(ManhattanDistanceLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<L2NormLossGradOp<float>>(L2NormLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<L2NormLossGradOp<float>>(L2NormLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<NegativeLogLikelihoodLossGradOp<float>>(NegativeLogLikelihoodLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = std::make_shared<MSELossGradOp<float>>(MSELossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<MAELossGradOp<float>>(MAELossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<MRSELossGradOp<float>>(MRSELossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = std::make_shared<MLELossGradOp<float>>(MLELossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceMuLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceLogVarLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new BCEWithLogitsLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CrossEntropyWithLogitsLossGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeLBLossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeUBLossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new KLDivergenceCatLossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MAPELossGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  delete op_class;
}

BOOST_AUTO_TEST_CASE(constructorLossFunctionOpToLossFunctionTensorOp)
{
	LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice>();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorLossFunctionOpToLossFunctionTensorOp)
{
	LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice>();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpLossFunctionOpToLossFunctionTensorOp)
{
	LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	LossFunctionOp<float>* op_class;
	LossFunctionTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new ManhattanDistanceLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ManhattanDistanceLossTensorOp");

	op_class = new L2NormLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormLossTensorOp");

	op_class = new L2NormLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormLossTensorOp");

	op_class = new NegativeLogLikelihoodLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "NegativeLogLikelihoodLossTensorOp");

	op_class = new MSELossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSELossTensorOp");

  op_class = new MAELossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MAELossTensorOp");

  op_class = new MRSELossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MRSELossTensorOp");

  op_class = new MLELossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MLELossTensorOp");

	op_class = new KLDivergenceMuLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceMuLossTensorOp");

	op_class = new KLDivergenceLogVarLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceLogVarLossTensorOp");

	op_class = new BCEWithLogitsLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "BCEWithLogitsLossTensorOp");

	op_class = new CrossEntropyWithLogitsLossOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CrossEntropyWithLogitsLossTensorOp");

  op_class = new MSERangeLBLossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeLBLossTensorOp");

  op_class = new MSERangeUBLossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeUBLossTensorOp");

  op_class = new KLDivergenceCatLossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceCatLossTensorOp");

  op_class = new MAPELossOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MAPELossTensorOp");

  delete op_class;
  delete op_tensor_class;
}

BOOST_AUTO_TEST_CASE(getTensorParamsLossFunctionOpToLossFunctionTensorOp)
{
	LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	LossFunctionOp<float>* op_class = nullptr;
	std::vector<float> params;

	op_class = new ManhattanDistanceLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new L2NormLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new L2NormLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new NegativeLogLikelihoodLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MSELossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MAELossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MRSELossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MLELossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceMuLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceLogVarLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new BCEWithLogitsLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CrossEntropyWithLogitsLossOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeLBLossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeUBLossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new KLDivergenceCatLossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MAPELossOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  delete op_class;
}

BOOST_AUTO_TEST_CASE(constructorMetricFunctionOpToMetricFunctionTensorOp)
{
  MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
  MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice>* nullPointer = nullptr;
  ptr = new MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorMetricFunctionOpToMetricFunctionTensorOp)
{
  MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice>* ptr = nullptr;
  ptr = new MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(convertOpToTensorOpMetricFunctionOpToMetricFunctionTensorOp)
{
  MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
  MetricFunctionOp<float>* op_class;
  MetricFunctionTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

  op_class = new AccuracyBCOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "AccuracyBCTensorOp");

  op_class = new AccuracyMCMicroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "AccuracyMCMicroTensorOp");

  op_class = new AccuracyMCMacroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "AccuracyMCMacroTensorOp");

  op_class = new PrecisionBCOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PrecisionBCTensorOp");

  op_class = new PrecisionMCMicroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PrecisionMCMicroTensorOp");

  op_class = new PrecisionMCMacroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PrecisionMCMacroTensorOp");

  op_class = new RecallBCOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "RecallBCTensorOp");

  op_class = new RecallMCMicroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "RecallMCMicroTensorOp");

  op_class = new RecallMCMacroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "RecallMCMacroTensorOp");

  op_class = new F1ScoreBCOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "F1ScoreBCTensorOp");

  op_class = new F1ScoreMCMicroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "F1ScoreMCMicroTensorOp");

  op_class = new F1ScoreMCMacroOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "F1ScoreMCMacroTensorOp");

  op_class = new MAEOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MAETensorOp");

  op_class = new CosineSimilarityOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CosineSimilarityTensorOp");

  op_class = new PearsonROp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PearsonRTensorOp");

  op_class = new EuclideanDistOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "EuclideanDistTensorOp");

  op_class = new ManhattanDistOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ManhattanDistTensorOp");

  op_class = new LogarithmicDistOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogarithmicDistTensorOp");

  op_class = new JeffreysAndMatusitaDistOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "JeffreysAndMatusitaDistTensorOp");

  op_class = new PercentDifferenceOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PercentDifferenceTensorOp");

  delete op_class;
  delete op_tensor_class;
}

BOOST_AUTO_TEST_CASE(getTensorParamsMetricFunctionOpToMetricFunctionTensorOp)
{
  MetricFunctionOpToMetricFunctionTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
  MetricFunctionOp<float>* op_class = nullptr;
  std::vector<float> params;

  op_class = new AccuracyBCOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new AccuracyMCMicroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new AccuracyMCMacroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new PrecisionBCOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new PrecisionMCMicroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new PrecisionMCMacroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new RecallBCOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new RecallMCMicroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new RecallMCMacroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new F1ScoreBCOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new F1ScoreMCMicroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new F1ScoreMCMacroOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MAEOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new CosineSimilarityOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new PearsonROp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new EuclideanDistOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new ManhattanDistOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new LogarithmicDistOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new JeffreysAndMatusitaDistOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new PercentDifferenceOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  delete op_class;
}

BOOST_AUTO_TEST_SUITE_END()