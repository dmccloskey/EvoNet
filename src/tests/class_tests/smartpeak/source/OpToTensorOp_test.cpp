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
	ActivationOp<float>* op_class;
	ActivationTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new ReLUOp<float>(1, 2, 3);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReLUTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ReLUGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReLUGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ELUOp<float>(1.0f, 2.0f, 3.0f, 4.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ELUTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ELUGradOp<float>(1.0f, 2.0f, 3.0f, 4.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ELUGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new SigmoidOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SigmoidTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new SigmoidGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SigmoidGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new TanHOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "TanHTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new TanHGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "TanHGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ReTanHOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReTanHTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ReTanHGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReTanHGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new LinearOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LinearTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new LinearGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LinearGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new InverseOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "InverseTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new InverseGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "InverseGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ExponentialOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ExponentialTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new ExponentialGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ExponentialGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new LogOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new LogGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new PowOp<float>(1.0f,2.0f,3.0f,2.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PowTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new PowGradOp<float>(1.0f, 2.0f, 3.0f, 2.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PowGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new LeakyReLUOp<float>(1.0f, 2.0f, 3.0f, 4.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LeakyReLUTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new LeakyReLUGradOp<float>(1.0f, 2.0f, 3.0f, 4.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LeakyReLUGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new SinOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SinTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new SinGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SinGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new CosOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CosTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);

	op_class = new CosGradOp<float>(1.0f, 2.0f, 3.0f);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CosGradTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getEps(), 1);
  BOOST_CHECK_EQUAL(op_tensor_class->getMin(), 2);
  BOOST_CHECK_EQUAL(op_tensor_class->getMax(), 3);
}

BOOST_AUTO_TEST_CASE(getTensorParamsActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	ActivationOp<float>* op_class = new ReLUOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ReLUGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ELUOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ELUGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new SigmoidOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new SigmoidGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new TanHOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new TanHGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ReTanHOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ReTanHGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new LinearOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new LinearGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new InverseOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new InverseGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ExponentialOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ExponentialGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new LogOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new LogGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new PowOp<float>(2);
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new PowGradOp<float>(2);
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new LeakyReLUOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new LeakyReLUGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new SinOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new SinGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CosOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CosGradOp<float>();
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
	SolverOp<float>* op_class;
	SolverTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new SGDOp<float>(0.1, 0.9, 10.0, 1.0);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SGDTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);

	op_class = new AdamOp<float>(0.001, 0.9, 0.999, 1e-8, 10.0, 1.0);
  op_class->setGradientThreshold(10);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "AdamTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);

	op_class = new DummySolverOp<float>();
  op_class->setGradientThreshold(10);
  op_class->setGradientNoiseSigma(1);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "DummySolverTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);

	op_class = new SGDNoiseOp<float>(0.1, 0.9, 1);
  op_class->setGradientThreshold(10);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SGDNoiseTensorOp");
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientThreshold(), 10);
  BOOST_CHECK_EQUAL(op_tensor_class->getGradientNoiseSigma(), 1);
}

BOOST_AUTO_TEST_CASE(getTensorParamsSolverOpToSolverTensorOp)
{
	SolverOpToSolverTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	SolverOp<float>* op_class = new SGDOp<float>(1, 2);
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 3);
	BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 0);

	op_class = new AdamOp<float>(1, 2, 3, 4);
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 6);
	BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 3); 
	BOOST_CHECK_EQUAL(params[3], 4); BOOST_CHECK_EQUAL(params[4], 0); BOOST_CHECK_EQUAL(params[5], 0);

	op_class = new DummySolverOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new SGDNoiseOp<float>(1, 2, 3);
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 4);
	BOOST_CHECK_EQUAL(params[0], 1); BOOST_CHECK_EQUAL(params[1], 2); BOOST_CHECK_EQUAL(params[2], 0);
	BOOST_CHECK_EQUAL(params[3], 3); 
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
	IntegrationOp<float>* op_class;
	IntegrationTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new SumOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SumTensorOp");

	op_class = new ProdOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdTensorOp");

  op_class = new ProdSCOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdSCTensorOp");

	op_class = new MaxOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxTensorOp");

  op_class = new MinOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MinTensorOp");

	//op_class = new VarOp<float>(); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarTensorOp");

	op_class = new CountOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountTensorOp");

	op_class = new VarModOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationOpToIntegrationTensorOp)
{
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	IntegrationOp<float>* op_class = new SumOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ProdOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new ProdSCOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MaxOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MinOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MeanOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CountOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new VarModOp<float>();
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
	IntegrationErrorOp<float>* op_class;
	IntegrationErrorTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new SumErrorOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SumErrorTensorOp");

	op_class = new ProdErrorOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdErrorTensorOp");

	op_class = new MaxErrorOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxErrorTensorOp");

  op_class = new MinErrorOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MinErrorTensorOp");

	op_class = new MeanErrorOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanErrorTensorOp");

	//op_class = new VarErrorOp<float>(); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarErrorTensorOp");

	op_class = new CountErrorOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountErrorTensorOp");

	op_class = new VarModErrorOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModErrorTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationErrorOpToIntegrationErrorTensorOp)
{
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	IntegrationErrorOp<float>* op_class = new SumErrorOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ProdErrorOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MaxErrorOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MinErrorOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MeanErrorOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CountErrorOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new VarModErrorOp<float>();
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
	IntegrationWeightGradOp<float>* op_class;
	IntegrationWeightGradTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new SumWeightGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SumWeightGradTensorOp");

	op_class = new ProdWeightGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdWeightGradTensorOp");

	op_class = new MaxWeightGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxWeightGradTensorOp");

  op_class = new MinWeightGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MinWeightGradTensorOp");

	op_class = new MeanWeightGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanWeightGradTensorOp");

	//op_class = new VarWeightGradOp<float>(); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarWeightGradTensorOp");

	op_class = new CountWeightGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountWeightGradTensorOp");

	op_class = new VarModWeightGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModWeightGradTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationWeightGradOpToIntegrationWeightGradTensorOp)
{
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	IntegrationWeightGradOp<float>* op_class = new SumWeightGradOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new ProdWeightGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MaxWeightGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MinWeightGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MeanWeightGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CountWeightGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new VarModWeightGradOp<float>();
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
	LossFunctionGradOp<float>* op_class;
	LossFunctionGradTensorOp<float, Eigen::DefaultDevice>* op_tensor_class;

	op_class = new EuclideanDistanceGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "EuclideanDistanceGradTensorOp");

	op_class = new L2NormGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormGradTensorOp");

	op_class = new L2NormGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormGradTensorOp");

	op_class = new NegativeLogLikelihoodGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "NegativeLogLikelihoodGradTensorOp");

	op_class = new MSEGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSEGradTensorOp");

	op_class = new KLDivergenceMuGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceMuGradTensorOp");

	op_class = new KLDivergenceLogVarGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceLogVarGradTensorOp");

	op_class = new BCEWithLogitsGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "BCEWithLogitsGradTensorOp");

	op_class = new CrossEntropyWithLogitsGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CrossEntropyWithLogitsGradTensorOp");

  op_class = new MSERangeLBGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeLBGradTensorOp");

  op_class = new MSERangeUBGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeUBGradTensorOp");

  op_class = new KLDivergenceCatGradOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceCatGradTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsLossFunctionGradOpToLossFunctionGradTensorOp)
{
	LossFunctionGradOpToLossFunctionGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	LossFunctionGradOp<float>* op_class = nullptr;
	std::vector<float> params;

	op_class = new EuclideanDistanceGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new L2NormGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new L2NormGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new NegativeLogLikelihoodGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MSEGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceMuGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceLogVarGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new BCEWithLogitsGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CrossEntropyWithLogitsGradOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeLBGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeUBGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new KLDivergenceCatGradOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);
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

	op_class = new EuclideanDistanceOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "EuclideanDistanceTensorOp");

	op_class = new L2NormOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormTensorOp");

	op_class = new L2NormOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "L2NormTensorOp");

	op_class = new NegativeLogLikelihoodOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "NegativeLogLikelihoodTensorOp");

	op_class = new MSEOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSETensorOp");

	op_class = new KLDivergenceMuOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceMuTensorOp");

	op_class = new KLDivergenceLogVarOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceLogVarTensorOp");

	op_class = new BCEWithLogitsOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "BCEWithLogitsTensorOp");

	op_class = new CrossEntropyWithLogitsOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CrossEntropyWithLogitsTensorOp");

  op_class = new MSERangeLBOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeLBTensorOp");

  op_class = new MSERangeUBOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MSERangeUBTensorOp");

  op_class = new KLDivergenceCatOp<float>();
  op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
  BOOST_CHECK_EQUAL(op_tensor_class->getName(), "KLDivergenceCatTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsLossFunctionOpToLossFunctionTensorOp)
{
	LossFunctionOpToLossFunctionTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	LossFunctionOp<float>* op_class = nullptr;
	std::vector<float> params;

	op_class = new EuclideanDistanceOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new L2NormOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new L2NormOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new NegativeLogLikelihoodOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new MSEOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceMuOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new KLDivergenceLogVarOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new BCEWithLogitsOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	op_class = new CrossEntropyWithLogitsOp<float>();
	params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeLBOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new MSERangeUBOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);

  op_class = new KLDivergenceCatOp<float>();
  params = op_to_tensor_op.getTensorParams(op_class);
  BOOST_CHECK_EQUAL(params.size(), 0);
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
}

BOOST_AUTO_TEST_SUITE_END()