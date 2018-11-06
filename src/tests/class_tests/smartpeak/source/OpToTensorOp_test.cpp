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

	op_class = new ReLUOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReLUTensorOp");

	op_class = new ReLUGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReLUGradTensorOp");

	op_class = new ELUOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ELUTensorOp");

	op_class = new ELUGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ELUGradTensorOp");

	op_class = new SigmoidOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SigmoidTensorOp");

	op_class = new SigmoidGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SigmoidGradTensorOp");

	op_class = new TanHOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "TanHTensorOp");

	op_class = new TanHGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "TanHGradTensorOp");

	op_class = new ReTanHOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReTanHTensorOp");

	op_class = new ReTanHGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ReTanHGradTensorOp");

	op_class = new LinearOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LinearTensorOp");

	op_class = new LinearGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LinearGradTensorOp");

	op_class = new InverseOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "InverseTensorOp");

	op_class = new InverseGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "InverseGradTensorOp");

	op_class = new ExponentialOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ExponentialTensorOp");

	op_class = new ExponentialGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ExponentialGradTensorOp");

	op_class = new LogOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogTensorOp");

	op_class = new LogGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LogGradTensorOp");

	op_class = new PowOp<float>(2);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PowTensorOp");

	op_class = new PowGradOp<float>(2);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "PowGradTensorOp");

	op_class = new LeakyReLUOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LeakyReLUTensorOp");

	op_class = new LeakyReLUGradOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "LeakyReLUGradTensorOp");
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

	op_class = new SGDOp<float>(0.1, 0.9);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SGDTensorOp");

	op_class = new AdamOp<float>(0.001, 0.9, 0.999, 1e-8);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "AdamTensorOp");

	op_class = new DummySolverOp<float>();
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "DummySolverTensorOp");

	op_class = new SGDNoiseOp<float>(0.1, 0.9, 0.1);
	op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	BOOST_CHECK_EQUAL(op_tensor_class->getName(), "SGDNoiseTensorOp");
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

	//op_class = new ProdOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdTensorOp");

	//op_class = new MaxOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxTensorOp");

	//op_class = new MeanOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanTensorOp");

	//op_class = new VarOp<float>(); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarTensorOp");

	//op_class = new CountOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountTensorOp");

	//op_class = new VarModOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationOpToIntegrationTensorOp)
{
	IntegrationOpToIntegrationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	IntegrationOp<float>* op_class = new SumOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new ProdOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new MaxOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new MeanOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new CountOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarModOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);
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

	//op_class = new ProdErrorOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdErrorTensorOp");

	//op_class = new MaxErrorOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxErrorTensorOp");

	//op_class = new MeanErrorOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanErrorTensorOp");

	//op_class = new VarErrorOp<float>(); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarErrorTensorOp");

	//op_class = new CountErrorOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountErrorTensorOp");

	//op_class = new VarModErrorOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModErrorTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationErrorOpToIntegrationErrorTensorOp)
{
	IntegrationErrorOpToIntegrationErrorTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	IntegrationErrorOp<float>* op_class = new SumErrorOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new ProdErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new MaxErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new MeanErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new CountErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarModErrorOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);
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

	//op_class = new ProdWeightGradOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "ProdWeightGradTensorOp");

	//op_class = new MaxWeightGradOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MaxWeightGradTensorOp");

	//op_class = new MeanWeightGradOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "MeanWeightGradTensorOp");

	//op_class = new VarWeightGradOp<float>(); //TODO...
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarWeightGradTensorOp");

	//op_class = new CountWeightGradOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "CountWeightGradTensorOp");

	//op_class = new VarModWeightGradOp<float>();
	//op_tensor_class = op_to_tensor_op.convertOpToTensorOp(op_class);
	//BOOST_CHECK_EQUAL(op_tensor_class->getName(), "VarModWeightGradTensorOp");
}

BOOST_AUTO_TEST_CASE(getTensorParamsIntegrationWeightGradOpToIntegrationWeightGradTensorOp)
{
	IntegrationWeightGradOpToIntegrationWeightGradTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;

	IntegrationWeightGradOp<float>* op_class = new SumWeightGradOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new ProdWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new MaxWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new MeanWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new CountWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);

	//op_class = new VarModWeightGradOp<float>();
	//params = op_to_tensor_op.getTensorParams(op_class);
	//BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()