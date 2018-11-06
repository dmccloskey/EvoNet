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

	// TODO...
}

BOOST_AUTO_TEST_CASE(getTensorParamsActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	ActivationOp<float>* op_class = new ReLUOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);
	// TODO...
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

	SolverOp<float>* op_class = new SGDOp<float>(0, 1);
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 2);
	BOOST_CHECK_EQUAL(params[0], 0); BOOST_CHECK_EQUAL(params[1], 1);

	op_class = new AdamOp<float>(0, 1, 2, 3);
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 2);
	BOOST_CHECK_EQUAL(params[0], 0); BOOST_CHECK_EQUAL(params[1], 1);

	op_class = new DummySolverOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 2);
	BOOST_CHECK_EQUAL(params[0], 0); BOOST_CHECK_EQUAL(params[1], 1);

	op_class = new SGDNoiseOp<float>(0, 1, 2);
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 2);
	BOOST_CHECK_EQUAL(params[0], 0); BOOST_CHECK_EQUAL(params[1], 1);
}

BOOST_AUTO_TEST_SUITE_END()