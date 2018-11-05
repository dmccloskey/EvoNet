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
}

BOOST_AUTO_TEST_CASE(getTensorParamsActivationOpToActivationTensorOp)
{
	ActivationOpToActivationTensorOp<float, Eigen::DefaultDevice> op_to_tensor_op;
	ActivationOp<float>* op_class = new ReLUOp<float>();
	std::vector<float> params = op_to_tensor_op.getTensorParams(op_class);
	BOOST_CHECK_EQUAL(params.size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()