/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Preprocessing test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/Preprocessing.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(preprocessing)

BOOST_AUTO_TEST_CASE(P_selectRandomElement)
{
	// [TODO: make test; currently, combined with selectRandomNode1]
}

BOOST_AUTO_TEST_CASE(P_UnitScale)
{
	Eigen::Tensor<float, 2> data(2, 2);
	data.setValues({{ 0, 2 }, { 3, 4 }});
	UnitScale<float> unit_scale(data);
	BOOST_CHECK_CLOSE(unit_scale.getUnitScale(), 0.25, 1e-6);

	Eigen::Tensor<float, 2> data_test = data.unaryExpr(UnitScale<float>(data));
	
	BOOST_CHECK_CLOSE(data_test(0, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(data_test(1, 1), 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(P_LinearScale)
{
	Eigen::Tensor<float, 2> data(2, 2);
	data.setValues({ { 0, 2 }, { 4, 8 } });

	Eigen::Tensor<float, 2> data_test = data.unaryExpr(LinearScale<float>(0, 8, -1, 1));

	BOOST_CHECK_CLOSE(data_test(0, 0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(data_test(0, 1), -0.5, 1e-6);
	BOOST_CHECK_CLOSE(data_test(1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(data_test(1, 1), 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(P_LabelSmoother)
{
	Eigen::Tensor<float, 1> data(2);
	data.setValues({ 0, 1 });

	Eigen::Tensor<float, 1> data_test = data.unaryExpr(LabelSmoother<float>(0.1, 0.2));

	BOOST_CHECK_CLOSE(data_test(0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(data_test(1), 0.8, 1e-4);
}

BOOST_AUTO_TEST_CASE(P_OneHotEncoder)
{
	// TODO
}
BOOST_AUTO_TEST_CASE(SFcheckNan)
{
	Eigen::Tensor<float, 1> values(2);
	values.setConstant(5.0f);
	Eigen::Tensor<float, 1> test(2);

	// control
	test = values.unaryExpr(std::ptr_fun(checkNan<float>));
	BOOST_CHECK_CLOSE(test(0), 5.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 5.0, 1e-3);

	// test
	values(0) = NAN; //NaN
	values(1) = INFINITY; //infinity
	test = values.unaryExpr(std::ptr_fun(checkNan<float>));
	BOOST_CHECK_CLOSE(test(0), NAN, 1e-3);
	BOOST_CHECK_CLOSE(test(1), INFINITY, 1e-3);
}

BOOST_AUTO_TEST_CASE(SFsubstituteNanInf)
{
	Eigen::Tensor<float, 1> values(3);
	values.setConstant(5.0f);
	Eigen::Tensor<float, 1> test(3);

	// control
	test = values.unaryExpr(std::ptr_fun(substituteNanInf<float>));
	BOOST_CHECK_CLOSE(test(0), 5.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 5.0, 1e-3);

	// test
	values(0) = NAN; //NaN
	values(1) = INFINITY; //infinity
	values(2) = -INFINITY; //infinity
	test = values.unaryExpr(std::ptr_fun(substituteNanInf<float>));
	BOOST_CHECK_CLOSE(test(0), 0.0, 1e-3);
	BOOST_CHECK_CLOSE(test(1), 1e9, 1e-3);
	BOOST_CHECK_CLOSE(test(2), -1e9, 1e-3);
}

BOOST_AUTO_TEST_CASE(SFClipOp)
{
	Eigen::Tensor<float, 1> net_input(3);
	net_input.setValues({ 0.0f, 1.0f, 0.5f });

	// test input
	Eigen::Tensor<float, 1> result = net_input.unaryExpr(ClipOp<float>(0.1f, 0.0f, 1.0f));
	BOOST_CHECK_CLOSE(result(0), 0.1, 1e-3);
	BOOST_CHECK_CLOSE(result(1), 0.9, 1e-3);
	BOOST_CHECK_CLOSE(result(2), 0.5, 1e-3);
}

BOOST_AUTO_TEST_CASE(SFRandBinaryOp)
{
	Eigen::Tensor<float, 1> net_input(2);
	net_input.setValues({ 2.0f, 2.0f });
	Eigen::Tensor<float, 1> result;

	// test input
	result = net_input.unaryExpr(RandBinaryOp<float>(0.0f));
	BOOST_CHECK_CLOSE(result(0), 2.0, 1e-3);
	BOOST_CHECK_CLOSE(result(1), 2.0, 1e-3);
	result = net_input.unaryExpr(RandBinaryOp<float>(1.0f));
	BOOST_CHECK_CLOSE(result(0), 0.0, 1e-3);
	BOOST_CHECK_CLOSE(result(1), 0.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(assertClose)
{
	BOOST_CHECK(!assert_close<float>(1.1, 1.2, 1e-4, 1e-4));
	BOOST_CHECK(assert_close<float>(1.1, 1.2, 1, 1));
}

BOOST_AUTO_TEST_CASE(P_GumbelSampler)
{
	Eigen::Tensor<float, 2> gumbel_samples = GumbelSampler<float>(2, 3);
	BOOST_CHECK_LE(gumbel_samples(0, 0), 10);
	BOOST_CHECK_GE(gumbel_samples(0, 0), -10);
	BOOST_CHECK_LE(gumbel_samples(1, 2), 10);
	BOOST_CHECK_GE(gumbel_samples(1, 2), -10);
	std::cout << gumbel_samples << std::endl;
}

BOOST_AUTO_TEST_CASE(P_GaussianSampler)
{
	Eigen::Tensor<float, 2> gaussian_samples = GaussianSampler<float>(2, 3);
	BOOST_CHECK_LE(gaussian_samples(0, 0), 2);
	BOOST_CHECK_GE(gaussian_samples(0, 0), -2);
	BOOST_CHECK_LE(gaussian_samples(1, 2), 2);
	BOOST_CHECK_GE(gaussian_samples(1, 2), -2);
	std::cout << gaussian_samples << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()