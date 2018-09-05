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

BOOST_AUTO_TEST_SUITE_END()