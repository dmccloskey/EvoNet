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

BOOST_AUTO_TEST_CASE(P_UnitScaleFunctor)
{
	Eigen::Tensor<float, 2> data(2, 2);
	data.setValues({{ 0, 2 }, { 3, 4 }});
	UnitScaleFunctor<float> unit_scale(data);
	BOOST_CHECK_CLOSE(unit_scale.getUnitScale(), 0.25, 1e-6);

	Eigen::Tensor<float, 2> data_test = data.unaryExpr(UnitScaleFunctor<float>(data));
	
	BOOST_CHECK_CLOSE(data_test(0, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(data_test(1, 1), 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(P_LinearScaleFunctor)
{
	Eigen::Tensor<float, 2> data(2, 2);
	data.setValues({ { 0, 2 }, { 4, 8 } });

	Eigen::Tensor<float, 2> data_test = data.unaryExpr(LinearScaleFunctor<float>(0, 8, -1, 1));

	BOOST_CHECK_CLOSE(data_test(0, 0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(data_test(0, 1), -0.5, 1e-6);
	BOOST_CHECK_CLOSE(data_test(1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(data_test(1, 1), 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(P_LinearScale)
{
  Eigen::Tensor<float, 3> data(2, 2, 2);
  data.setValues({
    {{ 0, 2 }, { 4, 8 }},
    {{ 1, 1 }, { 3, 5 }}
    });

  // Test default initialization for the domain and setters
  LinearScale<float, 3> linearScale1(-1, 1);
  linearScale1.setDomain(0, 8);
  Eigen::Tensor<float, 3> data_test = linearScale1(data);

  BOOST_CHECK_CLOSE(data_test(0, 0, 0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 0, 1), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 0), -0.75, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 1), -0.75, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 0), -0.25, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 1), 0.25, 1e-6);

  // Test with manual domain and range initialization
  LinearScale<float, 3> linearScale(0, 8, -1, 1);
  data_test = linearScale(data);

  BOOST_CHECK_CLOSE(data_test(0, 0, 0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 0, 1), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 0), -0.75, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 1), -0.75, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 0), -0.25, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 1), 0.25, 1e-6);

  // Test with domain calculation and range initialization
  LinearScale<float, 3> linearScale2(data, -1, 1);
  data_test = linearScale2(data);

  BOOST_CHECK_CLOSE(data_test(0, 0, 0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 0, 1), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 1), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 0), -0.75, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 1), -0.75, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 0), -0.25, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 1), 0.25, 1e-6);
}

BOOST_AUTO_TEST_CASE(P_Standardize)
{
  Eigen::Tensor<float, 3> data(2, 2, 2);
  data.setValues({
    {{ 0, 2 }, { 4, 8 }},
    {{ 1, 3 }, { 3, 5 }}
    });

  // Test default initialization with setters and getters
  Standardize<float, 3> standardize1;
  standardize1.setMeanAndVar(1, 2);
  BOOST_CHECK_CLOSE(standardize1.getMean(), 1, 1e-6);
  BOOST_CHECK_CLOSE(standardize1.getVar(), 2, 1e-6);
  standardize1.setMeanAndVar(data);
  BOOST_CHECK_CLOSE(standardize1.getMean(), 3.25, 1e-6);
  BOOST_CHECK_CLOSE(standardize1.getVar(), 6.21428585, 1e-6);

  // Test with data initialization and getters
  Standardize<float, 3> standardize(data);
  BOOST_CHECK_CLOSE(standardize.getMean(), 3.25, 1e-6);
  BOOST_CHECK_CLOSE(standardize.getVar(), 6.21428585, 1e-6);

  // Test operator
  Eigen::Tensor<float, 3> data_test = standardize(data);
  BOOST_CHECK_CLOSE(data_test(0, 0, 0), -1.30373025, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 0, 1), -0.501434684, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 0), 0.300860822, 1e-6);
  BOOST_CHECK_CLOSE(data_test(0, 1, 1), 1.90545189, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 0), -0.902582467, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 0, 1), -0.100286946, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 0), -0.100286946, 1e-6);
  BOOST_CHECK_CLOSE(data_test(1, 1, 1), 0.702008605, 1e-6);
}

BOOST_AUTO_TEST_CASE(P_MakeShuffleMatrix)
{
  const int shuffle_dim_size = 8;
  std::vector<int> indices = { 0, 1, 2, 3, 4, 5, 6, 7 };

  // Test default initialization with setters and getters
  MakeShuffleMatrix<float> shuffle1;
  shuffle1.setIndices(8);
  BOOST_CHECK(shuffle1.getIndices() != indices);
  for (int i = 0; i < shuffle_dim_size; ++i) {
    BOOST_CHECK_GE(shuffle1.getIndices().at(i), 0);
    BOOST_CHECK_LE(shuffle1.getIndices().at(i), 7);
  }
  shuffle1.setShuffleMatrix(true);
  //std::cout << "Shuffle_matrix\n" << shuffle1.getShuffleMatrix() << std::endl;
  for (int i = 0; i < shuffle_dim_size; ++i) {
    Eigen::Tensor<float, 0> row_sum = shuffle1.getShuffleMatrix().chip(i, 0).sum();
    BOOST_CHECK_EQUAL(row_sum(0), 1);
  }

  // Test initialization with dim size
  MakeShuffleMatrix<float> shuffle2(shuffle_dim_size, true);
  BOOST_CHECK(shuffle2.getIndices() != indices);
  for (int i = 0; i < shuffle_dim_size; ++i) {
    BOOST_CHECK_GE(shuffle2.getIndices().at(i), 0);
    BOOST_CHECK_LE(shuffle2.getIndices().at(i), 7);
  }

  // Test initialization with indices to use
  MakeShuffleMatrix<float> shuffle3(indices, true);
  BOOST_CHECK(shuffle3.getIndices() == indices);
  //std::cout << "Shuffle_matrix\n" << shuffle3.getShuffleMatrix() << std::endl;
  for (int i = 0; i < shuffle_dim_size; ++i) {
    BOOST_CHECK_EQUAL(shuffle3.getShuffleMatrix()(i, i), 1);
    Eigen::Tensor<float, 0> row_sum = shuffle3.getShuffleMatrix().chip(i, 0).sum();
    BOOST_CHECK_EQUAL(row_sum(0), 1);
  }

  // Test row/column shuffling on toy data
  Eigen::Tensor<float, 2> data(2, 3);
  data.setValues({ {1,2,3},{4,5,6} });
  MakeShuffleMatrix<float> shuffle_col(std::vector<int>({1,2,0}), true);
  Eigen::Tensor<float, 2> col_shuffle = data;
  shuffle_col(col_shuffle, true);
  BOOST_CHECK_EQUAL(col_shuffle(0, 0), 2);
  BOOST_CHECK_EQUAL(col_shuffle(0, 1), 3);
  BOOST_CHECK_EQUAL(col_shuffle(0, 2), 1);
  BOOST_CHECK_EQUAL(col_shuffle(1, 0), 5);
  BOOST_CHECK_EQUAL(col_shuffle(1, 1), 6);
  BOOST_CHECK_EQUAL(col_shuffle(1, 2), 4);
  MakeShuffleMatrix<float> shuffle_row(std::vector<int>({ 1,0 }), false);
  Eigen::Tensor<float, 2> row_shuffle = data;
  shuffle_row(row_shuffle, false);
  BOOST_CHECK_EQUAL(row_shuffle(0, 0), 4);
  BOOST_CHECK_EQUAL(row_shuffle(0, 1), 5);
  BOOST_CHECK_EQUAL(row_shuffle(0, 2), 6);
  BOOST_CHECK_EQUAL(row_shuffle(1, 0), 1);
  BOOST_CHECK_EQUAL(row_shuffle(1, 1), 2);
  BOOST_CHECK_EQUAL(row_shuffle(1, 2), 3);

  // Test row/column shuffling on toy data
  Eigen::Tensor<double, 2> data_db(2, 3);
  data_db.setValues({ {1,2,3},{4,5,6} });
  MakeShuffleMatrix<double> shuffle_col_db(std::vector<int>({ 1,2,0 }), true);
  Eigen::Tensor<double, 2> col_shuffle_db = data_db;
  shuffle_col_db(col_shuffle_db, true);
  BOOST_CHECK_EQUAL(col_shuffle_db(0, 0), 2);
  BOOST_CHECK_EQUAL(col_shuffle_db(0, 1), 3);
  BOOST_CHECK_EQUAL(col_shuffle_db(0, 2), 1);
  BOOST_CHECK_EQUAL(col_shuffle_db(1, 0), 5);
  BOOST_CHECK_EQUAL(col_shuffle_db(1, 1), 6);
  BOOST_CHECK_EQUAL(col_shuffle_db(1, 2), 4);
  MakeShuffleMatrix<double> shuffle_row_db(std::vector<int>({ 1,0 }), false);
  Eigen::Tensor<double, 2> row_shuffle_db = data_db;
  shuffle_row_db(row_shuffle_db, false);
  BOOST_CHECK_EQUAL(row_shuffle_db(0, 0), 4);
  BOOST_CHECK_EQUAL(row_shuffle_db(0, 1), 5);
  BOOST_CHECK_EQUAL(row_shuffle_db(0, 2), 6);
  BOOST_CHECK_EQUAL(row_shuffle_db(1, 0), 1);
  BOOST_CHECK_EQUAL(row_shuffle_db(1, 1), 2);
  BOOST_CHECK_EQUAL(row_shuffle_db(1, 2), 3);
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

BOOST_AUTO_TEST_CASE(P_GaussianMixture)
{
	// TODO
}

BOOST_AUTO_TEST_CASE(P_SwissRoll)
{
	// TODO
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