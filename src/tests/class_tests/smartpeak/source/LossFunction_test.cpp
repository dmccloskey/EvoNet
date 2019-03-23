/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/LossFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossFunction)

/**
  EuclideanDistanceOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceOp) 
{
  EuclideanDistanceOp<double>* ptrReLU = nullptr;
  EuclideanDistanceOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceOp) 
{
  EuclideanDistanceOp<double>* ptrReLU = nullptr;
	ptrReLU = new EuclideanDistanceOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceOp1) 
{
  EuclideanDistanceOp<float> operation;
  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
  BOOST_CHECK_CLOSE(error(1), 2.44948983, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceOp2)
{
	EuclideanDistanceOp<float> operation;
	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3});

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 1.0, 1e-6);
}

/**
  EuclideanDistanceGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradOp<double>* ptrReLU = nullptr;
  EuclideanDistanceGradOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradOp<double>* ptrReLU = nullptr;
	ptrReLU = new EuclideanDistanceGradOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceGradOp1) 
{
  EuclideanDistanceGradOp<float> operation;
  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -0.267261237, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -0.534522474, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -0.801783681, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 0.408248276, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -0.408248276, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -0.816496551, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceGradOp2)
{
	EuclideanDistanceGradOp<float> operation;
	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2});
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3});

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1), -1.0, 1e-6);
}

/**
  L2NormOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormOp) 
{
  L2NormOp<double>* ptrL2Norm = nullptr;
  L2NormOp<double>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormOp) 
{
  L2NormOp<double>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormOp<double>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp1) 
{
  L2NormOp<float> operation;
  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), 7.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1), 3.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp2)
{
	L2NormOp<float> operation;
	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 0.5, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 0.5, 1e-6);
}

/**
  L2NormGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormGradOp) 
{
  L2NormGradOp<double>* ptrL2Norm = nullptr;
  L2NormGradOp<double>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormGradOp) 
{
  L2NormGradOp<double>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormGradOp<double>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp1) 
{
  L2NormGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -2.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -3.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -2.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp2)
{
	L2NormGradOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1), -1.0, 1e-6);
}

/**
  CrossEntropyOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyOp) 
{
  BCEOp<double>* ptrCrossEntropy = nullptr;
  BCEOp<double>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyOp) 
{
  BCEOp<double>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new BCEOp<double>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp1) 
{
  BCEOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{.1f, .1f, .6f, .2f}, {.1f, .1f, .6f, .2f}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}}); 

  // DEPRECATED
  //Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  //BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
  //BOOST_CHECK_CLOSE(error(1), 2.44948983, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp2)
{
	BCEOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1.0f, 1.0f });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 0.1f, 0.9f });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 2.30257511, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 0.10535942, 1e-6);
}

/**
  CrossEntropyGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyGradOp) 
{
  BCEGradOp<double>* ptrCrossEntropy = nullptr;
  BCEGradOp<double>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyGradOp) 
{
  BCEGradOp<double>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new BCEGradOp<double>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp1) 
{
  BCEGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs);
  y_true.setValues({ { .1f, .1f, .6f, .2f },{ .1f, .1f, .6f, .2f } });
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs);
  y_pred.setValues({ { 1.0f, 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f, 0.0f } });

  // DEPRECATED
  //Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  //BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(0, 1), -1.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(0, 2), -2.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(0, 3), -3.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 0), 1.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 2), -1.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 3), -2.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp2)
{
	BCEGradOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1.0f, 1.0f });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 0.1f, 0.9f });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), -9.99998856, 1e-6);
	BOOST_CHECK_CLOSE(error(1),	-1.11109996, 1e-6);
}

/**
  NegativeLogLikelihoodOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodOp<double>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodOp<double>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodOp<double>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodOp<double>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp1) 
{
  NegativeLogLikelihoodOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0), -3.17805386, 1e-6);
  BOOST_CHECK_CLOSE(error(1), -6.35610771, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp2)
{
	NegativeLogLikelihoodOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 0.0f, 1.0f });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 0.1f, 0.9f });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 0.105360545, 1e-6);
}

/**
  NegativeLogLikelihoodGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradOp<double>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodGradOp<double>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradOp<double>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodGradOp<double>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp1) 
{
  NegativeLogLikelihoodGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  BOOST_CHECK_CLOSE(error(0, 0), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 2), -0.333333343, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 3), -0.25, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), -2.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 2), -0.666666687, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 3), -0.5, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp2)
{
	NegativeLogLikelihoodGradOp<float> operation;

	const int outputs = 4;
	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 0, 1 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 0.1f, 0.9f });

  // DEPRECATED
	//Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	//BOOST_CHECK_CLOSE(error(0), 0.0, 1e-6);
	//BOOST_CHECK_CLOSE(error(1), -1.11110985, 1e-6);
}

/**
  MSEOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEOp) 
{
  MSEOp<double>* ptrMSE = nullptr;
  MSEOp<double>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEOp) 
{
  MSEOp<double>* ptrMSE = nullptr;
	ptrMSE = new MSEOp<double>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEOp1) 
{
  MSEOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  // DEPRECATED
  //Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
  //BOOST_CHECK_CLOSE(error(0), 3.5, 1e-6);
  //BOOST_CHECK_CLOSE(error(1), 1.5, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEOp2)
{
	MSEOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

  // DEPRECATED
	//Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	//BOOST_CHECK_CLOSE(error(0), 0.25, 1e-6);
	//BOOST_CHECK_CLOSE(error(1), 0.25, 1e-6);
}

/**
  MSEGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEGradOp) 
{
  MSEGradOp<double>* ptrMSE = nullptr;
  MSEGradOp<double>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEGradOp) 
{
  MSEGradOp<double>* ptrMSE = nullptr;
	ptrMSE = new MSEGradOp<double>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEGradOp1) 
{
  MSEGradOp<float> operation;

  const int outputs = 4;
  const int batch_size = 2;
  Eigen::Tensor<float, 2> y_true(batch_size, outputs); 
  y_true.setValues({{1, 1, 1, 1}, {2, 2, 2, 2}}); 
  Eigen::Tensor<float, 2> y_pred(batch_size, outputs); 
  y_pred.setValues({{1, 2, 3, 4}, {1, 2, 3, 4}}); 

  // DEPRECATED
  //Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
  //BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(0, 1), -0.5, 1e-6);
  //BOOST_CHECK_CLOSE(error(0, 2), -1.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(0, 3), -1.5, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 0), 0.5, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 2), -0.5, 1e-6);
  //BOOST_CHECK_CLOSE(error(1, 3), -1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEGradOp2)
{
	MSEGradOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

  // DEPRECATED
	//Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	//BOOST_CHECK_CLOSE(error(0), -0.5, 1e-6);
	//BOOST_CHECK_CLOSE(error(1), -0.5, 1e-6);
}

/**
	KLDivergenceMuOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuOp)
{
	KLDivergenceMuOp<double>* ptrKLDivergenceMu = nullptr;
	KLDivergenceMuOp<double>* nullPointerKLDivergenceMu = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuOp)
{
	KLDivergenceMuOp<double>* ptrKLDivergenceMu = nullptr;
	ptrKLDivergenceMu = new KLDivergenceMuOp<double>();
	delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuOp2)
{
	KLDivergenceMuOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 1.5, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 4.0, 1e-6);
}

/**
	KLDivergenceMuGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuGradOp)
{
	KLDivergenceMuGradOp<double>* ptrKLDivergenceMu = nullptr;
	KLDivergenceMuGradOp<double>* nullPointerKLDivergenceMu = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuGradOp)
{
	KLDivergenceMuGradOp<double>* ptrKLDivergenceMu = nullptr;
	ptrKLDivergenceMu = new KLDivergenceMuGradOp<double>();
	delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuGradOp2)
{
	KLDivergenceMuGradOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 4.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 6.0, 1e-6);
}

/**
	KLDivergenceLogVarOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarOp)
{
	KLDivergenceLogVarOp<double>* ptrKLDivergenceLogVar = nullptr;
	KLDivergenceLogVarOp<double>* nullPointerKLDivergenceLogVar = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarOp)
{
	KLDivergenceLogVarOp<double>* ptrKLDivergenceLogVar = nullptr;
	ptrKLDivergenceLogVar = new KLDivergenceLogVarOp<double>();
	delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarOp2)
{
	KLDivergenceLogVarOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 2.1945281, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 8.04276848, 1e-6);
}

/**
	KLDivergenceLogVarGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarGradOp<double>* ptrKLDivergenceLogVar = nullptr;
	KLDivergenceLogVarGradOp<double>* nullPointerKLDivergenceLogVar = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarGradOp<double>* ptrKLDivergenceLogVar = nullptr;
	ptrKLDivergenceLogVar = new KLDivergenceLogVarGradOp<double>();
	delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarGradOp2)
{
	KLDivergenceLogVarGradOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1, 2 });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ 2, 3 });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 3.1945281, 1e-6);
	BOOST_CHECK_CLOSE(error(1), 9.54276848, 1e-6);
}

/**
BCEWithLogitsOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsOp)
{
	BCEWithLogitsOp<double>* ptrBCEWithLogits = nullptr;
	BCEWithLogitsOp<double>* nullPointerBCEWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsOp)
{
	BCEWithLogitsOp<double>* ptrBCEWithLogits = nullptr;
	ptrBCEWithLogits = new BCEWithLogitsOp<double>();
	delete ptrBCEWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsOp2)
{
	BCEWithLogitsOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1.0f, 1.0f });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ -0.954242509f, 0.954242509f });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), 1.28001761, 1e-6); // Why are these different than CrossEntropy?
	BOOST_CHECK_CLOSE(error(1), 0.325775027, 1e-6);
}

/**
BCEWithLogitsGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsGradOp)
{
	BCEWithLogitsGradOp<double>* ptrBCEWithLogits = nullptr;
	BCEWithLogitsGradOp<double>* nullPointerBCEWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsGradOp)
{
	BCEWithLogitsGradOp<double>* ptrBCEWithLogits = nullptr;
	ptrBCEWithLogits = new BCEWithLogitsGradOp<double>();
	delete ptrBCEWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsGradOp2)
{
	BCEWithLogitsGradOp<float> operation;

	const int batch_size = 2;
	Eigen::Tensor<float, 1> y_true(batch_size);
	y_true.setValues({ 1.0f, 1.0f });
	Eigen::Tensor<float, 1> y_pred(batch_size);
	y_pred.setValues({ -0.954242509f, 0.954242509f });

	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
	BOOST_CHECK_CLOSE(error(0), -0.721967578, 1e-6); // Why are these different than CrossEntropy?
	BOOST_CHECK_CLOSE(error(1), -0.278032422, 1e-6);
}

/**
  MSERangeUBOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBOp)
{
  MSERangeUBOp<double>* ptrMSERangeUB = nullptr;
  MSERangeUBOp<double>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBOp)
{
  MSERangeUBOp<double>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBOp<double>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeUBOp)
{
  MSERangeUBOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeUBOp");
}

/**
  MSERangeUBGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBGradOp)
{
  MSERangeUBGradOp<double>* ptrMSERangeUB = nullptr;
  MSERangeUBGradOp<double>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBGradOp)
{
  MSERangeUBGradOp<double>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBGradOp<double>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeUBGradOp)
{
  MSERangeUBGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeUBGradOp");
}

/**
  MSERangeLBOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBOp)
{
  MSERangeLBOp<double>* ptrMSERangeLB = nullptr;
  MSERangeLBOp<double>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBOp)
{
  MSERangeLBOp<double>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBOp<double>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeLBOp)
{
  MSERangeLBOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeLBOp");
}

/**
  MSERangeLBGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBGradOp)
{
  MSERangeLBGradOp<double>* ptrMSERangeLB = nullptr;
  MSERangeLBGradOp<double>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBGradOp)
{
  MSERangeLBGradOp<double>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBGradOp<double>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeLBGradOp)
{
  MSERangeLBGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeLBGradOp");
}

BOOST_AUTO_TEST_SUITE_END()