/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunction3 test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/LossFunction3.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossFunction3)
//
///**
//  EuclideanDistanceOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceOp) 
//{
//  EuclideanDistanceOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//  EuclideanDistanceOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceOp) 
//{
//  EuclideanDistanceOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new EuclideanDistanceOp<float, Eigen::DefaultDevice>();
//  delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceOp1) 
//{
//  EuclideanDistanceOp<float, Eigen::DefaultDevice> operation;
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{1, 0}, {2, 0}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{2, 0}, {3, 0}}); 
//
//  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
//  BOOST_CHECK_CLOSE(error(1), 2.44948983, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceOp2)
//{
//	EuclideanDistanceOp<float, Eigen::DefaultDevice> operation;
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3});
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 1.0, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 1.0, 1e-6);
//}
//
///**
//  EuclideanDistanceGradOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceGradOp) 
//{
//  EuclideanDistanceGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//  EuclideanDistanceGradOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
//  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
//}
//
//BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceGradOp) 
//{
//  EuclideanDistanceGradOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
//	ptrReLU = new EuclideanDistanceGradOp<float, Eigen::DefaultDevice>();
//  delete ptrReLU;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceGradOp1) 
//{
//  EuclideanDistanceGradOp<float, Eigen::DefaultDevice> operation;
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{1, 0}, {2, 0}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{2, 0}, {3, 0}}); 
//
//  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 1), -0.267261237, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 2), -0.534522474, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 3), -0.801783681, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 0), 0.408248276, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 2), -0.408248276, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 3), -0.816496551, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceGradOp2)
//{
//	EuclideanDistanceGradOp<float, Eigen::DefaultDevice> operation;
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2});
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3});
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), -1.0, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), -1.0, 1e-6);
//}
//
///**
//  L2NormOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorL2NormOp) 
//{
//  L2NormOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
//  L2NormOp<float, Eigen::DefaultDevice>* nullPointerL2Norm = nullptr;
//  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
//}
//
//BOOST_AUTO_TEST_CASE(destructorL2NormOp) 
//{
//  L2NormOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
//	ptrL2Norm = new L2NormOp<float, Eigen::DefaultDevice>();
//  delete ptrL2Norm;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp1) 
//{
//  L2NormOp<float, Eigen::DefaultDevice> operation;
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{1, 0}, {2, 0}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{2, 0}, {3, 0}}); 
//
//  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0), 7.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1), 3.0, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp2)
//{
//	L2NormOp<float, Eigen::DefaultDevice> operation;
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3 });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 0.5, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 0.5, 1e-6);
//}
//
///**
//  L2NormGradOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorL2NormGradOp) 
//{
//  L2NormGradOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
//  L2NormGradOp<float, Eigen::DefaultDevice>* nullPointerL2Norm = nullptr;
//  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
//}
//
//BOOST_AUTO_TEST_CASE(destructorL2NormGradOp) 
//{
//  L2NormGradOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
//	ptrL2Norm = new L2NormGradOp<float, Eigen::DefaultDevice>();
//  delete ptrL2Norm;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp1) 
//{
//  L2NormGradOp<float, Eigen::DefaultDevice> operation;
//
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{1, 0}, {2, 0}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{2, 0}, {3, 0}}); 
//
//  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 1), -1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 2), -2.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 3), -3.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 0), 1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 2), -1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 3), -2.0, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp2)
//{
//	L2NormGradOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3 });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), -1.0, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), -1.0, 1e-6);
//}
//
///**
//  CrossEntropyOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorCrossEntropyOp) 
//{
//  BCEOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
//  BCEOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropy = nullptr;
//  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
//}
//
//BOOST_AUTO_TEST_CASE(destructorCrossEntropyOp) 
//{
//  BCEOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
//	ptrCrossEntropy = new BCEOp<float, Eigen::DefaultDevice>();
//  delete ptrCrossEntropy;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp1) 
//{
//  BCEOp<float, Eigen::DefaultDevice> operation;
//
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{.1f, .1f, .6f, .2f}, {.1f, .1f, .6f, .2f}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}}); 
//
//  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0), 3.7416575, 1e-6);
//  BOOST_CHECK_CLOSE(error(1), 2.44948983, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp2)
//{
//	BCEOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1.0f, 1.0f });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 0.1f, 0.9f });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 2.30257511, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 0.10535942, 1e-6);
//}
//
///**
//  CrossEntropyGradOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorCrossEntropyGradOp) 
//{
//  BCEGradOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
//  BCEGradOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropy = nullptr;
//  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
//}
//
//BOOST_AUTO_TEST_CASE(destructorCrossEntropyGradOp) 
//{
//  BCEGradOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
//	ptrCrossEntropy = new BCEGradOp<float, Eigen::DefaultDevice>();
//  delete ptrCrossEntropy;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp1) 
//{
//  BCEGradOp<float, Eigen::DefaultDevice> operation;
//
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size);
//  y_true.setValues({ { .1f, .1f, .6f, .2f },{ .1f, .1f, .6f, .2f } });
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size);
//  y_pred.setValues({ { 1.0f, 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f, 0.0f } });
//
//  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0, 0), 0.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 1), -1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 2), -2.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 3), -3.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 0), 1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 1), 0.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 2), -1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 3), -2.0, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp2)
//{
//	BCEGradOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1.0f, 1.0f });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 0.1f, 0.9f });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), -9.99998856, 1e-6);
//	BOOST_CHECK_CLOSE(error(1),	-1.11109996, 1e-6);
//}
//
///**
//  NegativeLogLikelihoodOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodOp) 
//{
//  NegativeLogLikelihoodOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
//  NegativeLogLikelihoodOp<float, Eigen::DefaultDevice>* nullPointerNegativeLogLikelihood = nullptr;
//  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
//}
//
//BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodOp) 
//{
//  NegativeLogLikelihoodOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
//	ptrNegativeLogLikelihood = new NegativeLogLikelihoodOp<float, Eigen::DefaultDevice>();
//  delete ptrNegativeLogLikelihood;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp1) 
//{
//  NegativeLogLikelihoodOp<float, Eigen::DefaultDevice> operation;
//
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{1, 0}, {2, 0}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{2, 0}, {3, 0}}); 
//
//  Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0), -3.17805386, 1e-6);
//  BOOST_CHECK_CLOSE(error(1), -6.35610771, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp2)
//{
//	NegativeLogLikelihoodOp<float, Eigen::DefaultDevice> operation(1.0);
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 0.0f, 1.0f });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 0.1f, 0.9f });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 0.0, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 0.105360545, 1e-6);
//}
//
///**
//  NegativeLogLikelihoodGradOp Tests
//*/ 
//BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodGradOp) 
//{
//  NegativeLogLikelihoodGradOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
//  NegativeLogLikelihoodGradOp<float, Eigen::DefaultDevice>* nullPointerNegativeLogLikelihood = nullptr;
//  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
//}
//
//BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodGradOp) 
//{
//  NegativeLogLikelihoodGradOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
//	ptrNegativeLogLikelihood = new NegativeLogLikelihoodGradOp<float, Eigen::DefaultDevice>();
//  delete ptrNegativeLogLikelihood;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp1) 
//{
//  NegativeLogLikelihoodGradOp<float, Eigen::DefaultDevice> operation;
//
//  const int memory_size = 2;
//  const int batch_size = 2;
//  Eigen::Tensor<float, 2> y_true(batch_size, memory_size); 
//  y_true.setValues({{1, 0}, {2, 0}}); 
//  Eigen::Tensor<float, 2> y_pred(batch_size, memory_size); 
//  y_pred.setValues({{2, 0}, {3, 0}}); 
//
//  Eigen::Tensor<float, 2> error = operation(y_pred, y_true);
//  BOOST_CHECK_CLOSE(error(0, 0), -1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 1), -0.5, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 2), -0.333333343, 1e-6);
//  BOOST_CHECK_CLOSE(error(0, 3), -0.25, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 0), -2.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 1), -1.0, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 2), -0.666666687, 1e-6);
//  BOOST_CHECK_CLOSE(error(1, 3), -0.5, 1e-6);
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp2)
//{
//	NegativeLogLikelihoodGradOp<float, Eigen::DefaultDevice> operation(1.0);
//
//	const int memory_size = 2;
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 0, 1 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 0.1f, 0.9f });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 0.0, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), -1.11110985, 1e-6);
//}
//
/**
  MSEOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEOp) 
{
  MSEOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  MSEOp<float, Eigen::DefaultDevice>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEOp) 
{
  MSEOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
	ptrMSE = new MSEOp<float, Eigen::DefaultDevice>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEOp1) 
{
  MSEOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 2}, {0, 0}},
		{{1, 2}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0};
	Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0.25, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 1.25, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  MSEGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEGradOp) 
{
  MSEGradOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  MSEGradOp<float, Eigen::DefaultDevice>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEGradOp) 
{
  MSEGradOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
	ptrMSE = new MSEGradOp<float, Eigen::DefaultDevice>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEGradOp1) 
{
  MSEGradOp<float, Eigen::DefaultDevice> operation;

  const int memory_size = 2;
  const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
  Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size); 
	y_true.setValues({
		{{1, 2}, {0, 0}}, 
		{{1, 2}, {0, 0}}
		});
  Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{1, 1}, {0, 0}},
		{{2, 2}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0, 0), -0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -0.5, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

///**
//	KLDivergenceMuOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuOp)
//{
//	KLDivergenceMuOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
//	KLDivergenceMuOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceMu = nullptr;
//	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
//}
//
//BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuOp)
//{
//	KLDivergenceMuOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
//	ptrKLDivergenceMu = new KLDivergenceMuOp<float, Eigen::DefaultDevice>();
//	delete ptrKLDivergenceMu;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuOp2)
//{
//	KLDivergenceMuOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3 });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 1.5, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 4.0, 1e-6);
//}
//
///**
//	KLDivergenceMuGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuGradOp)
//{
//	KLDivergenceMuGradOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
//	KLDivergenceMuGradOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceMu = nullptr;
//	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
//}
//
//BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuGradOp)
//{
//	KLDivergenceMuGradOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
//	ptrKLDivergenceMu = new KLDivergenceMuGradOp<float, Eigen::DefaultDevice>();
//	delete ptrKLDivergenceMu;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuGradOp2)
//{
//	KLDivergenceMuGradOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3 });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 4.0, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 6.0, 1e-6);
//}
//
///**
//	KLDivergenceLogVarOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarOp)
//{
//	KLDivergenceLogVarOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
//	KLDivergenceLogVarOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceLogVar = nullptr;
//	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
//}
//
//BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarOp)
//{
//	KLDivergenceLogVarOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
//	ptrKLDivergenceLogVar = new KLDivergenceLogVarOp<float, Eigen::DefaultDevice>();
//	delete ptrKLDivergenceLogVar;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarOp2)
//{
//	KLDivergenceLogVarOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3 });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 2.1945281, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 8.04276848, 1e-6);
//}
//
///**
//	KLDivergenceLogVarGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarGradOp)
//{
//	KLDivergenceLogVarGradOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
//	KLDivergenceLogVarGradOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceLogVar = nullptr;
//	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
//}
//
//BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarGradOp)
//{
//	KLDivergenceLogVarGradOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
//	ptrKLDivergenceLogVar = new KLDivergenceLogVarGradOp<float, Eigen::DefaultDevice>();
//	delete ptrKLDivergenceLogVar;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarGradOp2)
//{
//	KLDivergenceLogVarGradOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1, 2 });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ 2, 3 });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 3.1945281, 1e-6);
//	BOOST_CHECK_CLOSE(error(1), 9.54276848, 1e-6);
//}
//
///**
//BCEWithLogitsOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsOp)
//{
//	BCEWithLogitsOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
//	BCEWithLogitsOp<float, Eigen::DefaultDevice>* nullPointerBCEWithLogits = nullptr;
//	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
//}
//
//BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsOp)
//{
//	BCEWithLogitsOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
//	ptrBCEWithLogits = new BCEWithLogitsOp<float, Eigen::DefaultDevice>();
//	delete ptrBCEWithLogits;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsOp2)
//{
//	BCEWithLogitsOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1.0f, 1.0f });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ -0.954242509f, 0.954242509f });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), 1.28001761, 1e-6); // Why are these different than CrossEntropy?
//	BOOST_CHECK_CLOSE(error(1), 0.325775027, 1e-6);
//}
//
///**
//BCEWithLogitsGradOp Tests
//*/
//BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsGradOp)
//{
//	BCEWithLogitsGradOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
//	BCEWithLogitsGradOp<float, Eigen::DefaultDevice>* nullPointerBCEWithLogits = nullptr;
//	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
//}
//
//BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsGradOp)
//{
//	BCEWithLogitsGradOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
//	ptrBCEWithLogits = new BCEWithLogitsGradOp<float, Eigen::DefaultDevice>();
//	delete ptrBCEWithLogits;
//}
//
//BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsGradOp2)
//{
//	BCEWithLogitsGradOp<float, Eigen::DefaultDevice> operation;
//
//	const int batch_size = 2;
//	Eigen::Tensor<float, 1> y_true(batch_size);
//	y_true.setValues({ 1.0f, 1.0f });
//	Eigen::Tensor<float, 1> y_pred(batch_size);
//	y_pred.setValues({ -0.954242509f, 0.954242509f });
//
//	Eigen::Tensor<float, 1> error = operation(y_pred, y_true);
//	BOOST_CHECK_CLOSE(error(0), -0.721967578, 1e-6); // Why are these different than CrossEntropy?
//	BOOST_CHECK_CLOSE(error(1), -0.278032422, 1e-6);
//}

BOOST_AUTO_TEST_SUITE_END()