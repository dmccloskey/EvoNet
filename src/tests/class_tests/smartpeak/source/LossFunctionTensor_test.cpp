/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunctionTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/LossFunctionTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossFunctionTensor)

/**
  EuclideanDistanceOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceOp) 
{
  EuclideanDistanceTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  EuclideanDistanceTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceOp) 
{
  EuclideanDistanceTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new EuclideanDistanceTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceOp) 
{
  EuclideanDistanceTensorOp<float, Eigen::DefaultDevice> operation;

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

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 1, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0), 3, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  EuclideanDistanceGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  EuclideanDistanceGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new EuclideanDistanceGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionEuclideanDistanceGradOp) 
{
  EuclideanDistanceGradTensorOp<float, Eigen::DefaultDevice> operation;

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
	BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-6); //-nan
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -0.999999, 1e-4);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -0.999999, 1e-4);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -0.999999, 1e-4);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
  L2NormOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormOp) 
{
  L2NormTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
  L2NormTensorOp<float, Eigen::DefaultDevice>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormOp) 
{
  L2NormTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormTensorOp<float, Eigen::DefaultDevice>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormOp) 
{
  L2NormTensorOp<float, Eigen::DefaultDevice> operation;

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

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), -0.5, 1e-6); //TODO
	BOOST_CHECK_CLOSE(error(1, 0), -3.5, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  L2NormGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormGradOp) 
{
  L2NormGradTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
  L2NormGradTensorOp<float, Eigen::DefaultDevice>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormGradOp) 
{
  L2NormGradTensorOp<float, Eigen::DefaultDevice>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(operationfunctionL2NormGradOp) 
{
  L2NormGradTensorOp<float, Eigen::DefaultDevice> operation;

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
	BOOST_CHECK_CLOSE(error(1, 0, 0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -2.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
  CrossEntropyOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyOp) 
{
  BCETensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
  BCETensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyOp) 
{
  BCETensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new BCETensorOp<float, Eigen::DefaultDevice>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyOp) 
{
  BCETensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{1, 0}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{.1, .9}, {0, 0}},
		{{.9, .1}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 4.60514975, 1e-6); 
	BOOST_CHECK_CLOSE(error(1, 0), 0.21071884, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  CrossEntropyGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyGradOp) 
{
  BCEGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
  BCEGradTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyGradOp) 
{
  BCEGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new BCEGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyGradOp) 
{
  BCEGradTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{1, 0}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{.1, .9}, {0, 0}},
		{{.9, .1}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(error(0, 0, 0), -9.99988937, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -1.11109877, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), 9.99988651, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), 1.11109889, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
  NegativeLogLikelihoodOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodTensorOp<float, Eigen::DefaultDevice>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodTensorOp<float, Eigen::DefaultDevice>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{1, 0}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{.1, .9}, {0, 0}},
		{{.9, .1}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 1.15129256, 1e-6); //TODO
	BOOST_CHECK_CLOSE(error(1, 0), 0.0526802726, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  NegativeLogLikelihoodGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodGradTensorOp<float, Eigen::DefaultDevice>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradTensorOp<float, Eigen::DefaultDevice>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrNegativeLogLikelihood;
}

BOOST_AUTO_TEST_CASE(operationfunctionNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodGradTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{1, 0}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{.1, .9}, {0, 0}},
		{{.9, .1}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(error(0, 0, 0), -4.99994993, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -0.555554926, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
  MSEOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEOp) 
{
  MSETensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  MSETensorOp<float, Eigen::DefaultDevice>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEOp) 
{
  MSETensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
	ptrMSE = new MSETensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEOp) 
{
  MSETensorOp<float, Eigen::DefaultDevice> operation;

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
  MSEGradTensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
  MSEGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEGradOp) 
{
  MSEGradTensorOp<float, Eigen::DefaultDevice>* ptrMSE = nullptr;
	ptrMSE = new MSEGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSE;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSEGradOp) 
{
  MSEGradTensorOp<float, Eigen::DefaultDevice> operation;

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

/**
	KLDivergenceMuOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuOp)
{
	KLDivergenceMuTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
	KLDivergenceMuTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceMu = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuOp)
{
	KLDivergenceMuTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
	ptrKLDivergenceMu = new KLDivergenceMuTensorOp<float, Eigen::DefaultDevice>();
	delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuOp)
{
	KLDivergenceMuTensorOp<float, Eigen::DefaultDevice> operation;

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

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0), 3, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
	KLDivergenceMuGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuGradOp)
{
	KLDivergenceMuGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
	KLDivergenceMuGradTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceMu = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuGradOp)
{
	KLDivergenceMuGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceMu = nullptr;
	ptrKLDivergenceMu = new KLDivergenceMuGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceMuGradOp)
{
	KLDivergenceMuGradTensorOp<float, Eigen::DefaultDevice> operation;

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
	BOOST_CHECK_CLOSE(error(0, 0, 0), -2.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -4.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -2.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -4.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
	KLDivergenceLogVarOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarOp)
{
	KLDivergenceLogVarTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
	KLDivergenceLogVarTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceLogVar = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarOp)
{
	KLDivergenceLogVarTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
	ptrKLDivergenceLogVar = new KLDivergenceLogVarTensorOp<float, Eigen::DefaultDevice>();
	delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarOp2)
{
	KLDivergenceLogVarTensorOp<float, Eigen::DefaultDevice> operation;

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

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 1.29744244, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0), 2.43656349, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
	KLDivergenceLogVarGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
	KLDivergenceLogVarGradTensorOp<float, Eigen::DefaultDevice>* nullPointerKLDivergenceLogVar = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarGradTensorOp<float, Eigen::DefaultDevice>* ptrKLDivergenceLogVar = nullptr;
	ptrKLDivergenceLogVar = new KLDivergenceLogVarGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(operationfunctionKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarGradTensorOp<float, Eigen::DefaultDevice> operation;

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
	BOOST_CHECK_CLOSE(error(0, 0, 0), -1.14872122, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -2.21828175, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -1.14872122, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -2.21828175, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
BCEWithLogitsOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsOp)
{
	BCEWithLogitsTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
	BCEWithLogitsTensorOp<float, Eigen::DefaultDevice>* nullPointerBCEWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsOp)
{
	BCEWithLogitsTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
	ptrBCEWithLogits = new BCEWithLogitsTensorOp<float, Eigen::DefaultDevice>();
	delete ptrBCEWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsOp)
{
	BCEWithLogitsTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{0, 1}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{1, 2}, {0, 0}},
		{{1, 2}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 2.44018984, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0), 3.44018984, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
BCEWithLogitsGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsGradOp)
{
	BCEWithLogitsGradTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
	BCEWithLogitsGradTensorOp<float, Eigen::DefaultDevice>* nullPointerBCEWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsGradOp)
{
	BCEWithLogitsGradTensorOp<float, Eigen::DefaultDevice>* ptrBCEWithLogits = nullptr;
	ptrBCEWithLogits = new BCEWithLogitsGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrBCEWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionBCEWithLogitsGradOp)
{
	BCEWithLogitsGradTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{0, 1}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{1, 2}, {0, 0}},
		{{1, 2}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(error(0, 0, 0), 0.268941432, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -0.731058598, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -0.880797088, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -0.880797088, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
	CrossEntropyWithLogitsOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCrossEntropyWithLogitsOp)
{
	CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
	CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropyWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrCrossEntropyWithLogits, nullPointerCrossEntropyWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyWithLogitsOp)
{
	CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
	ptrCrossEntropyWithLogits = new CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice>();
	delete ptrCrossEntropyWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyWithLogitsOp1)
{
	CrossEntropyWithLogitsTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{0, 1}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{1, 2}, {0, 0}},
		{{1, 2}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
	BOOST_CHECK_CLOSE(error(0, 0), 0.656630814, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
	CrossEntropyWithLogitsGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCrossEntropyWithLogitsGradOp)
{
	CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
	CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice>* nullPointerCrossEntropyWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrCrossEntropyWithLogits, nullPointerCrossEntropyWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyWithLogitsGradOp)
{
	CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice>* ptrCrossEntropyWithLogits = nullptr;
	ptrCrossEntropyWithLogits = new CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrCrossEntropyWithLogits;
}

BOOST_AUTO_TEST_CASE(operationfunctionCrossEntropyWithLogitsGradOp1)
{
	CrossEntropyWithLogitsGradTensorOp<float, Eigen::DefaultDevice> operation;

	const int memory_size = 2;
	const int batch_size = 2;
	const int layer_size = 2;
	const int time_step = 0;
	Eigen::Tensor<float, 3> y_true(batch_size, memory_size, layer_size);
	y_true.setValues({
		{{1, 0}, {0, 0}},
		{{0, 1}, {0, 0}}
		});
	Eigen::Tensor<float, 3> y_pred(batch_size, memory_size, layer_size);
	y_pred.setValues({
		{{1, 2}, {0, 0}},
		{{1, 2}, {0, 0}}
		});

	float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Eigen::DefaultDevice device;

	operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
	Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
	BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 0), -0.5, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 0, 1), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 0, 1), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
  MSERangeUBOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBOp)
{
  MSERangeUBTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  MSERangeUBTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBOp)
{
  MSERangeUBTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeUBOp)
{
  MSERangeUBTensorOp<float, Eigen::DefaultDevice> operation;

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
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 2.25, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 1, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  MSERangeUBGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBGradOp)
{
  MSERangeUBGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  MSERangeUBGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBGradOp)
{
  MSERangeUBGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeUBGradOp)
{
  MSERangeUBGradTensorOp<float, Eigen::DefaultDevice> operation;

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
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 0, 1), -1.5, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0, 1), -1.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

/**
  MSERangeLBOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBOp)
{
  MSERangeLBTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  MSERangeLBTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBOp)
{
  MSERangeLBTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeLBOp)
{
  MSERangeLBTensorOp<float, Eigen::DefaultDevice> operation;

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
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> error(error_ptr, batch_size, memory_size);
  BOOST_CHECK_CLOSE(error(0, 0), 0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0), 0.25, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1), 0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1), 0, 1e-6);
}

/**
  MSERangeLBGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBGradOp)
{
  MSERangeLBGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  MSERangeLBGradTensorOp<float, Eigen::DefaultDevice>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBGradOp)
{
  MSERangeLBGradTensorOp<float, Eigen::DefaultDevice>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(operationfunctionMSERangeLBGradOp)
{
  MSERangeLBGradTensorOp<float, Eigen::DefaultDevice> operation;

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
    {{1, 3}, {0, 0}},
    {{0, 2}, {0, 0}}
    });

  float error_ptr[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  Eigen::DefaultDevice device;

  operation(y_pred.data(), y_true.data(), error_ptr, batch_size, memory_size, layer_size, time_step, device);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> error(error_ptr, batch_size, memory_size, layer_size);
  BOOST_CHECK_CLOSE(error(0, 0, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0, 0), 0.5, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1, 0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 0, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(0, 1, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 0, 1), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(error(1, 1, 1), 0.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()