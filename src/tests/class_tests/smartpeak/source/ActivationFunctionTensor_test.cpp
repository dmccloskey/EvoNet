/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ActivationFunctionTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ActivationFunctionTensor.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(activationFunction)

/**
  ReLUTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluTensorOp) 
{
  ReLUTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ReLUTensorOp<float, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorReluTensorOp) 
{
  ReLUTensorOp<float, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new ReLUTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(operationfunctionReluTensorOp) 
{
  ReLUTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameReLUTensorOp)
{
	ReLUTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUTensorOp");
}

/**
  ReLUGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluGradTensorOp) 
{
  ReLUGradTensorOp<float, Eigen::DefaultDevice>* ptrReLUGrad = nullptr;
  ReLUGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReLUGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReLUGrad, nullPointerReLUGrad);
}

BOOST_AUTO_TEST_CASE(destructorReluGradTensorOp) 
{
  ReLUGradTensorOp<float, Eigen::DefaultDevice>* ptrReLUGrad = nullptr;
	ptrReLUGrad = new ReLUGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReLUGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionReluGradTensorOp) 
{
  ReLUGradTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameReLUGradTensorOp)
{
	ReLUGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUGradTensorOp");
}

/**
  ELUTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluTensorOp) 
{
  ELUTensorOp<float, Eigen::DefaultDevice>* ptrELU = nullptr;
  ELUTensorOp<float, Eigen::DefaultDevice>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluTensorOp) 
{
  ELUTensorOp<float, Eigen::DefaultDevice>* ptrELU = nullptr;
	ptrELU = new ELUTensorOp<float, Eigen::DefaultDevice>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluTensorOp) 
{
  ELUTensorOp<float, Eigen::DefaultDevice> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionEluTensorOp) 
{
  ELUTensorOp<float, Eigen::DefaultDevice> operation(1.0);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-0.63212055882855767,-0.63212055882855767}, {0,0}},
		{{-0.99995460007023751,-0.99995460007023751}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameELUTensorOp)
{
	ELUTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ELUTensorOp");
}

/**
  ELUGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluGradTensorOp) 
{
  ELUGradTensorOp<float, Eigen::DefaultDevice>* ptrELU = nullptr;
  ELUGradTensorOp<float, Eigen::DefaultDevice>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluGradTensorOp) 
{
  ELUGradTensorOp<float, Eigen::DefaultDevice>* ptrELU = nullptr;
	ptrELU = new ELUGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluGradTensorOp) 
{
  ELUGradTensorOp<float, Eigen::DefaultDevice> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionEluGradTensorOp) 
{
  ELUGradTensorOp<float, Eigen::DefaultDevice> operation(1.0);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0.36787944117144233,0.36787944117144233}, {0,0}},
		{{4.54187393e-05,4.54187393e-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameELUGradTensorOp)
{
	ELUGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ELUGradTensorOp");
}

/**
  SigmoidTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSigmoidTensorOp) 
{
  SigmoidTensorOp<float, Eigen::DefaultDevice>* ptrSigmoid = nullptr;
  SigmoidTensorOp<float, Eigen::DefaultDevice>* nullPointerSigmoid = nullptr;
  BOOST_CHECK_EQUAL(ptrSigmoid, nullPointerSigmoid);
}

BOOST_AUTO_TEST_CASE(destructorSigmoidTensorOp) 
{
  SigmoidTensorOp<float, Eigen::DefaultDevice>* ptrSigmoid = nullptr;
	ptrSigmoid = new SigmoidTensorOp<float, Eigen::DefaultDevice>();
  delete ptrSigmoid;
}

BOOST_AUTO_TEST_CASE(operationfunctionSigmoidTensorOp) 
{
  SigmoidTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.5,0.5}, {0,0}},
		{{0.7310585786300049,0.7310585786300049}, {0,0}},
		{{0.99995460213129761,0.99995460213129761}, {0,0}},
		{{0.2689414213699951,0.2689414213699951}, {0,0}},
		{{4.5397868702434395e-05,4.5397868702434395e-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSigmoidTensorOp)
{
	SigmoidTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SigmoidTensorOp");
}

/**
  SigmoidGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSigmoidGradTensorOp) 
{
  SigmoidGradTensorOp<float, Eigen::DefaultDevice>* ptrSigmoidGrad = nullptr;
  SigmoidGradTensorOp<float, Eigen::DefaultDevice>* nullPointerSigmoidGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrSigmoidGrad, nullPointerSigmoidGrad);
}

BOOST_AUTO_TEST_CASE(destructorSigmoidGradTensorOp) 
{
  SigmoidGradTensorOp<float, Eigen::DefaultDevice>* ptrSigmoidGrad = nullptr;
	ptrSigmoidGrad = new SigmoidGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrSigmoidGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionSigmoidGradTensorOp) 
{
  SigmoidGradTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.25,0.25}, {0,0}},
		{{0.19661193324148185,0.19661193324148185}, {0,0}},
		{{4.54166766e-05,4.54166766e-05}, {0,0}},
		{{0.19661193324148185,0.19661193324148185}, {0,0}},
		{{4.53958091e-05,4.53958091e-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSigmoidGradTensorOp)
{
	SigmoidGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SigmoidGradTensorOp");
}

/**
  TanHTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorTanHTensorOp) 
{
  TanHTensorOp<float, Eigen::DefaultDevice>* ptrTanH = nullptr;
  TanHTensorOp<float, Eigen::DefaultDevice>* nullPointerTanH = nullptr;
  BOOST_CHECK_EQUAL(ptrTanH, nullPointerTanH);
}

BOOST_AUTO_TEST_CASE(destructorTanHTensorOp) 
{
  TanHTensorOp<float, Eigen::DefaultDevice>* ptrTanH = nullptr;
	ptrTanH = new TanHTensorOp<float, Eigen::DefaultDevice>();
  delete ptrTanH;
}

BOOST_AUTO_TEST_CASE(operationfunctionTanHTensorOp) 
{
  TanHTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.0,0.0}, {0,0}},
		{{0.76159415595576485,0.76159415595576485}, {0,0}},
		{{0.99999999587769262,0.99999999587769262}, {0,0}},
		{{-0.76159415595576485,-0.76159415595576485}, {0,0}},
		{{-0.99999999587769262,-0.99999999587769262}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameTanHTensorOp)
{
	TanHTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "TanHTensorOp");
}

/**
  TanHGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorTanHGradTensorOp) 
{
  TanHGradTensorOp<float, Eigen::DefaultDevice>* ptrTanHGrad = nullptr;
  TanHGradTensorOp<float, Eigen::DefaultDevice>* nullPointerTanHGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrTanHGrad, nullPointerTanHGrad);
}

BOOST_AUTO_TEST_CASE(destructorTanHGradTensorOp) 
{
  TanHGradTensorOp<float, Eigen::DefaultDevice>* ptrTanHGrad = nullptr;
	ptrTanHGrad = new TanHGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrTanHGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionTanHGradTensorOp) 
{
  TanHGradTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{0,0}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameTanHGradTensorOp)
{
	TanHGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "TanHGradTensorOp");
}

/**
  ReTanHTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReTanHTensorOp) 
{
  ReTanHTensorOp<float, Eigen::DefaultDevice>* ptrReTanH = nullptr;
  ReTanHTensorOp<float, Eigen::DefaultDevice>* nullPointerReTanH = nullptr;
  BOOST_CHECK_EQUAL(ptrReTanH, nullPointerReTanH);
}

BOOST_AUTO_TEST_CASE(destructorReTanHTensorOp) 
{
  ReTanHTensorOp<float, Eigen::DefaultDevice>* ptrReTanH = nullptr;
	ptrReTanH = new ReTanHTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReTanH;
}

// [TODO: need to re-implement]
//BOOST_AUTO_TEST_CASE(operationfunctionReTanHTensorOp) 
//{
//  ReTanHTensorOp<float, Eigen::DefaultDevice> operation;
//	const int batch_size = 5;
//	const int memory_size = 2;
//	const int layer_size = 2;
//	Eigen::DefaultDevice device;
//	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
//	input.setValues({
//		{{0,0}, {0,0}},
//		{{1,1}, {0,0}},
//		{{10,10}, {0,0}},
//		{{-1,-1}, {0,0}},
//		{{-10,-10}, {0,0}} });
//	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
//	output.setZero();
//	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
//	test.setValues({
//		{{0,0}, {0,0}},
//		{{0.76159415595576485,0.76159415595576485}, {0,0}},
//		{{0.99999999587769262,0.99999999587769262}, {0,0}},
//		{{0,0}, {0,0}},
//		{{0,0}, {0,0}} });
//
//	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
//
//	// Test
//	for (int i = 0; i < batch_size; ++i) {
//		for (int j = 0; j < memory_size; ++j) {
//			for (int k = 0; k < layer_size; ++k) {
//				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
//			}
//		}
//	}
//}

BOOST_AUTO_TEST_CASE(getNameReTanHTensorOp)
{
	ReTanHTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReTanHTensorOp");
}

/**
  ReTanHGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReTanHGradTensorOp) 
{
  ReTanHGradTensorOp<float, Eigen::DefaultDevice>* ptrReTanHGrad = nullptr;
  ReTanHGradTensorOp<float, Eigen::DefaultDevice>* nullPointerReTanHGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReTanHGrad, nullPointerReTanHGrad);
}

BOOST_AUTO_TEST_CASE(destructorReTanHGradTensorOp) 
{
  ReTanHGradTensorOp<float, Eigen::DefaultDevice>* ptrReTanHGrad = nullptr;
	ptrReTanHGrad = new ReTanHGradTensorOp<float, Eigen::DefaultDevice>();
  delete ptrReTanHGrad;
}

// TODO: need to re-implement
//BOOST_AUTO_TEST_CASE(operationfunctionReTanHGradTensorOp) 
//{
//  ReTanHGradTensorOp<float, Eigen::DefaultDevice> operation;
//	const int batch_size = 5;
//	const int memory_size = 2;
//	const int layer_size = 2;
//	Eigen::DefaultDevice device;
//	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
//	input.setValues({
//		{{0,0}, {0,0}},
//		{{1,1}, {0,0}},
//		{{10,10}, {0,0}},
//		{{-1,-1}, {0,0}},
//		{{-10,-10}, {0,0}} });
//	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
//	output.setZero();
//	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
//	test.setValues({
//		{{0,0}, {0,0}},
//		{{0.41997434161402614,0.41997434161402614}, {0,0}},
//		{{8.2446147686709992e-09,8.2446147686709992e-09}, {0,0}},
//		{{0,0}, {0,0}},
//		{{0,0}, {0,0}} });
//
//	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
//
//	// Test
//	for (int i = 0; i < batch_size; ++i) {
//		for (int j = 0; j < memory_size; ++j) {
//			for (int k = 0; k < layer_size; ++k) {
//				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
//			}
//		}
//	}
//}

BOOST_AUTO_TEST_CASE(getNameReTanHGradTensorOp)
{
	ReTanHGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReTanHGradTensorOp");
}

/**
LinearTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLinearTensorOp)
{
	LinearTensorOp<float, Eigen::DefaultDevice>* ptrLinear = nullptr;
	LinearTensorOp<float, Eigen::DefaultDevice>* nullPointerLinear = nullptr;
	BOOST_CHECK_EQUAL(ptrLinear, nullPointerLinear);
}

BOOST_AUTO_TEST_CASE(destructorLinearTensorOp)
{
	LinearTensorOp<float, Eigen::DefaultDevice>* ptrLinear = nullptr;
	ptrLinear = new LinearTensorOp<float, Eigen::DefaultDevice>();
	delete ptrLinear;
}

BOOST_AUTO_TEST_CASE(operationfunctionLinearTensorOp)
{
	LinearTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameLinearTensorOp)
{
	LinearTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LinearTensorOp");
}

/**
LinearGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLinearGradTensorOp)
{
	LinearGradTensorOp<float, Eigen::DefaultDevice>* ptrLinearGrad = nullptr;
	LinearGradTensorOp<float, Eigen::DefaultDevice>* nullPointerLinearGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrLinearGrad, nullPointerLinearGrad);
}

BOOST_AUTO_TEST_CASE(destructorLinearGradTensorOp)
{
	LinearGradTensorOp<float, Eigen::DefaultDevice>* ptrLinearGrad = nullptr;
	ptrLinearGrad = new LinearGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrLinearGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionLinearGradTensorOp)
{
	LinearGradTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameLinearGradTensorOp)
{
	LinearGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LinearGradTensorOp");
}

/**
InverseTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorInverseTensorOp)
{
	InverseTensorOp<float, Eigen::DefaultDevice>* ptrInverse = nullptr;
	InverseTensorOp<float, Eigen::DefaultDevice>* nullPointerInverse = nullptr;
	BOOST_CHECK_EQUAL(ptrInverse, nullPointerInverse);
}

BOOST_AUTO_TEST_CASE(destructorInverseTensorOp)
{
	InverseTensorOp<float, Eigen::DefaultDevice>* ptrInverse = nullptr;
	ptrInverse = new InverseTensorOp<float, Eigen::DefaultDevice>();
	delete ptrInverse;
}

BOOST_AUTO_TEST_CASE(operationfunctionInverseTensorOp)
{
	InverseTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{0.1,0.1}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.1,-0.1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i,j,k), test(i,j,k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameInverseTensorOp)
{
	InverseTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "InverseTensorOp");
}

/**
InverseGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorInverseGradTensorOp)
{
	InverseGradTensorOp<float, Eigen::DefaultDevice>* ptrInverseGrad = nullptr;
	InverseGradTensorOp<float, Eigen::DefaultDevice>* nullPointerInverseGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrInverseGrad, nullPointerInverseGrad);
}

BOOST_AUTO_TEST_CASE(destructorInverseGradTensorOp)
{
	InverseGradTensorOp<float, Eigen::DefaultDevice>* ptrInverseGrad = nullptr;
	ptrInverseGrad = new InverseGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrInverseGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionInverseGradTensorOp)
{
	InverseGradTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.01,-0.01}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.01,-0.01}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameInverseGradTensorOp)
{
	InverseGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "InverseGradTensorOp");
}

/**
ExponentialTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorExponentialTensorOp)
{
	ExponentialTensorOp<float, Eigen::DefaultDevice>* ptrExponential = nullptr;
	ExponentialTensorOp<float, Eigen::DefaultDevice>* nullPointerExponential = nullptr;
	BOOST_CHECK_EQUAL(ptrExponential, nullPointerExponential);
}

BOOST_AUTO_TEST_CASE(destructorExponentialTensorOp)
{
	ExponentialTensorOp<float, Eigen::DefaultDevice>* ptrExponential = nullptr;
	ptrExponential = new ExponentialTensorOp<float, Eigen::DefaultDevice>();
	delete ptrExponential;
}

BOOST_AUTO_TEST_CASE(operationfunctionExponentialTensorOp)
{
	ExponentialTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameExponentialTensorOp)
{
	ExponentialTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialTensorOp");
}

/**
ExponentialGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorExponentialGradTensorOp)
{
	ExponentialGradTensorOp<float, Eigen::DefaultDevice>* ptrExponentialGrad = nullptr;
	ExponentialGradTensorOp<float, Eigen::DefaultDevice>* nullPointerExponentialGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrExponentialGrad, nullPointerExponentialGrad);
}

BOOST_AUTO_TEST_CASE(destructorExponentialGradTensorOp)
{
	ExponentialGradTensorOp<float, Eigen::DefaultDevice>* ptrExponentialGrad = nullptr;
	ptrExponentialGrad = new ExponentialGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrExponentialGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionExponentialGradTensorOp)
{
	ExponentialGradTensorOp<float, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{2.718281828,2.718281828}, {0,0}},
		{{22026.46579,22026.46579}, {0,0}},
		{{0.367879441,0.367879441}, {0,0}},
		{{4.53999E-05,4.53999E-05}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameExponentialGradTensorOp)
{
	ExponentialGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialGradTensorOp");
}

/**
	PowTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPowTensorOp)
{
	PowTensorOp<float, Eigen::DefaultDevice>* ptrPow = nullptr;
	PowTensorOp<float, Eigen::DefaultDevice>* nullPointerPow = nullptr;
	BOOST_CHECK_EQUAL(ptrPow, nullPointerPow);
}

BOOST_AUTO_TEST_CASE(destructorPowTensorOp)
{
	PowTensorOp<float, Eigen::DefaultDevice>* ptrPow = nullptr;
	ptrPow = new PowTensorOp<float, Eigen::DefaultDevice>(2);
	delete ptrPow;
}

BOOST_AUTO_TEST_CASE(operationfunctionPowTensorOp)
{
	PowTensorOp<float, Eigen::DefaultDevice> operation(0.5);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{3.1622776601683795,3.1622776601683795}, {0,0}},
		{{-1.0e9,-1.0e9}, {0,0}},
		{{-1.0e9,-1.0e9}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNamePowTensorOp)
{
	PowTensorOp<float, Eigen::DefaultDevice> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowTensorOp");
}

/**
	PowGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPowGradTensorOp)
{
	PowGradTensorOp<float, Eigen::DefaultDevice>* ptrPowGrad = nullptr;
	PowGradTensorOp<float, Eigen::DefaultDevice>* nullPointerPowGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrPowGrad, nullPointerPowGrad);
}

BOOST_AUTO_TEST_CASE(destructorPowGradTensorOp)
{
	PowGradTensorOp<float, Eigen::DefaultDevice>* ptrPowGrad = nullptr;
	ptrPowGrad = new PowGradTensorOp<float, Eigen::DefaultDevice>(0.5);
	delete ptrPowGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionPowGradTensorOp)
{
	PowGradTensorOp<float, Eigen::DefaultDevice> operation(0.5);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1.0e9,1.0e9}, {0,0}},
		{{0.5,0.5}, {0,0}},
		{{0.15811388300841897,0.15811388300841897}, {0,0}},
		{{-1.0e9,-1.0e9}, {0,0}},
		{{-1.0e9,-1.0e9}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNamePowGradTensorOp)
{
	PowGradTensorOp<float, Eigen::DefaultDevice> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowGradTensorOp");
}

/**
	LeakyReLUTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<float, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	LeakyReLUTensorOp<float, Eigen::DefaultDevice>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

BOOST_AUTO_TEST_CASE(destructorLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<float, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUTensorOp<float, Eigen::DefaultDevice>();
	delete ptrLeakyReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<float, Eigen::DefaultDevice> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<float, Eigen::DefaultDevice> operation(0.1);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-0.1,-0.1}, {0,0}},
		{{-1,-1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUTensorOp");
}

/**
	LeakyReLUGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<float, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	LeakyReLUGradTensorOp<float, Eigen::DefaultDevice>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

BOOST_AUTO_TEST_CASE(destructorLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<float, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUGradTensorOp<float, Eigen::DefaultDevice>();
	delete ptrLeakyReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<float, Eigen::DefaultDevice> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<float, Eigen::DefaultDevice> operation(0.1);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<float, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<float, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<float, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0.1,0.1}, {0,0}},
		{{0.1,0.1}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<float, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUGradTensorOp");
}

BOOST_AUTO_TEST_SUITE_END()