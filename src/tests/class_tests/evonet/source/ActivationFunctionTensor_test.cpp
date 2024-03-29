/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ActivationFunctionTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/ml/ActivationFunctionTensor.h>

#include <iostream>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(activationFunction)

/**
  ReLUTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluTensorOp) 
{
  ReLUTensorOp<double, Eigen::DefaultDevice>* ptrReLU = nullptr;
  ReLUTensorOp<double, Eigen::DefaultDevice>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorReluTensorOp) 
{
  ReLUTensorOp<double, Eigen::DefaultDevice>* ptrReLU = nullptr;
	ptrReLU = new ReLUTensorOp<double, Eigen::DefaultDevice>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersReluTensorOp)
{
  // Test defaults
  ReLUTensorOp<double, Eigen::DefaultDevice> operation_defaults;
  BOOST_CHECK_CLOSE(operation_defaults.getEps(), 1e-24, 1e-6);
  BOOST_CHECK_CLOSE(operation_defaults.getMin(), -1e9, 1e-6);
  BOOST_CHECK_CLOSE(operation_defaults.getMax(), 1e9, 1e-6);

  // Test setters
  operation_defaults.setEps(10);
  operation_defaults.setMin(20);
  operation_defaults.setMax(30);
  BOOST_CHECK_CLOSE(operation_defaults.getEps(), 10, 1e-6);
  BOOST_CHECK_CLOSE(operation_defaults.getMin(), 20, 1e-6);
  BOOST_CHECK_CLOSE(operation_defaults.getMax(), 30, 1e-6);

  // Test constructor
  ReLUTensorOp<double, Eigen::DefaultDevice> operation(10, 20, 30);
  BOOST_CHECK_CLOSE(operation.getEps(), 10, 1e-6);
  BOOST_CHECK_CLOSE(operation.getMin(), 20, 1e-6);
  BOOST_CHECK_CLOSE(operation.getMax(), 30, 1e-6);
}

BOOST_AUTO_TEST_CASE(operationfunctionReluTensorOp) 
{
  ReLUTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	ReLUTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUTensorOp");
}

/**
  ReLUGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluGradTensorOp) 
{
  ReLUGradTensorOp<double, Eigen::DefaultDevice>* ptrReLUGrad = nullptr;
  ReLUGradTensorOp<double, Eigen::DefaultDevice>* nullPointerReLUGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReLUGrad, nullPointerReLUGrad);
}

BOOST_AUTO_TEST_CASE(destructorReluGradTensorOp) 
{
  ReLUGradTensorOp<double, Eigen::DefaultDevice>* ptrReLUGrad = nullptr;
	ptrReLUGrad = new ReLUGradTensorOp<double, Eigen::DefaultDevice>();
  delete ptrReLUGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionReluGradTensorOp) 
{
  ReLUGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
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
	ReLUGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUGradTensorOp");
}

/**
  ELUTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluTensorOp) 
{
  ELUTensorOp<double, Eigen::DefaultDevice>* ptrELU = nullptr;
  ELUTensorOp<double, Eigen::DefaultDevice>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluTensorOp) 
{
  ELUTensorOp<double, Eigen::DefaultDevice>* ptrELU = nullptr;
	ptrELU = new ELUTensorOp<double, Eigen::DefaultDevice>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluTensorOp) 
{
  ELUTensorOp<double, Eigen::DefaultDevice> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionEluTensorOp) 
{
  ELUTensorOp<double, Eigen::DefaultDevice> operation(1.0);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	ELUTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ELUTensorOp");
}

/**
  ELUGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluGradTensorOp) 
{
  ELUGradTensorOp<double, Eigen::DefaultDevice>* ptrELU = nullptr;
  ELUGradTensorOp<double, Eigen::DefaultDevice>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluGradTensorOp) 
{
  ELUGradTensorOp<double, Eigen::DefaultDevice>* ptrELU = nullptr;
	ptrELU = new ELUGradTensorOp<double, Eigen::DefaultDevice>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluGradTensorOp) 
{
  ELUGradTensorOp<double, Eigen::DefaultDevice> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionEluGradTensorOp) 
{
  ELUGradTensorOp<double, Eigen::DefaultDevice> operation(1.0);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{1,1}, {0,0}},
		{{0.36787944117144233,0.36787944117144233}, {0,0}},
		{{4.5399929762490743e-05,4.5399929762490743e-05}, {0,0}} });

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
	ELUGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ELUGradTensorOp");
}

/**
  SigmoidTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSigmoidTensorOp) 
{
  SigmoidTensorOp<double, Eigen::DefaultDevice>* ptrSigmoid = nullptr;
  SigmoidTensorOp<double, Eigen::DefaultDevice>* nullPointerSigmoid = nullptr;
  BOOST_CHECK_EQUAL(ptrSigmoid, nullPointerSigmoid);
}

BOOST_AUTO_TEST_CASE(destructorSigmoidTensorOp) 
{
  SigmoidTensorOp<double, Eigen::DefaultDevice>* ptrSigmoid = nullptr;
	ptrSigmoid = new SigmoidTensorOp<double, Eigen::DefaultDevice>();
  delete ptrSigmoid;
}

BOOST_AUTO_TEST_CASE(operationfunctionSigmoidTensorOp) 
{
  SigmoidTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	SigmoidTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SigmoidTensorOp");
}

/**
  SigmoidGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSigmoidGradTensorOp) 
{
  SigmoidGradTensorOp<double, Eigen::DefaultDevice>* ptrSigmoidGrad = nullptr;
  SigmoidGradTensorOp<double, Eigen::DefaultDevice>* nullPointerSigmoidGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrSigmoidGrad, nullPointerSigmoidGrad);
}

BOOST_AUTO_TEST_CASE(destructorSigmoidGradTensorOp) 
{
  SigmoidGradTensorOp<double, Eigen::DefaultDevice>* ptrSigmoidGrad = nullptr;
	ptrSigmoidGrad = new SigmoidGradTensorOp<double, Eigen::DefaultDevice>();
  delete ptrSigmoidGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionSigmoidGradTensorOp) 
{
  SigmoidGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0.25,0.25}, {0,0}},
		{{0.19661193324148185,0.19661193324148185}, {0,0}},
		{{4.5395807735907655e-05,4.5395807735907655e-05}, {0,0}},
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
	SigmoidGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SigmoidGradTensorOp");
}

/**
  TanHTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorTanHTensorOp) 
{
  TanHTensorOp<double, Eigen::DefaultDevice>* ptrTanH = nullptr;
  TanHTensorOp<double, Eigen::DefaultDevice>* nullPointerTanH = nullptr;
  BOOST_CHECK_EQUAL(ptrTanH, nullPointerTanH);
}

BOOST_AUTO_TEST_CASE(destructorTanHTensorOp) 
{
  TanHTensorOp<double, Eigen::DefaultDevice>* ptrTanH = nullptr;
	ptrTanH = new TanHTensorOp<double, Eigen::DefaultDevice>();
  delete ptrTanH;
}

BOOST_AUTO_TEST_CASE(operationfunctionTanHTensorOp) 
{
  TanHTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	TanHTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "TanHTensorOp");
}

/**
  TanHGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorTanHGradTensorOp) 
{
  TanHGradTensorOp<double, Eigen::DefaultDevice>* ptrTanHGrad = nullptr;
  TanHGradTensorOp<double, Eigen::DefaultDevice>* nullPointerTanHGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrTanHGrad, nullPointerTanHGrad);
}

BOOST_AUTO_TEST_CASE(destructorTanHGradTensorOp) 
{
  TanHGradTensorOp<double, Eigen::DefaultDevice>* ptrTanHGrad = nullptr;
	ptrTanHGrad = new TanHGradTensorOp<double, Eigen::DefaultDevice>();
  delete ptrTanHGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionTanHGradTensorOp) 
{
  TanHGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1,1}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{8.2446145466263943e-09,8.2446145466263943e-09}, {0,0}},
		{{0.41997434161402614,0.41997434161402614}, {0,0}},
		{{8.2446145466263943e-09,8.2446145466263943e-09}, {0,0}} });

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
	TanHGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "TanHGradTensorOp");
}

/**
  ReTanHTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReTanHTensorOp) 
{
  ReTanHTensorOp<double, Eigen::DefaultDevice>* ptrReTanH = nullptr;
  ReTanHTensorOp<double, Eigen::DefaultDevice>* nullPointerReTanH = nullptr;
  BOOST_CHECK_EQUAL(ptrReTanH, nullPointerReTanH);
}

BOOST_AUTO_TEST_CASE(destructorReTanHTensorOp) 
{
  ReTanHTensorOp<double, Eigen::DefaultDevice>* ptrReTanH = nullptr;
	ptrReTanH = new ReTanHTensorOp<double, Eigen::DefaultDevice>();
  delete ptrReTanH;
}

// [TODO: need to re-implement]
//BOOST_AUTO_TEST_CASE(operationfunctionReTanHTensorOp) 
//{
//  ReTanHTensorOp<double, Eigen::DefaultDevice> operation;
//	const int batch_size = 5;
//	const int memory_size = 2;
//	const int layer_size = 2;
//	Eigen::DefaultDevice device;
//	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
//	input.setValues({
//		{{0,0}, {0,0}},
//		{{1,1}, {0,0}},
//		{{10,10}, {0,0}},
//		{{-1,-1}, {0,0}},
//		{{-10,-10}, {0,0}} });
//	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
//	output.setZero();
//	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	ReTanHTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReTanHTensorOp");
}

/**
  ReTanHGradTensorOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReTanHGradTensorOp) 
{
  ReTanHGradTensorOp<double, Eigen::DefaultDevice>* ptrReTanHGrad = nullptr;
  ReTanHGradTensorOp<double, Eigen::DefaultDevice>* nullPointerReTanHGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReTanHGrad, nullPointerReTanHGrad);
}

BOOST_AUTO_TEST_CASE(destructorReTanHGradTensorOp) 
{
  ReTanHGradTensorOp<double, Eigen::DefaultDevice>* ptrReTanHGrad = nullptr;
	ptrReTanHGrad = new ReTanHGradTensorOp<double, Eigen::DefaultDevice>();
  delete ptrReTanHGrad;
}

// TODO: need to re-implement
//BOOST_AUTO_TEST_CASE(operationfunctionReTanHGradTensorOp) 
//{
//  ReTanHGradTensorOp<double, Eigen::DefaultDevice> operation;
//	const int batch_size = 5;
//	const int memory_size = 2;
//	const int layer_size = 2;
//	Eigen::DefaultDevice device;
//	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
//	input.setValues({
//		{{0,0}, {0,0}},
//		{{1,1}, {0,0}},
//		{{10,10}, {0,0}},
//		{{-1,-1}, {0,0}},
//		{{-10,-10}, {0,0}} });
//	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
//	output.setZero();
//	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	ReTanHGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReTanHGradTensorOp");
}

/**
LinearTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLinearTensorOp)
{
	LinearTensorOp<double, Eigen::DefaultDevice>* ptrLinear = nullptr;
	LinearTensorOp<double, Eigen::DefaultDevice>* nullPointerLinear = nullptr;
	BOOST_CHECK_EQUAL(ptrLinear, nullPointerLinear);
}

BOOST_AUTO_TEST_CASE(destructorLinearTensorOp)
{
	LinearTensorOp<double, Eigen::DefaultDevice>* ptrLinear = nullptr;
	ptrLinear = new LinearTensorOp<double, Eigen::DefaultDevice>();
	delete ptrLinear;
}

BOOST_AUTO_TEST_CASE(operationfunctionLinearTensorOp)
{
	LinearTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	LinearTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LinearTensorOp");
}

/**
LinearGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLinearGradTensorOp)
{
	LinearGradTensorOp<double, Eigen::DefaultDevice>* ptrLinearGrad = nullptr;
	LinearGradTensorOp<double, Eigen::DefaultDevice>* nullPointerLinearGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrLinearGrad, nullPointerLinearGrad);
}

BOOST_AUTO_TEST_CASE(destructorLinearGradTensorOp)
{
	LinearGradTensorOp<double, Eigen::DefaultDevice>* ptrLinearGrad = nullptr;
	ptrLinearGrad = new LinearGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrLinearGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionLinearGradTensorOp)
{
	LinearGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	LinearGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LinearGradTensorOp");
}

/**
InverseTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorInverseTensorOp)
{
	InverseTensorOp<double, Eigen::DefaultDevice>* ptrInverse = nullptr;
	InverseTensorOp<double, Eigen::DefaultDevice>* nullPointerInverse = nullptr;
	BOOST_CHECK_EQUAL(ptrInverse, nullPointerInverse);
}

BOOST_AUTO_TEST_CASE(destructorInverseTensorOp)
{
	InverseTensorOp<double, Eigen::DefaultDevice>* ptrInverse = nullptr;
	ptrInverse = new InverseTensorOp<double, Eigen::DefaultDevice>();
	delete ptrInverse;
}

BOOST_AUTO_TEST_CASE(operationfunctionInverseTensorOp)
{
	InverseTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	InverseTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "InverseTensorOp");
}

/**
InverseGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorInverseGradTensorOp)
{
	InverseGradTensorOp<double, Eigen::DefaultDevice>* ptrInverseGrad = nullptr;
	InverseGradTensorOp<double, Eigen::DefaultDevice>* nullPointerInverseGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrInverseGrad, nullPointerInverseGrad);
}

BOOST_AUTO_TEST_CASE(destructorInverseGradTensorOp)
{
	InverseGradTensorOp<double, Eigen::DefaultDevice>* ptrInverseGrad = nullptr;
	ptrInverseGrad = new InverseGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrInverseGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionInverseGradTensorOp)
{
	InverseGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	InverseGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "InverseGradTensorOp");
}

/**
ExponentialTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorExponentialTensorOp)
{
	ExponentialTensorOp<double, Eigen::DefaultDevice>* ptrExponential = nullptr;
	ExponentialTensorOp<double, Eigen::DefaultDevice>* nullPointerExponential = nullptr;
	BOOST_CHECK_EQUAL(ptrExponential, nullPointerExponential);
}

BOOST_AUTO_TEST_CASE(destructorExponentialTensorOp)
{
	ExponentialTensorOp<double, Eigen::DefaultDevice>* ptrExponential = nullptr;
	ptrExponential = new ExponentialTensorOp<double, Eigen::DefaultDevice>();
	delete ptrExponential;
}

BOOST_AUTO_TEST_CASE(operationfunctionExponentialTensorOp)
{
	ExponentialTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	ExponentialTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialTensorOp");
}

/**
ExponentialGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorExponentialGradTensorOp)
{
	ExponentialGradTensorOp<double, Eigen::DefaultDevice>* ptrExponentialGrad = nullptr;
	ExponentialGradTensorOp<double, Eigen::DefaultDevice>* nullPointerExponentialGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrExponentialGrad, nullPointerExponentialGrad);
}

BOOST_AUTO_TEST_CASE(destructorExponentialGradTensorOp)
{
	ExponentialGradTensorOp<double, Eigen::DefaultDevice>* ptrExponentialGrad = nullptr;
	ptrExponentialGrad = new ExponentialGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrExponentialGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionExponentialGradTensorOp)
{
	ExponentialGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	ExponentialGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialGradTensorOp");
}

/**
LogTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLogTensorOp)
{
	LogTensorOp<double, Eigen::DefaultDevice>* ptrLog = nullptr;
	LogTensorOp<double, Eigen::DefaultDevice>* nullPointerLog = nullptr;
	BOOST_CHECK_EQUAL(ptrLog, nullPointerLog);
}

BOOST_AUTO_TEST_CASE(destructorLogTensorOp)
{
	LogTensorOp<double, Eigen::DefaultDevice>* ptrLog = nullptr;
	ptrLog = new LogTensorOp<double, Eigen::DefaultDevice>();
	delete ptrLog;
}

BOOST_AUTO_TEST_CASE(operationfunctionLogTensorOp)
{
	LogTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} 
		});
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{-55.262042231857095,-55.262042231857095}, {0,0}},
		{{0,0}, {0,0}},
		{{2.3025850929940459,2.3025850929940459}, {0,0}},
		{{-55.262042231857095,-55.262042231857095}, {0,0}},
		{{-55.262042231857095,-55.262042231857095}, {0,0}}
		});

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

BOOST_AUTO_TEST_CASE(getNameLogTensorOp)
{
	LogTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LogTensorOp");
}

/**
LogGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLogGradTensorOp)
{
	LogGradTensorOp<double, Eigen::DefaultDevice>* ptrLogGrad = nullptr;
	LogGradTensorOp<double, Eigen::DefaultDevice>* nullPointerLogGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrLogGrad, nullPointerLogGrad);
}

BOOST_AUTO_TEST_CASE(destructorLogGradTensorOp)
{
	LogGradTensorOp<double, Eigen::DefaultDevice>* ptrLogGrad = nullptr;
	ptrLogGrad = new LogGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrLogGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionLogGradTensorOp)
{
	LogGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}}, // was 1e9 prior to selection
		{{1,1}, {0,0}},
		{{0.1,0.1}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-0.1,-0.1}, {0,0}} });

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

BOOST_AUTO_TEST_CASE(getNameLogGradTensorOp)
{
	LogGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LogGradTensorOp");
}

/**
	PowTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPowTensorOp)
{
	PowTensorOp<double, Eigen::DefaultDevice>* ptrPow = nullptr;
	PowTensorOp<double, Eigen::DefaultDevice>* nullPointerPow = nullptr;
	BOOST_CHECK_EQUAL(ptrPow, nullPointerPow);
}

BOOST_AUTO_TEST_CASE(destructorPowTensorOp)
{
	PowTensorOp<double, Eigen::DefaultDevice>* ptrPow = nullptr;
	ptrPow = new PowTensorOp<double, Eigen::DefaultDevice>(2);
	delete ptrPow;
}

BOOST_AUTO_TEST_CASE(operationfunctionPowTensorOp)
{
	PowTensorOp<double, Eigen::DefaultDevice> operation(0.5);
	const int batch_size = 7;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
    {{1e12,1e12}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}},
    {{-1e12,-1e12}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{3.1622776601683795,3.1622776601683795}, {0,0}},
    {{31622.776601683792,31622.776601683792}, {0,0}},
		{{0,0}, {0,0}},
		{{0,0}, {0,0}},
    {{0,0}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}

  PowTensorOp<double, Eigen::DefaultDevice> operation2(2.0);

  output.setZero();
  test.setValues({
    {{0,0}, {0,0}},
    {{1,1}, {0,0}},
    {{100,100}, {0,0}},
    {{1e9,1e9}, {0,0}},
    {{1,1}, {0,0}},  // TODO: Clip does not fix -nan(ind)
    {{100,100}, {0,0}},  // TODO: Clip does not fix -nan(ind)
    {{1e9,1e9}, {0,0}} });

  operation2(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

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
	PowTensorOp<double, Eigen::DefaultDevice> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowTensorOp");
}

/**
	PowGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPowGradTensorOp)
{
	PowGradTensorOp<double, Eigen::DefaultDevice>* ptrPowGrad = nullptr;
	PowGradTensorOp<double, Eigen::DefaultDevice>* nullPointerPowGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrPowGrad, nullPointerPowGrad);
}

BOOST_AUTO_TEST_CASE(destructorPowGradTensorOp)
{
	PowGradTensorOp<double, Eigen::DefaultDevice>* ptrPowGrad = nullptr;
	ptrPowGrad = new PowGradTensorOp<double, Eigen::DefaultDevice>(0.5);
	delete ptrPowGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionPowGradTensorOp)
{
	PowGradTensorOp<double, Eigen::DefaultDevice> operation(0.5);
	const int batch_size = 7;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
    {{0,0}, {0,0}},
    {{1,1}, {0,0}},
    {{10,10}, {0,0}},
    {{1e9,1e9}, {0,0}},
    {{-1,-1}, {0,0}},
    {{-10,-10}, {0,0}},
    {{-1e9,-1e9}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
	test.setValues({
		{{1.0e9,1.0e9}, {0,0}},
		{{0.5,0.5}, {0,0}},
		{{0.15811388300841897,0.15811388300841897}, {0,0}},
    {{1.5811388300841898e-05,1.5811388300841898e-05}, {0,0}},
		{{1.0e9,1.0e9}, {0,0}},  // TODO: why is this not zero?
		{{1.0e9,1.0e9}, {0,0}},
    {{1.0e9,1.0e9}, {0,0}} });

	operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

	// Test
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < memory_size; ++j) {
			for (int k = 0; k < layer_size; ++k) {
				BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4);
			}
		}
	}

  PowGradTensorOp<double, Eigen::DefaultDevice> operation2(2);
  input.setValues({
    {{0,0}, {0,0}},
    {{1,1}, {0,0}},
    {{10,10}, {0,0}},
    {{1e9,1e9}, {0,0}},
    {{-1,-1}, {0,0}},
    {{-10,-10}, {0,0}},
    {{-1e9,-1e9}, {0,0}} });
  output.setZero();
  test.setValues({
    {{0,0}, {0,0}},
    {{2.0,2.0}, {0,0}},
    {{20,20}, {0,0}},
    {{1e9,1e9}, {0,0}},
    {{-2,-2}, {0,0}},
    {{-20,-20}, {0,0}},
    {{-1e9,-1e9}, {0,0}} });

  operation2(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

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
	PowGradTensorOp<double, Eigen::DefaultDevice> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowGradTensorOp");
}

/**
	LeakyReLUTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<double, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	LeakyReLUTensorOp<double, Eigen::DefaultDevice>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

BOOST_AUTO_TEST_CASE(destructorLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<double, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUTensorOp<double, Eigen::DefaultDevice>();
	delete ptrLeakyReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<double, Eigen::DefaultDevice> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionLeakyReLUTensorOp)
{
	LeakyReLUTensorOp<double, Eigen::DefaultDevice> operation(0.1);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	LeakyReLUTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUTensorOp");
}

/**
	LeakyReLUGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<double, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	LeakyReLUGradTensorOp<double, Eigen::DefaultDevice>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

BOOST_AUTO_TEST_CASE(destructorLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<double, Eigen::DefaultDevice>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrLeakyReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<double, Eigen::DefaultDevice> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(operationfunctionLeakyReLUGradTensorOp)
{
	LeakyReLUGradTensorOp<double, Eigen::DefaultDevice> operation(0.1);
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
	LeakyReLUGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUGradTensorOp");
}

/**
SinTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinTensorOp)
{
	SinTensorOp<double, Eigen::DefaultDevice>* ptrSin = nullptr;
	SinTensorOp<double, Eigen::DefaultDevice>* nullPointerSin = nullptr;
	BOOST_CHECK_EQUAL(ptrSin, nullPointerSin);
}

BOOST_AUTO_TEST_CASE(destructorSinTensorOp)
{
	SinTensorOp<double, Eigen::DefaultDevice>* ptrSin = nullptr;
	ptrSin = new SinTensorOp<double, Eigen::DefaultDevice>();
	delete ptrSin;
}

BOOST_AUTO_TEST_CASE(operationfunctionSinTensorOp)
{
	SinTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
				//BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4); //TODO: fixme
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSinTensorOp)
{
	SinTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinTensorOp");
}

/**
SinGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinGradTensorOp)
{
	SinGradTensorOp<double, Eigen::DefaultDevice>* ptrSinGrad = nullptr;
	SinGradTensorOp<double, Eigen::DefaultDevice>* nullPointerSinGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrSinGrad, nullPointerSinGrad);
}

BOOST_AUTO_TEST_CASE(destructorSinGradTensorOp)
{
	SinGradTensorOp<double, Eigen::DefaultDevice>* ptrSinGrad = nullptr;
	ptrSinGrad = new SinGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrSinGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionSinGradTensorOp)
{
	SinGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
				//BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4); //TODO: fixme
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameSinGradTensorOp)
{
	SinGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinGradTensorOp");
}

/**
CosTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCosTensorOp)
{
	CosTensorOp<double, Eigen::DefaultDevice>* ptrCos = nullptr;
	CosTensorOp<double, Eigen::DefaultDevice>* nullPointerCos = nullptr;
	BOOST_CHECK_EQUAL(ptrCos, nullPointerCos);
}

BOOST_AUTO_TEST_CASE(destructorCosTensorOp)
{
	CosTensorOp<double, Eigen::DefaultDevice>* ptrCos = nullptr;
	ptrCos = new CosTensorOp<double, Eigen::DefaultDevice>();
	delete ptrCos;
}

BOOST_AUTO_TEST_CASE(operationfunctionCosTensorOp)
{
	CosTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
				//BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4); //TODO: fixme
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameCosTensorOp)
{
	CosTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CosTensorOp");
}

/**
CosGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCosGradTensorOp)
{
	CosGradTensorOp<double, Eigen::DefaultDevice>* ptrCosGrad = nullptr;
	CosGradTensorOp<double, Eigen::DefaultDevice>* nullPointerCosGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrCosGrad, nullPointerCosGrad);
}

BOOST_AUTO_TEST_CASE(destructorCosGradTensorOp)
{
	CosGradTensorOp<double, Eigen::DefaultDevice>* ptrCosGrad = nullptr;
	ptrCosGrad = new CosGradTensorOp<double, Eigen::DefaultDevice>();
	delete ptrCosGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionCosGradTensorOp)
{
	CosGradTensorOp<double, Eigen::DefaultDevice> operation;
	const int batch_size = 5;
	const int memory_size = 2;
	const int layer_size = 2;
	Eigen::DefaultDevice device;
	Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
	input.setValues({
		{{0,0}, {0,0}},
		{{1,1}, {0,0}},
		{{10,10}, {0,0}},
		{{-1,-1}, {0,0}},
		{{-10,-10}, {0,0}} });
	Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
	output.setZero();
	Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
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
				//BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4); //TODO: fixme
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(getNameCosGradTensorOp)
{
	CosGradTensorOp<double, Eigen::DefaultDevice> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CosGradTensorOp");
}

BOOST_AUTO_TEST_CASE(operationfunctionBatchNormTensorOp)
{
  BatchNormTensorOp<double, Eigen::DefaultDevice> operation;
  const int batch_size = 5;
  const int memory_size = 2;
  const int layer_size = 2;
  Eigen::DefaultDevice device;
  Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
  input.setValues({
    {{0,0.1}, {0,0}},
    {{1,1}, {0,0}},
    {{10,10}, {0,0}},
    {{-1,-1}, {0,0}},
    {{-10,-10}, {0,0}} });
  Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
  output.setZero();
  Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
  test.setValues({
    {{0,2.2360679774997898}, {0,0}},
    {{2.2360679774997898,2.2360679774997898}, {0,0}},
    {{2.2360679774997898,2.2360679774997898}, {0,0}},
    {{-2.2360679774997898,-2.2360679774997898}, {0,0}},
    {{-2.2360679774997898,-2.2360679774997898}, {0,0}} });

  operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

  // Test
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < memory_size; ++j) {
      for (int k = 0; k < layer_size; ++k) {
        BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4); //TODO: fixme
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(getNameBatchNormTensorOp)
{
  BatchNormTensorOp<double, Eigen::DefaultDevice> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "BatchNormTensorOp");
}

/**
BatchNormGradTensorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBatchNormGradTensorOp)
{
  BatchNormGradTensorOp<double, Eigen::DefaultDevice>* ptrBatchNormGrad = nullptr;
  BatchNormGradTensorOp<double, Eigen::DefaultDevice>* nullPointerBatchNormGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrBatchNormGrad, nullPointerBatchNormGrad);
}

BOOST_AUTO_TEST_CASE(destructorBatchNormGradTensorOp)
{
  BatchNormGradTensorOp<double, Eigen::DefaultDevice>* ptrBatchNormGrad = nullptr;
  ptrBatchNormGrad = new BatchNormGradTensorOp<double, Eigen::DefaultDevice>();
  delete ptrBatchNormGrad;
}

BOOST_AUTO_TEST_CASE(operationfunctionBatchNormGradTensorOp)
{
  BatchNormGradTensorOp<double, Eigen::DefaultDevice> operation;
  const int batch_size = 5;
  const int memory_size = 2;
  const int layer_size = 2;
  Eigen::DefaultDevice device;
  Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
  input.setValues({
    {{0,0.1}, {0,0}},
    {{1,1}, {0,0}},
    {{10,10}, {0,0}},
    {{-1,-1}, {0,0}},
    {{-10,-10}, {0,0}} });
  Eigen::Tensor<double, 3> output(batch_size, memory_size, layer_size);
  output.setZero();
  Eigen::Tensor<double, 3> test(batch_size, memory_size, layer_size);
  test.setValues({
    {{0,4.5519144009631268e-15}, {0,0}},
    {{0,1.7948528498063337e-15}, {0,0}},
    {{0,1.776535187764253e-15}, {0,0}},
    {{0,1.7934306267839513e-15}, {0,0}},
    {{0,1.7765337666674127e-15}, {0,0}} });

  operation(input.data(), output.data(), batch_size, memory_size, layer_size, 0, device);

  // Test
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < memory_size; ++j) {
      for (int k = 0; k < layer_size; ++k) {
        BOOST_CHECK_CLOSE(output(i, j, k), test(i, j, k), 1e-4); //TODO: fixme
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(getNameBatchNormGradTensorOp)
{
  BatchNormGradTensorOp<double, Eigen::DefaultDevice> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "BatchNormGradTensorOp");
}

BOOST_AUTO_TEST_CASE(operatorGradientCheckTensorOp)
{
  // Define the gradient checker
  GradientCheckTensorOp<double, Eigen::DefaultDevice> operation;
  operation.eps_ = 1e-7;

  // Setup the input
  const int batch_size = 5;
  const int memory_size = 2;
  const int layer_size = 2;
  Eigen::DefaultDevice device;
  Eigen::Tensor<double, 3> input(batch_size, memory_size, layer_size);
  input.setValues({
    {{0,0}, {0,0}},
    {{1,1}, {0,0}},
    {{10,10}, {0,0}},
    {{-1,-1}, {0,0}},
    {{-10,-10}, {0,0}} });
  Eigen::Tensor<double, 3> input_f_plus(batch_size, memory_size, layer_size);
  input_f_plus.setZero();
  Eigen::Tensor<double, 3> input_f_min(batch_size, memory_size, layer_size);
  input_f_min.setZero();
  Eigen::Tensor<double, 3> input_b(batch_size, memory_size, layer_size);
  input_b.setZero();
  Eigen::Tensor<double, 0> output;
  output.setZero();

  // Check Sigmoid
  operation.forward_ = std::make_shared<SigmoidTensorOp<double, Eigen::DefaultDevice>>(SigmoidTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<SigmoidGradTensorOp<double, Eigen::DefaultDevice>>(SigmoidGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check ReLU
  input.setValues({ {{0.1,0.1}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<ReLUTensorOp<double, Eigen::DefaultDevice>>(ReLUTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<ReLUGradTensorOp<double, Eigen::DefaultDevice>>(ReLUGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check ELU
  input.setValues({ {{0,0}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<ELUTensorOp<double, Eigen::DefaultDevice>>(ELUTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<ELUGradTensorOp<double, Eigen::DefaultDevice>>(ELUGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check LeakyReLU
  input.setValues({ {{0.1,0.1}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<LeakyReLUTensorOp<double, Eigen::DefaultDevice>>(LeakyReLUTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<LeakyReLUGradTensorOp<double, Eigen::DefaultDevice>>(LeakyReLUGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check TanH
  input.setValues({ {{0,0}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<TanHTensorOp<double, Eigen::DefaultDevice>>(TanHTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<TanHGradTensorOp<double, Eigen::DefaultDevice>>(TanHGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check Linear
  input.setValues({ {{0,0}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<LinearTensorOp<double, Eigen::DefaultDevice>>(LinearTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<LinearGradTensorOp<double, Eigen::DefaultDevice>>(LinearGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check Pow
  input.setValues({ {{0,0}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<PowTensorOp<double, Eigen::DefaultDevice>>(PowTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<PowGradTensorOp<double, Eigen::DefaultDevice>>(PowGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check Inverse
  input.setValues({ {{0.1,0.1}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<InverseTensorOp<double, Eigen::DefaultDevice>>(InverseTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<InverseGradTensorOp<double, Eigen::DefaultDevice>>(InverseGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check Log
  input.setValues({ {{0.1,0.1}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{100,100}, {0,0}}, {{1000,1000}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<LogTensorOp<double, Eigen::DefaultDevice>>(LogTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<LogGradTensorOp<double, Eigen::DefaultDevice>>(LogGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check Exponential
  input.setValues({ {{0,0}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<ExponentialTensorOp<double, Eigen::DefaultDevice>>(ExponentialTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<ExponentialGradTensorOp<double, Eigen::DefaultDevice>>(ExponentialGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);

  // Check BatchNorm
  input.setValues({ {{10,0.1}, {0,0}}, {{1,1}, {0,0}}, {{10,10}, {0,0}}, {{-1,-1}, {0,0}}, {{-10,-10}, {0,0}} });
  input_f_plus.setZero();
  input_f_min.setZero();
  input_b.setZero();
  output.setZero();
  operation.forward_ = std::make_shared<BatchNormTensorOp<double, Eigen::DefaultDevice>>(BatchNormTensorOp<double, Eigen::DefaultDevice>());
  operation.reverse_ = std::make_shared<BatchNormGradTensorOp<double, Eigen::DefaultDevice>>(BatchNormGradTensorOp<double, Eigen::DefaultDevice>());
  operation(input.data(), input_f_plus.data(), input_f_min.data(), input_b.data(), output.data(), batch_size, memory_size, layer_size, 0, device);
  BOOST_CHECK(output(0) < operation.eps_);
  std::cout << "Gradient Check BatchNorm: " << output(0) << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()