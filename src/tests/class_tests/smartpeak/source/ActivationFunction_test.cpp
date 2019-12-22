/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ActivationFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/ActivationFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(activationFunction)

/**
  ReLUOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluOp) 
{
  ReLUOp<double>* ptrReLU = nullptr;
  ReLUOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorReluOp) 
{
  ReLUOp<double>* ptrReLU = nullptr;
	ptrReLU = new ReLUOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersReluOp)
{
  // Test defaults
  ReLUOp<double> operation_defaults;
  BOOST_CHECK_CLOSE(operation_defaults.getEps(), 1e-6, 1e-6);
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
  ReLUOp<double> operation(10, 20, 30);
  BOOST_CHECK_CLOSE(operation.getEps(), 10, 1e-6);
  BOOST_CHECK_CLOSE(operation.getMin(), 20, 1e-6);
  BOOST_CHECK_CLOSE(operation.getMax(), 30, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameReLUOp)
{
	ReLUOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUOp");
}
BOOST_AUTO_TEST_CASE(copyReluOp)
{
  std::shared_ptr<ActivationOp<double>> relu_ptr_1 = std::make_shared<ReLUOp<double>>();
  std::shared_ptr<ActivationOp<double>> relu_ptr_2;
  relu_ptr_2 = std::shared_ptr<ActivationOp<double>>(relu_ptr_1);
  BOOST_CHECK_EQUAL(relu_ptr_1.get(), relu_ptr_2.get());
  relu_ptr_2 = std::shared_ptr<ActivationOp<double>>(relu_ptr_1.get()->copy());
  BOOST_CHECK_NE(relu_ptr_1.get(), relu_ptr_2.get());
}

/**
  ReLUGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReluGradOp) 
{
  ReLUGradOp<double>* ptrReLUGrad = nullptr;
  ReLUGradOp<double>* nullPointerReLUGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReLUGrad, nullPointerReLUGrad);
}

BOOST_AUTO_TEST_CASE(destructorReluGradOp) 
{
  ReLUGradOp<double>* ptrReLUGrad = nullptr;
	ptrReLUGrad = new ReLUGradOp<double>();
  delete ptrReLUGrad;
}

BOOST_AUTO_TEST_CASE(getNameReLUGradOp)
{
	ReLUGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUGradOp");
}

/**
  ELUOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluOp) 
{
  ELUOp<double>* ptrELU = nullptr;
  ELUOp<double>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluOp) 
{
  ELUOp<double>* ptrELU = nullptr;
	ptrELU = new ELUOp<double>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluOp) 
{
  ELUOp<double> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(getNameELUOp)
{
	ELUOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ELUOp");
}

/**
  ELUGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEluGradOp) 
{
  ELUGradOp<double>* ptrELU = nullptr;
  ELUGradOp<double>* nullPointerELU = nullptr;
  BOOST_CHECK_EQUAL(ptrELU, nullPointerELU);
}

BOOST_AUTO_TEST_CASE(destructorEluGradOp) 
{
  ELUGradOp<double>* ptrELU = nullptr;
	ptrELU = new ELUGradOp<double>();
  delete ptrELU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEluGradOp) 
{
  ELUGradOp<double> operation;
  operation.setAlpha(1.0);

  BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(getNameELUGradOp)
{
	ELUGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ELUGradOp");
}

/**
  SigmoidOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSigmoidOp) 
{
  SigmoidOp<double>* ptrSigmoid = nullptr;
  SigmoidOp<double>* nullPointerSigmoid = nullptr;
  BOOST_CHECK_EQUAL(ptrSigmoid, nullPointerSigmoid);
}

BOOST_AUTO_TEST_CASE(destructorSigmoidOp) 
{
  SigmoidOp<double>* ptrSigmoid = nullptr;
	ptrSigmoid = new SigmoidOp<double>();
  delete ptrSigmoid;
}

BOOST_AUTO_TEST_CASE(getNameSigmoidOp)
{
	SigmoidOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SigmoidOp");
}

/**
  SigmoidGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSigmoidGradOp) 
{
  SigmoidGradOp<double>* ptrSigmoidGrad = nullptr;
  SigmoidGradOp<double>* nullPointerSigmoidGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrSigmoidGrad, nullPointerSigmoidGrad);
}

BOOST_AUTO_TEST_CASE(destructorSigmoidGradOp) 
{
  SigmoidGradOp<double>* ptrSigmoidGrad = nullptr;
	ptrSigmoidGrad = new SigmoidGradOp<double>();
  delete ptrSigmoidGrad;
}

BOOST_AUTO_TEST_CASE(getNameSigmoidGradOp)
{
	SigmoidGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SigmoidGradOp");
}

/**
  TanHOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorTanHOp) 
{
  TanHOp<double>* ptrTanH = nullptr;
  TanHOp<double>* nullPointerTanH = nullptr;
  BOOST_CHECK_EQUAL(ptrTanH, nullPointerTanH);
}

BOOST_AUTO_TEST_CASE(destructorTanHOp) 
{
  TanHOp<double>* ptrTanH = nullptr;
	ptrTanH = new TanHOp<double>();
  delete ptrTanH;
}

BOOST_AUTO_TEST_CASE(getNameTanHOp)
{
	TanHOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "TanHOp");
}

/**
  TanHGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorTanHGradOp) 
{
  TanHGradOp<double>* ptrTanHGrad = nullptr;
  TanHGradOp<double>* nullPointerTanHGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrTanHGrad, nullPointerTanHGrad);
}

BOOST_AUTO_TEST_CASE(destructorTanHGradOp) 
{
  TanHGradOp<double>* ptrTanHGrad = nullptr;
	ptrTanHGrad = new TanHGradOp<double>();
  delete ptrTanHGrad;
}

BOOST_AUTO_TEST_CASE(getNameTanHGradOp)
{
	TanHGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "TanHGradOp");
}

/**
  ReTanHOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReTanHOp) 
{
  ReTanHOp<double>* ptrReTanH = nullptr;
  ReTanHOp<double>* nullPointerReTanH = nullptr;
  BOOST_CHECK_EQUAL(ptrReTanH, nullPointerReTanH);
}

BOOST_AUTO_TEST_CASE(destructorReTanHOp) 
{
  ReTanHOp<double>* ptrReTanH = nullptr;
	ptrReTanH = new ReTanHOp<double>();
  delete ptrReTanH;
}

BOOST_AUTO_TEST_CASE(getNameReTanHOp)
{
	ReTanHOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReTanHOp");
}

/**
  ReTanHGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorReTanHGradOp) 
{
  ReTanHGradOp<double>* ptrReTanHGrad = nullptr;
  ReTanHGradOp<double>* nullPointerReTanHGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrReTanHGrad, nullPointerReTanHGrad);
}

BOOST_AUTO_TEST_CASE(destructorReTanHGradOp) 
{
  ReTanHGradOp<double>* ptrReTanHGrad = nullptr;
	ptrReTanHGrad = new ReTanHGradOp<double>();
  delete ptrReTanHGrad;
}

BOOST_AUTO_TEST_CASE(getNameReTanHGradOp)
{
	ReTanHGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReTanHGradOp");
}

/**
LinearOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLinearOp)
{
	LinearOp<double>* ptrLinear = nullptr;
	LinearOp<double>* nullPointerLinear = nullptr;
	BOOST_CHECK_EQUAL(ptrLinear, nullPointerLinear);
}

BOOST_AUTO_TEST_CASE(destructorLinearOp)
{
	LinearOp<double>* ptrLinear = nullptr;
	ptrLinear = new LinearOp<double>();
	delete ptrLinear;
}

BOOST_AUTO_TEST_CASE(getNameLinearOp)
{
	LinearOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LinearOp");
}

/**
LinearGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLinearGradOp)
{
	LinearGradOp<double>* ptrLinearGrad = nullptr;
	LinearGradOp<double>* nullPointerLinearGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrLinearGrad, nullPointerLinearGrad);
}

BOOST_AUTO_TEST_CASE(destructorLinearGradOp)
{
	LinearGradOp<double>* ptrLinearGrad = nullptr;
	ptrLinearGrad = new LinearGradOp<double>();
	delete ptrLinearGrad;
}

BOOST_AUTO_TEST_CASE(getNameLinearGradOp)
{
	LinearGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LinearGradOp");
}

/**
InverseOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorInverseOp)
{
	InverseOp<double>* ptrInverse = nullptr;
	InverseOp<double>* nullPointerInverse = nullptr;
	BOOST_CHECK_EQUAL(ptrInverse, nullPointerInverse);
}

BOOST_AUTO_TEST_CASE(destructorInverseOp)
{
	InverseOp<double>* ptrInverse = nullptr;
	ptrInverse = new InverseOp<double>();
	delete ptrInverse;
}

BOOST_AUTO_TEST_CASE(getNameInverseOp)
{
	InverseOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "InverseOp");
}

/**
InverseGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorInverseGradOp)
{
	InverseGradOp<double>* ptrInverseGrad = nullptr;
	InverseGradOp<double>* nullPointerInverseGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrInverseGrad, nullPointerInverseGrad);
}

BOOST_AUTO_TEST_CASE(destructorInverseGradOp)
{
	InverseGradOp<double>* ptrInverseGrad = nullptr;
	ptrInverseGrad = new InverseGradOp<double>();
	delete ptrInverseGrad;
}

BOOST_AUTO_TEST_CASE(getNameInverseGradOp)
{
	InverseGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "InverseGradOp");
}

/**
ExponentialOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorExponentialOp)
{
	ExponentialOp<double>* ptrExponential = nullptr;
	ExponentialOp<double>* nullPointerExponential = nullptr;
	BOOST_CHECK_EQUAL(ptrExponential, nullPointerExponential);
}

BOOST_AUTO_TEST_CASE(destructorExponentialOp)
{
	ExponentialOp<double>* ptrExponential = nullptr;
	ptrExponential = new ExponentialOp<double>();
	delete ptrExponential;
}

BOOST_AUTO_TEST_CASE(getNameExponentialOp)
{
	ExponentialOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialOp");
}

/**
ExponentialGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorExponentialGradOp)
{
	ExponentialGradOp<double>* ptrExponentialGrad = nullptr;
	ExponentialGradOp<double>* nullPointerExponentialGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrExponentialGrad, nullPointerExponentialGrad);
}

BOOST_AUTO_TEST_CASE(destructorExponentialGradOp)
{
	ExponentialGradOp<double>* ptrExponentialGrad = nullptr;
	ptrExponentialGrad = new ExponentialGradOp<double>();
	delete ptrExponentialGrad;
}

BOOST_AUTO_TEST_CASE(getNameExponentialGradOp)
{
	ExponentialGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialGradOp");
}

/**
LogOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLogOp)
{
	LogOp<double>* ptrLog = nullptr;
	LogOp<double>* nullPointerLog = nullptr;
	BOOST_CHECK_EQUAL(ptrLog, nullPointerLog);
}

BOOST_AUTO_TEST_CASE(destructorLogOp)
{
	LogOp<double>* ptrLog = nullptr;
	ptrLog = new LogOp<double>();
	delete ptrLog;
}

BOOST_AUTO_TEST_CASE(getNameLogOp)
{
	LogOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LogOp");
}

/**
LogGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLogGradOp)
{
	LogGradOp<double>* ptrLogGrad = nullptr;
	LogGradOp<double>* nullPointerLogGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrLogGrad, nullPointerLogGrad);
}

BOOST_AUTO_TEST_CASE(destructorLogGradOp)
{
	LogGradOp<double>* ptrLogGrad = nullptr;
	ptrLogGrad = new LogGradOp<double>();
	delete ptrLogGrad;
}

BOOST_AUTO_TEST_CASE(getNameLogGradOp)
{
	LogGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LogGradOp");
}

/**
	PowOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPowOp)
{
	PowOp<double>* ptrPow = nullptr;
	PowOp<double>* nullPointerPow = nullptr;
	BOOST_CHECK_EQUAL(ptrPow, nullPointerPow);
}

BOOST_AUTO_TEST_CASE(destructorPowOp)
{
	PowOp<double>* ptrPow = nullptr;
	ptrPow = new PowOp<double>(2);
	delete ptrPow;
}

BOOST_AUTO_TEST_CASE(getNamePowOp)
{
	PowOp<double> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowOp");
}

/**
	PowGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorPowGradOp)
{
	PowGradOp<double>* ptrPowGrad = nullptr;
	PowGradOp<double>* nullPointerPowGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrPowGrad, nullPointerPowGrad);
}

BOOST_AUTO_TEST_CASE(destructorPowGradOp)
{
	PowGradOp<double>* ptrPowGrad = nullptr;
	ptrPowGrad = new PowGradOp<double>(0.5);
	delete ptrPowGrad;
}

BOOST_AUTO_TEST_CASE(getNamePowGradOp)
{
	PowGradOp<double> operation(0.5);

	BOOST_CHECK_EQUAL(operation.getName(), "PowGradOp");
}

/**
	LeakyReLUOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLeakyReLUOp)
{
	LeakyReLUOp<double>* ptrLeakyReLU = nullptr;
	LeakyReLUOp<double>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

BOOST_AUTO_TEST_CASE(destructorLeakyReLUOp)
{
	LeakyReLUOp<double>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUOp<double>();
	delete ptrLeakyReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersLeakyReLUOp)
{
	LeakyReLUOp<double> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(getNameLeakyReLUOp)
{
	LeakyReLUOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUOp");
}

/**
	LeakyReLUGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorLeakyReLUGradOp)
{
	LeakyReLUGradOp<double>* ptrLeakyReLU = nullptr;
	LeakyReLUGradOp<double>* nullPointerLeakyReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrLeakyReLU, nullPointerLeakyReLU);
}

BOOST_AUTO_TEST_CASE(destructorLeakyReLUGradOp)
{
	LeakyReLUGradOp<double>* ptrLeakyReLU = nullptr;
	ptrLeakyReLU = new LeakyReLUGradOp<double>();
	delete ptrLeakyReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersLeakyReLUGradOp)
{
	LeakyReLUGradOp<double> operation;
	operation.setAlpha(1.0);

	BOOST_CHECK_EQUAL(operation.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(getNameLeakyReLUGradOp)
{
	LeakyReLUGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUGradOp");
}

/**
SinOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinOp)
{
	SinOp<double>* ptrSin = nullptr;
	SinOp<double>* nullPointerSin = nullptr;
	BOOST_CHECK_EQUAL(ptrSin, nullPointerSin);
}

BOOST_AUTO_TEST_CASE(destructorSinOp)
{
	SinOp<double>* ptrSin = nullptr;
	ptrSin = new SinOp<double>();
	delete ptrSin;
}

BOOST_AUTO_TEST_CASE(getNameSinOp)
{
	SinOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinOp");
}

/**
SinGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSinGradOp)
{
	SinGradOp<double>* ptrSinGrad = nullptr;
	SinGradOp<double>* nullPointerSinGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrSinGrad, nullPointerSinGrad);
}

BOOST_AUTO_TEST_CASE(destructorSinGradOp)
{
	SinGradOp<double>* ptrSinGrad = nullptr;
	ptrSinGrad = new SinGradOp<double>();
	delete ptrSinGrad;
}

BOOST_AUTO_TEST_CASE(getNameSinGradOp)
{
	SinGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SinGradOp");
}

/**
CosOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCosOp)
{
	CosOp<double>* ptrCos = nullptr;
	CosOp<double>* nullPointerCos = nullptr;
	BOOST_CHECK_EQUAL(ptrCos, nullPointerCos);
}

BOOST_AUTO_TEST_CASE(destructorCosOp)
{
	CosOp<double>* ptrCos = nullptr;
	ptrCos = new CosOp<double>();
	delete ptrCos;
}

BOOST_AUTO_TEST_CASE(getNameCosOp)
{
	CosOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CosOp");
}

/**
CosGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCosGradOp)
{
	CosGradOp<double>* ptrCosGrad = nullptr;
	CosGradOp<double>* nullPointerCosGrad = nullptr;
	BOOST_CHECK_EQUAL(ptrCosGrad, nullPointerCosGrad);
}

BOOST_AUTO_TEST_CASE(destructorCosGradOp)
{
	CosGradOp<double>* ptrCosGrad = nullptr;
	ptrCosGrad = new CosGradOp<double>();
	delete ptrCosGrad;
}

BOOST_AUTO_TEST_CASE(getNameCosGradOp)
{
	CosGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CosGradOp");
}

/**
BatchNormOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBatchNormOp)
{
  BatchNormOp<double>* ptrBatchNorm = nullptr;
  BatchNormOp<double>* nullPointerBatchNorm = nullptr;
  BOOST_CHECK_EQUAL(ptrBatchNorm, nullPointerBatchNorm);
}

BOOST_AUTO_TEST_CASE(destructorBatchNormOp)
{
  BatchNormOp<double>* ptrBatchNorm = nullptr;
  ptrBatchNorm = new BatchNormOp<double>();
  delete ptrBatchNorm;
}

BOOST_AUTO_TEST_CASE(getNameBatchNormOp)
{
  BatchNormOp<double> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "BatchNormOp");
}

/**
BatchNormGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBatchNormGradOp)
{
  BatchNormGradOp<double>* ptrBatchNormGrad = nullptr;
  BatchNormGradOp<double>* nullPointerBatchNormGrad = nullptr;
  BOOST_CHECK_EQUAL(ptrBatchNormGrad, nullPointerBatchNormGrad);
}

BOOST_AUTO_TEST_CASE(destructorBatchNormGradOp)
{
  BatchNormGradOp<double>* ptrBatchNormGrad = nullptr;
  ptrBatchNormGrad = new BatchNormGradOp<double>();
  delete ptrBatchNormGrad;
}

BOOST_AUTO_TEST_CASE(getNameBatchNormGradOp)
{
  BatchNormGradOp<double> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "BatchNormGradOp");
}

BOOST_AUTO_TEST_SUITE_END()