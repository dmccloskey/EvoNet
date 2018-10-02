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

BOOST_AUTO_TEST_CASE(operationfunctionReluOp) 
{
  ReLUOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(getNameReLUOp)
{
	ReLUOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ReLUOp");
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

BOOST_AUTO_TEST_CASE(operationfunctionReluGradOp) 
{
  ReLUGradOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.0, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionEluOp) 
{
  ELUOp<double> operation(1.0); 
  
  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), -0.63212055882855767, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), -0.99995460007023751, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionEluGradOp) 
{
  ELUGradOp<double> operation(1.0); 

  BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.36787944117144233, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 4.5399929762490743e-05, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionSigmoidOp) 
{
  SigmoidOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.5, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 0.7310585786300049, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 0.99995460213129761, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.2689414213699951, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 4.5397868702434395e-05, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionSigmoidGradOp) 
{
  SigmoidGradOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.25, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 0.19661193324148185, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 4.5395807735951673e-05, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.19661193324148185, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 4.5395807735907655e-05, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionTanHOp) 
{
  TanHOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 0.76159415595576485, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 0.99999999587769262, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), -0.76159415595576485, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), -0.99999999587769262, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionTanHGradOp) 
{
  TanHGradOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 0.41997434161402614, 1e-4);
  BOOST_CHECK_CLOSE(operation(10.0), 8.2446147686709992e-09, 1e-4);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.41997434161402614, 1e-4);
  BOOST_CHECK_CLOSE(operation(-10.0), 8.2446147686709992e-09, 1e-4);
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

BOOST_AUTO_TEST_CASE(operationfunctionReTanHOp) 
{
  ReTanHOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 0.76159415595576485, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 0.99999999587769262, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.0, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionReTanHGradOp) 
{
  ReTanHGradOp<double> operation;

  BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(1.0), 0.41997434161402614, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 8.2446147686709992e-09, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.0, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionLinearOp)
{
	LinearOp<double> operation;

	BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), -10.0, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionLinearGradOp)
{
	LinearGradOp<double> operation;

	BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), 1.0, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionInverseOp)
{
	InverseOp<double> operation;

	BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 0.1, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), -0.1, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionInverseGradOp)
{
	InverseGradOp<double> operation;

	BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), -0.01, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), -1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), -0.01, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionExponentialOp)
{
	ExponentialOp<double> operation;

	BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 2.718281828, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 22026.46579, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), 0.367879441, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), 4.53999E-05, 1e-4);
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

BOOST_AUTO_TEST_CASE(operationfunctionExponentialGradOp)
{
	ExponentialGradOp<double> operation;

	BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 2.718281828, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 22026.46579, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), 0.367879441, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), 4.53999E-05, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameExponentialGradOp)
{
	ExponentialGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ExponentialGradOp");
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

BOOST_AUTO_TEST_CASE(operationfunctionPowOp)
{
	PowOp<double> operation(0.5);

	BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 3.1622776601683795, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), -1.0e9, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), -1.0e9, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionPowGradOp)
{
	PowGradOp<double> operation(0.5);

	BOOST_CHECK_CLOSE(operation(0.0), 1.0e9, 1e-6);
	BOOST_CHECK_CLOSE(operation(1.0), 0.5, 1e-6);
	BOOST_CHECK_CLOSE(operation(10.0), 0.15811388300841897, 1e-6);
	BOOST_CHECK_CLOSE(operation(-1.0), -1.0e9, 1e-6);
	BOOST_CHECK_CLOSE(operation(-10.0), -1.0e9, 1e-6);
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

BOOST_AUTO_TEST_CASE(operationfunctionLeakyReLUOp)
{
	LeakyReLUOp<double> operation(0.1);

	BOOST_CHECK_CLOSE(operation(0.0), 0.0, 1e-4);
	BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(operation(10.0), 10.0, 1e-4);
	BOOST_CHECK_CLOSE(operation(-1.0), -0.1, 1e-4);
	BOOST_CHECK_CLOSE(operation(-10.0), -1.0, 1e-4);
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

BOOST_AUTO_TEST_CASE(operationfunctionLeakyReLUGradOp)
{
	LeakyReLUGradOp<double> operation(0.1);

	BOOST_CHECK_CLOSE(operation(0.0), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(operation(1.0), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(operation(10.0), 1.0, 1e-4);
	BOOST_CHECK_CLOSE(operation(-1.0), 0.1, 1e-4);
	BOOST_CHECK_CLOSE(operation(-10.0), 0.1, 1e-4);
}

BOOST_AUTO_TEST_CASE(getNameLeakyReLUGradOp)
{
	LeakyReLUGradOp<double> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "LeakyReLUGradOp");
}

BOOST_AUTO_TEST_SUITE_END()