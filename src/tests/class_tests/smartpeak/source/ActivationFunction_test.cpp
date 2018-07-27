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
  BOOST_CHECK_CLOSE(operation(1.0), 0.2689414213699951, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 4.5397868702434395e-05, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.7310585786300049, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 0.99995460213129761, 1e-6);
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
  BOOST_CHECK_CLOSE(operation(1.0), 0.41997434161402614, 1e-6);
  BOOST_CHECK_CLOSE(operation(10.0), 8.2446147686709992e-09, 1e-6);
  BOOST_CHECK_CLOSE(operation(-1.0), 0.41997434161402614, 1e-6);
  BOOST_CHECK_CLOSE(operation(-10.0), 8.2446147686709992e-09, 1e-6);
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

BOOST_AUTO_TEST_SUITE_END()