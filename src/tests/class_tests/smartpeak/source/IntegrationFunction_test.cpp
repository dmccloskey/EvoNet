/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE IntegrationFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/IntegrationFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(integrationFunction)

/**
 SumOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorSumOp) 
{
 SumOp<float>* ptrReLU = nullptr;
 SumOp<float>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumOp) 
{
	SumOp<float>* ptrReLU = nullptr;
	ptrReLU = new SumOp<float>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameSumOp)
{
	SumOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumOp");
}

/**
ProdOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdOp)
{
	ProdOp<float>* ptrReLU = nullptr;
	ProdOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdOp)
{
	ProdOp<float>* ptrReLU = nullptr;
	ptrReLU = new ProdOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameProdOp)
{
	ProdOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdOp");
}

/**
ProdSCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdSCOp)
{
  ProdSCOp<float>* ptrReLU = nullptr;
  ProdSCOp<float>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdSCOp)
{
  ProdSCOp<float>* ptrReLU = nullptr;
  ptrReLU = new ProdSCOp<float>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameProdSCOp)
{
  ProdSCOp<float> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "ProdSCOp");
}

/**
MaxOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxOp)
{
	MaxOp<float>* ptrReLU = nullptr;
	MaxOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxOp)
{
	MaxOp<float>* ptrReLU = nullptr;
	ptrReLU = new MaxOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMaxOp)
{
	MaxOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxOp");
}

/**
MinOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMinOp)
{
  MinOp<float>* ptrReLU = nullptr;
  MinOp<float>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMinOp)
{
  MinOp<float>* ptrReLU = nullptr;
  ptrReLU = new MinOp<float>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMinOp)
{
  MinOp<float> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "MinOp");
}

/**
 MeanOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanOp)
{
	MeanOp<float>* ptrReLU = nullptr;
	MeanOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanOp)
{
	MeanOp<float>* ptrReLU = nullptr;
	ptrReLU = new MeanOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMeanOp)
{
	MeanOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanOp");
}

/**
 VarModOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarModOp)
{
	VarModOp<float>* ptrReLU = nullptr;
	VarModOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModOp)
{
	VarModOp<float>* ptrReLU = nullptr;
	ptrReLU = new VarModOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameVarModOp)
{
	VarModOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModOp");
}

/**
 CountOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountOp)
{
	CountOp<float>* ptrReLU = nullptr;
	CountOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountOp)
{
	CountOp<float>* ptrReLU = nullptr;
	ptrReLU = new CountOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameCountOp)
{
	CountOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountOp");
}

/**
SumErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumErrorOp)
{
	SumErrorOp<float>* ptrReLU = nullptr;
	SumErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumErrorOp)
{
	SumErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new SumErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameSumErrorOp)
{
	SumErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumErrorOp");
}

/**
ProdErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdErrorOp)
{
	ProdErrorOp<float>* ptrReLU = nullptr;
	ProdErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdErrorOp)
{
	ProdErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new ProdErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameProdErrorOp)
{
	ProdErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdErrorOp");
}

/**
MaxErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxErrorOp)
{
	MaxErrorOp<float>* ptrReLU = nullptr;
	MaxErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxErrorOp)
{
	MaxErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new MaxErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMaxErrorOp)
{
	MaxErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxErrorOp");
}

/**
MinErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMinErrorOp)
{
  MinErrorOp<float>* ptrReLU = nullptr;
  MinErrorOp<float>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMinErrorOp)
{
  MinErrorOp<float>* ptrReLU = nullptr;
  ptrReLU = new MinErrorOp<float>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMinErrorOp)
{
  MinErrorOp<float> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "MinErrorOp");
}

/**
MeanErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanErrorOp)
{
	MeanErrorOp<float>* ptrReLU = nullptr;
	MeanErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanErrorOp)
{
	MeanErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new MeanErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMeanErrorOp)
{
	MeanErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanErrorOp");
}

/**
VarModErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarModErrorOp)
{
	VarModErrorOp<float>* ptrReLU = nullptr;
	VarModErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModErrorOp)
{
	VarModErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new VarModErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameVarModErrorOp)
{
	VarModErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModErrorOp");
}

/**
CountErrorOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountErrorOp)
{
	CountErrorOp<float>* ptrReLU = nullptr;
	CountErrorOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountErrorOp)
{
	CountErrorOp<float>* ptrReLU = nullptr;
	ptrReLU = new CountErrorOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameCountErrorOp)
{
	CountErrorOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountErrorOp");
}

/**
SumWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorSumWeightGradOp)
{
	SumWeightGradOp<float>* ptrReLU = nullptr;
	SumWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorSumWeightGradOp)
{
	SumWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new SumWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameSumWeightGradOp)
{
	SumWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "SumWeightGradOp");
}

/**
ProdWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorProdWeightGradOp)
{
	ProdWeightGradOp<float>* ptrReLU = nullptr;
	ProdWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorProdWeightGradOp)
{
	ProdWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new ProdWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameProdWeightGradOp)
{
	ProdWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "ProdWeightGradOp");
}

/**
MaxWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMaxWeightGradOp)
{
	MaxWeightGradOp<float>* ptrReLU = nullptr;
	MaxWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMaxWeightGradOp)
{
	MaxWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new MaxWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMaxWeightGradOp)
{
	MaxWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MaxWeightGradOp");
}

/**
MinWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMinWeightGradOp)
{
  MinWeightGradOp<float>* ptrReLU = nullptr;
  MinWeightGradOp<float>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMinWeightGradOp)
{
  MinWeightGradOp<float>* ptrReLU = nullptr;
  ptrReLU = new MinWeightGradOp<float>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMinWeightGradOp)
{
  MinWeightGradOp<float> operation;

  BOOST_CHECK_EQUAL(operation.getName(), "MinWeightGradOp");
}

/**
MeanWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMeanWeightGradOp)
{
	MeanWeightGradOp<float>* ptrReLU = nullptr;
	MeanWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorMeanWeightGradOp)
{
	MeanWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new MeanWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameMeanWeightGradOp)
{
	MeanWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "MeanWeightGradOp");
}

/**
VarModWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorVarModWeightGradOp)
{
	VarModWeightGradOp<float>* ptrReLU = nullptr;
	VarModWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorVarModWeightGradOp)
{
	VarModWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new VarModWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameVarModWeightGradOp)
{
	VarModWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "VarModWeightGradOp");
}

/**
CountWeightGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorCountWeightGradOp)
{
	CountWeightGradOp<float>* ptrReLU = nullptr;
	CountWeightGradOp<float>* nullPointerReLU = nullptr;
	BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorCountWeightGradOp)
{
	CountWeightGradOp<float>* ptrReLU = nullptr;
	ptrReLU = new CountWeightGradOp<float>();
	delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(getNameCountWeightGradOp)
{
	CountWeightGradOp<float> operation;

	BOOST_CHECK_EQUAL(operation.getName(), "CountWeightGradOp");
}

BOOST_AUTO_TEST_SUITE_END()