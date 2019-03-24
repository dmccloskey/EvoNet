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

BOOST_AUTO_TEST_CASE(gettersAndSettersEuclideanDistanceOp)
{
  EuclideanDistanceOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "EuclideanDistanceOp");
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

BOOST_AUTO_TEST_CASE(gettersAndSettersEuclideanDistanceGradOp)
{
  EuclideanDistanceGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "EuclideanDistanceGradOp");
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

BOOST_AUTO_TEST_CASE(gettersAndSettersL2NormOp)
{
  L2NormOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "L2NormOp");
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

BOOST_AUTO_TEST_CASE(gettersAndSettersL2NormGradOp)
{
  L2NormGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "L2NormGradOp");
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

BOOST_AUTO_TEST_CASE(gettersAndSettersBCEOp)
{
  BCEOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCEOp");
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

BOOST_AUTO_TEST_CASE(gettersAndSettersBCEGradOp)
{
  BCEGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCEGradOp");
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