/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE LossFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/LossFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(lossFunction)

/**
  ManhattanDistanceLossOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceOp) 
{
  ManhattanDistanceLossOp<double>* ptrReLU = nullptr;
  ManhattanDistanceLossOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceOp) 
{
  ManhattanDistanceLossOp<double>* ptrReLU = nullptr;
	ptrReLU = new ManhattanDistanceLossOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEuclideanDistanceOp)
{
  ManhattanDistanceLossOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "ManhattanDistanceLossOp");
}

/**
  ManhattanDistanceLossGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorEuclideanDistanceGradOp) 
{
  ManhattanDistanceLossGradOp<double>* ptrReLU = nullptr;
  ManhattanDistanceLossGradOp<double>* nullPointerReLU = nullptr;
  BOOST_CHECK_EQUAL(ptrReLU, nullPointerReLU);
}

BOOST_AUTO_TEST_CASE(destructorEuclideanDistanceGradOp) 
{
  ManhattanDistanceLossGradOp<double>* ptrReLU = nullptr;
	ptrReLU = new ManhattanDistanceLossGradOp<double>();
  delete ptrReLU;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersEuclideanDistanceGradOp)
{
  ManhattanDistanceLossGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "ManhattanDistanceLossGradOp");
}

/**
  L2NormLossOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormOp) 
{
  L2NormLossOp<double>* ptrL2Norm = nullptr;
  L2NormLossOp<double>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormOp) 
{
  L2NormLossOp<double>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormLossOp<double>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersL2NormOp)
{
  L2NormLossOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "L2NormLossOp");
}

/**
  L2NormLossGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorL2NormGradOp) 
{
  L2NormLossGradOp<double>* ptrL2Norm = nullptr;
  L2NormLossGradOp<double>* nullPointerL2Norm = nullptr;
  BOOST_CHECK_EQUAL(ptrL2Norm, nullPointerL2Norm);
}

BOOST_AUTO_TEST_CASE(destructorL2NormGradOp) 
{
  L2NormLossGradOp<double>* ptrL2Norm = nullptr;
	ptrL2Norm = new L2NormLossGradOp<double>();
  delete ptrL2Norm;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersL2NormGradOp)
{
  L2NormLossGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "L2NormLossGradOp");
}

/**
  CrossEntropyOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyOp) 
{
  BCELossOp<double>* ptrCrossEntropy = nullptr;
  BCELossOp<double>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyOp) 
{
  BCELossOp<double>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new BCELossOp<double>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersBCEOp)
{
  BCELossOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCELossOp");
}

/**
  CrossEntropyGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorCrossEntropyGradOp) 
{
  BCELossGradOp<double>* ptrCrossEntropy = nullptr;
  BCELossGradOp<double>* nullPointerCrossEntropy = nullptr;
  BOOST_CHECK_EQUAL(ptrCrossEntropy, nullPointerCrossEntropy);
}

BOOST_AUTO_TEST_CASE(destructorCrossEntropyGradOp) 
{
  BCELossGradOp<double>* ptrCrossEntropy = nullptr;
	ptrCrossEntropy = new BCELossGradOp<double>();
  delete ptrCrossEntropy;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersBCEGradOp)
{
  BCELossGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCELossGradOp");
}

/**
  NegativeLogLikelihoodLossOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodLossOp<double>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodLossOp<double>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodOp) 
{
  NegativeLogLikelihoodLossOp<double>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodLossOp<double>();
  delete ptrNegativeLogLikelihood;
}

/**
  NegativeLogLikelihoodLossGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodLossGradOp<double>* ptrNegativeLogLikelihood = nullptr;
  NegativeLogLikelihoodLossGradOp<double>* nullPointerNegativeLogLikelihood = nullptr;
  BOOST_CHECK_EQUAL(ptrNegativeLogLikelihood, nullPointerNegativeLogLikelihood);
}

BOOST_AUTO_TEST_CASE(destructorNegativeLogLikelihoodGradOp) 
{
  NegativeLogLikelihoodLossGradOp<double>* ptrNegativeLogLikelihood = nullptr;
	ptrNegativeLogLikelihood = new NegativeLogLikelihoodLossGradOp<double>();
  delete ptrNegativeLogLikelihood;
}

/**
  MSELossOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEOp) 
{
  MSELossOp<double>* ptrMSE = nullptr;
  MSELossOp<double>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEOp) 
{
  MSELossOp<double>* ptrMSE = nullptr;
	ptrMSE = new MSELossOp<double>();
  delete ptrMSE;
}

/**
  MSELossGradOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorMSEGradOp) 
{
  MSELossGradOp<double>* ptrMSE = nullptr;
  MSELossGradOp<double>* nullPointerMSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMSE, nullPointerMSE);
}

BOOST_AUTO_TEST_CASE(destructorMSEGradOp) 
{
  MSELossGradOp<double>* ptrMSE = nullptr;
	ptrMSE = new MSELossGradOp<double>();
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

/**
  KLDivergenceCatOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceCatOp)
{
  KLDivergenceCatOp<double>* ptrKLDivergenceCat = nullptr;
  KLDivergenceCatOp<double>* nullPointerKLDivergenceCat = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceCat, nullPointerKLDivergenceCat);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceCatOp)
{
  KLDivergenceCatOp<double>* ptrKLDivergenceCat = nullptr;
  ptrKLDivergenceCat = new KLDivergenceCatOp<double>();
  delete ptrKLDivergenceCat;
}

/**
  KLDivergenceCatGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceCatGradOp)
{
  KLDivergenceCatGradOp<double>* ptrKLDivergenceCat = nullptr;
  KLDivergenceCatGradOp<double>* nullPointerKLDivergenceCat = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceCat, nullPointerKLDivergenceCat);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceCatGradOp)
{
  KLDivergenceCatGradOp<double>* ptrKLDivergenceCat = nullptr;
  ptrKLDivergenceCat = new KLDivergenceCatGradOp<double>();
  delete ptrKLDivergenceCat;
}

BOOST_AUTO_TEST_SUITE_END()