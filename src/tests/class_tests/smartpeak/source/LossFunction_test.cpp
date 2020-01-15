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
  MAELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAEOp)
{
  MAELossOp<double>* ptrMAE = nullptr;
  MAELossOp<double>* nullPointerMAE = nullptr;
  BOOST_CHECK_EQUAL(ptrMAE, nullPointerMAE);
}

BOOST_AUTO_TEST_CASE(destructorMAEOp)
{
  MAELossOp<double>* ptrMAE = nullptr;
  ptrMAE = new MAELossOp<double>();
  delete ptrMAE;
}

/**
  MAELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAEGradOp)
{
  MAELossGradOp<double>* ptrMAE = nullptr;
  MAELossGradOp<double>* nullPointerMAE = nullptr;
  BOOST_CHECK_EQUAL(ptrMAE, nullPointerMAE);
}

BOOST_AUTO_TEST_CASE(destructorMAEGradOp)
{
  MAELossGradOp<double>* ptrMAE = nullptr;
  ptrMAE = new MAELossGradOp<double>();
  delete ptrMAE;
}

/**
  MRSELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMRSEOp)
{
  MRSELossOp<double>* ptrMRSE = nullptr;
  MRSELossOp<double>* nullPointerMRSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMRSE, nullPointerMRSE);
}

BOOST_AUTO_TEST_CASE(destructorMRSEOp)
{
  MRSELossOp<double>* ptrMRSE = nullptr;
  ptrMRSE = new MRSELossOp<double>();
  delete ptrMRSE;
}

/**
  MRSELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMRSEGradOp)
{
  MRSELossGradOp<double>* ptrMRSE = nullptr;
  MRSELossGradOp<double>* nullPointerMRSE = nullptr;
  BOOST_CHECK_EQUAL(ptrMRSE, nullPointerMRSE);
}

BOOST_AUTO_TEST_CASE(destructorMRSEGradOp)
{
  MRSELossGradOp<double>* ptrMRSE = nullptr;
  ptrMRSE = new MRSELossGradOp<double>();
  delete ptrMRSE;
}

/**
  MLELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMLEOp)
{
  MLELossOp<double>* ptrMLE = nullptr;
  MLELossOp<double>* nullPointerMLE = nullptr;
  BOOST_CHECK_EQUAL(ptrMLE, nullPointerMLE);
}

BOOST_AUTO_TEST_CASE(destructorMLEOp)
{
  MLELossOp<double>* ptrMLE = nullptr;
  ptrMLE = new MLELossOp<double>();
  delete ptrMLE;
}

/**
  MLELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMLEGradOp)
{
  MLELossGradOp<double>* ptrMLE = nullptr;
  MLELossGradOp<double>* nullPointerMLE = nullptr;
  BOOST_CHECK_EQUAL(ptrMLE, nullPointerMLE);
}

BOOST_AUTO_TEST_CASE(destructorMLEGradOp)
{
  MLELossGradOp<double>* ptrMLE = nullptr;
  ptrMLE = new MLELossGradOp<double>();
  delete ptrMLE;
}

/**
	KLDivergenceMuLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuOp)
{
	KLDivergenceMuLossOp<double>* ptrKLDivergenceMu = nullptr;
	KLDivergenceMuLossOp<double>* nullPointerKLDivergenceMu = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuOp)
{
	KLDivergenceMuLossOp<double>* ptrKLDivergenceMu = nullptr;
	ptrKLDivergenceMu = new KLDivergenceMuLossOp<double>();
	delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersKLDivergenceMuOp)
{
	KLDivergenceMuLossOp<float> operation(1e-3, 1, 5);
	BOOST_CHECK_EQUAL(operation.getName(), "KLDivergenceMuLossOp");
	BOOST_CHECK_EQUAL(operation.getParameters().size(), 3);
	BOOST_CHECK_CLOSE(operation.getParameters().at(0), 1e-3, 1e-4);
	BOOST_CHECK_EQUAL(operation.getParameters().at(1), 1);
	BOOST_CHECK_EQUAL(operation.getParameters().at(2), 5);
}

/**
	KLDivergenceMuLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceMuGradOp)
{
	KLDivergenceMuLossGradOp<double>* ptrKLDivergenceMu = nullptr;
	KLDivergenceMuLossGradOp<double>* nullPointerKLDivergenceMu = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceMu, nullPointerKLDivergenceMu);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceMuGradOp)
{
	KLDivergenceMuLossGradOp<double>* ptrKLDivergenceMu = nullptr;
	ptrKLDivergenceMu = new KLDivergenceMuLossGradOp<double>();
	delete ptrKLDivergenceMu;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersKLDivergenceGradMuOp)
{
	KLDivergenceMuLossGradOp<float> operation(1e-3, 1, 5);
	BOOST_CHECK_EQUAL(operation.getName(), "KLDivergenceMuLossGradOp");
	BOOST_CHECK_EQUAL(operation.getParameters().size(), 3);
	BOOST_CHECK_CLOSE(operation.getParameters().at(0), 1e-3, 1e-4);
	BOOST_CHECK_EQUAL(operation.getParameters().at(1), 1);
	BOOST_CHECK_EQUAL(operation.getParameters().at(2), 5);
}

/**
	KLDivergenceLogVarLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarOp)
{
	KLDivergenceLogVarLossOp<double>* ptrKLDivergenceLogVar = nullptr;
	KLDivergenceLogVarLossOp<double>* nullPointerKLDivergenceLogVar = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarOp)
{
	KLDivergenceLogVarLossOp<double>* ptrKLDivergenceLogVar = nullptr;
	ptrKLDivergenceLogVar = new KLDivergenceLogVarLossOp<double>();
	delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersKLDivergenceLogVarOp)
{
	KLDivergenceLogVarLossOp<float> operation(1e-3, 1, 5);
	BOOST_CHECK_EQUAL(operation.getName(), "KLDivergenceLogVarLossOp");
	BOOST_CHECK_EQUAL(operation.getParameters().size(), 3);
	BOOST_CHECK_CLOSE(operation.getParameters().at(0), 1e-3, 1e-4);
	BOOST_CHECK_EQUAL(operation.getParameters().at(1), 1);
	BOOST_CHECK_EQUAL(operation.getParameters().at(2), 5);
}

/**
	KLDivergenceLogVarLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarLossGradOp<double>* ptrKLDivergenceLogVar = nullptr;
	KLDivergenceLogVarLossGradOp<double>* nullPointerKLDivergenceLogVar = nullptr;
	BOOST_CHECK_EQUAL(ptrKLDivergenceLogVar, nullPointerKLDivergenceLogVar);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceLogVarGradOp)
{
	KLDivergenceLogVarLossGradOp<double>* ptrKLDivergenceLogVar = nullptr;
	ptrKLDivergenceLogVar = new KLDivergenceLogVarLossGradOp<double>();
	delete ptrKLDivergenceLogVar;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersKLDivergenceLogVarLossGradOp)
{
	KLDivergenceLogVarLossGradOp<float> operation(1e-3, 1, 5);
	BOOST_CHECK_EQUAL(operation.getName(), "KLDivergenceLogVarLossGradOp");
	BOOST_CHECK_EQUAL(operation.getParameters().size(), 3);
	BOOST_CHECK_CLOSE(operation.getParameters().at(0), 1e-3, 1e-4);
	BOOST_CHECK_EQUAL(operation.getParameters().at(1), 1);
	BOOST_CHECK_EQUAL(operation.getParameters().at(2), 5);
}

/**
BCEWithLogitsLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsOp)
{
	BCEWithLogitsLossOp<double>* ptrBCEWithLogits = nullptr;
	BCEWithLogitsLossOp<double>* nullPointerBCEWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsOp)
{
	BCEWithLogitsLossOp<double>* ptrBCEWithLogits = nullptr;
	ptrBCEWithLogits = new BCEWithLogitsLossOp<double>();
	delete ptrBCEWithLogits;
}

/**
BCEWithLogitsLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCEWithLogitsGradOp)
{
	BCEWithLogitsLossGradOp<double>* ptrBCEWithLogits = nullptr;
	BCEWithLogitsLossGradOp<double>* nullPointerBCEWithLogits = nullptr;
	BOOST_CHECK_EQUAL(ptrBCEWithLogits, nullPointerBCEWithLogits);
}

BOOST_AUTO_TEST_CASE(destructorBCEWithLogitsGradOp)
{
	BCEWithLogitsLossGradOp<double>* ptrBCEWithLogits = nullptr;
	ptrBCEWithLogits = new BCEWithLogitsLossGradOp<double>();
	delete ptrBCEWithLogits;
}

/**
  MSERangeUBLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBOp)
{
  MSERangeUBLossOp<double>* ptrMSERangeUB = nullptr;
  MSERangeUBLossOp<double>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBOp)
{
  MSERangeUBLossOp<double>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBLossOp<double>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeUBOp)
{
  MSERangeUBLossOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeUBLossOp");
}

/**
  MSERangeUBLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeUBGradOp)
{
  MSERangeUBLossGradOp<double>* ptrMSERangeUB = nullptr;
  MSERangeUBLossGradOp<double>* nullPointerMSERangeUB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeUB, nullPointerMSERangeUB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeUBGradOp)
{
  MSERangeUBLossGradOp<double>* ptrMSERangeUB = nullptr;
  ptrMSERangeUB = new MSERangeUBLossGradOp<double>();
  delete ptrMSERangeUB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeUBGradOp)
{
  MSERangeUBLossGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeUBLossGradOp");
}

/**
  MSERangeLBLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBOp)
{
  MSERangeLBLossOp<double>* ptrMSERangeLB = nullptr;
  MSERangeLBLossOp<double>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBOp)
{
  MSERangeLBLossOp<double>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBLossOp<double>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeLBOp)
{
  MSERangeLBLossOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeLBLossOp");
}

/**
  MSERangeLBLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMSERangeLBGradOp)
{
  MSERangeLBLossGradOp<double>* ptrMSERangeLB = nullptr;
  MSERangeLBLossGradOp<double>* nullPointerMSERangeLB = nullptr;
  BOOST_CHECK_EQUAL(ptrMSERangeLB, nullPointerMSERangeLB);
}

BOOST_AUTO_TEST_CASE(destructorMSERangeLBGradOp)
{
  MSERangeLBLossGradOp<double>* ptrMSERangeLB = nullptr;
  ptrMSERangeLB = new MSERangeLBLossGradOp<double>();
  delete ptrMSERangeLB;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMSERangeLBGradOp)
{
  MSERangeLBLossGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MSERangeLBLossGradOp");
}

/**
  KLDivergenceCatLossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceCatOp)
{
  KLDivergenceCatLossOp<double>* ptrKLDivergenceCat = nullptr;
  KLDivergenceCatLossOp<double>* nullPointerKLDivergenceCat = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceCat, nullPointerKLDivergenceCat);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceCatOp)
{
  KLDivergenceCatLossOp<double>* ptrKLDivergenceCat = nullptr;
  ptrKLDivergenceCat = new KLDivergenceCatLossOp<double>();
  delete ptrKLDivergenceCat;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersKLDivergenceCatOp)
{
  KLDivergenceCatLossOp<float> operation(1e-3, 1, 5);
  BOOST_CHECK_EQUAL(operation.getName(), "KLDivergenceCatLossOp");
	BOOST_CHECK_EQUAL(operation.getParameters().size(), 3);
	BOOST_CHECK_CLOSE(operation.getParameters().at(0), 1e-3, 1e-4);
	BOOST_CHECK_EQUAL(operation.getParameters().at(1), 1);
	BOOST_CHECK_EQUAL(operation.getParameters().at(2), 5);
}

/**
  KLDivergenceCatLossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorKLDivergenceCatGradOp)
{
  KLDivergenceCatLossGradOp<double>* ptrKLDivergenceCat = nullptr;
  KLDivergenceCatLossGradOp<double>* nullPointerKLDivergenceCat = nullptr;
  BOOST_CHECK_EQUAL(ptrKLDivergenceCat, nullPointerKLDivergenceCat);
}

BOOST_AUTO_TEST_CASE(destructorKLDivergenceCatGradOp)
{
  KLDivergenceCatLossGradOp<double>* ptrKLDivergenceCat = nullptr;
  ptrKLDivergenceCat = new KLDivergenceCatLossGradOp<double>();
  delete ptrKLDivergenceCat;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersKLDivergenceCatGradOp)
{
  KLDivergenceCatLossGradOp<float> operation(1e-3, 1, 5);
  BOOST_CHECK_EQUAL(operation.getName(), "KLDivergenceCatLossGradOp");
	BOOST_CHECK_EQUAL(operation.getParameters().size(), 3);
	BOOST_CHECK_CLOSE(operation.getParameters().at(0), 1e-3, 1e-4);
	BOOST_CHECK_EQUAL(operation.getParameters().at(1), 1);
	BOOST_CHECK_EQUAL(operation.getParameters().at(2), 5);
}

/**
  MAPELossOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAPELossOp)
{
  MAPELossOp<double>* ptrMAPELoss = nullptr;
  MAPELossOp<double>* nullPointerMAPELoss = nullptr;
  BOOST_CHECK_EQUAL(ptrMAPELoss, nullPointerMAPELoss);
}

BOOST_AUTO_TEST_CASE(destructorMAPELossOp)
{
  MAPELossOp<double>* ptrMAPELoss = nullptr;
  ptrMAPELoss = new MAPELossOp<double>();
  delete ptrMAPELoss;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMAPELossOp)
{
  MAPELossOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MAPELossOp");
}

/**
  MAPELossGradOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAPELossGradOp)
{
  MAPELossGradOp<double>* ptrMAPELoss = nullptr;
  MAPELossGradOp<double>* nullPointerMAPELoss = nullptr;
  BOOST_CHECK_EQUAL(ptrMAPELoss, nullPointerMAPELoss);
}

BOOST_AUTO_TEST_CASE(destructorMAPELossGradOp)
{
  MAPELossGradOp<double>* ptrMAPELoss = nullptr;
  ptrMAPELoss = new MAPELossGradOp<double>();
  delete ptrMAPELoss;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMAPELossGradOp)
{
  MAPELossGradOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MAPELossGradOp");
}

BOOST_AUTO_TEST_SUITE_END()