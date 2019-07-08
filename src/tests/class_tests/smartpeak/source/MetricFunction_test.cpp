/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MetricFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/MetricFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(MetricFunction1)

/**
  AccuracyBCOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorAccuracyBCOp) 
{
  AccuracyBCOp<double>* ptrMetFunc = nullptr;
  AccuracyBCOp<double>* nullPointerMetFunc = nullptr;
  BOOST_CHECK_EQUAL(ptrMetFunc, nullPointerMetFunc);
}

BOOST_AUTO_TEST_CASE(destructorAccuracyBCOp) 
{
  AccuracyBCOp<double>* ptrMetFunc = nullptr;
	ptrMetFunc = new AccuracyBCOp<double>();
  delete ptrMetFunc;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersAccuracyBCOp)
{
  AccuracyBCOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "AccuracyBCOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  AccuracyBCOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  AccuracyMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorAccuracyMCMicroOp)
{
  AccuracyMCMicroOp<double>* ptrMetFunc = nullptr;
  AccuracyMCMicroOp<double>* nullPointerMetFunc = nullptr;
  BOOST_CHECK_EQUAL(ptrMetFunc, nullPointerMetFunc);
}

BOOST_AUTO_TEST_CASE(destructorAccuracyMCMicroOp)
{
  AccuracyMCMicroOp<double>* ptrMetFunc = nullptr;
  ptrMetFunc = new AccuracyMCMicroOp<double>();
  delete ptrMetFunc;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersAccuracyMCMicroOp)
{
  AccuracyMCMicroOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "AccuracyMCMicroOp");
}

/**
  AccuracyMCMacroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorAccuracyMCMacroOp)
{
  AccuracyMCMacroOp<double>* ptrMetFunc = nullptr;
  AccuracyMCMacroOp<double>* nullPointerMetFunc = nullptr;
  BOOST_CHECK_EQUAL(ptrMetFunc, nullPointerMetFunc);
}

BOOST_AUTO_TEST_CASE(destructorAccuracyMCMacroOp)
{
  AccuracyMCMacroOp<double>* ptrMetFunc = nullptr;
  ptrMetFunc = new AccuracyMCMacroOp<double>();
  delete ptrMetFunc;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersAccuracyMCMacroOp)
{
  AccuracyMCMacroOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "AccuracyMCMacroOp");
}

/**
  PredictionBiasOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorPredictionBiasOp) 
{
  PredictionBiasOp<double>* ptrPredictionBias = nullptr;
  PredictionBiasOp<double>* nullPointerPredictionBias = nullptr;
  BOOST_CHECK_EQUAL(ptrPredictionBias, nullPointerPredictionBias);
}

BOOST_AUTO_TEST_CASE(destructorPredictionBiasOp) 
{
  PredictionBiasOp<double>* ptrPredictionBias = nullptr;
	ptrPredictionBias = new PredictionBiasOp<double>();
  delete ptrPredictionBias;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersPredictionBiasOp)
{
  PredictionBiasOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "PredictionBiasOp");
  BOOST_CHECK_EQUAL(operation.getParameters().size(), 0);
}

/**
  F1ScoreBCOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorF1ScoreBCOp) 
{
  F1ScoreBCOp<double>* ptrF1Score = nullptr;
  F1ScoreBCOp<double>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreBCOp) 
{
  F1ScoreBCOp<double>* ptrF1Score = nullptr;
	ptrF1Score = new F1ScoreBCOp<double>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersF1ScoreBCOp)
{
  F1ScoreBCOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "F1ScoreBCOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  F1ScoreBCOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  F1ScoreMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorF1ScoreMCMicroOp)
{
  F1ScoreMCMicroOp<double>* ptrF1Score = nullptr;
  F1ScoreMCMicroOp<double>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreMCMicroOp)
{
  F1ScoreMCMicroOp<double>* ptrF1Score = nullptr;
  ptrF1Score = new F1ScoreMCMicroOp<double>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersF1ScoreMCMicroOp)
{
  F1ScoreMCMicroOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "F1ScoreMCMicroOp");
}

/**
  F1ScoreMCMacroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorF1ScoreMCMacroOp)
{
  F1ScoreMCMacroOp<double>* ptrF1Score = nullptr;
  F1ScoreMCMacroOp<double>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreMCMacroOp)
{
  F1ScoreMCMacroOp<double>* ptrF1Score = nullptr;
  ptrF1Score = new F1ScoreMCMacroOp<double>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersF1ScoreMCMacroOp)
{
  F1ScoreMCMacroOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "F1ScoreMCMacroOp");
}

/**
  AUROCOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorAUROCOp) 
{
  AUROCOp<double>* ptrAUROC = nullptr;
  AUROCOp<double>* nullPointerAUROC = nullptr;
  BOOST_CHECK_EQUAL(ptrAUROC, nullPointerAUROC);
}

BOOST_AUTO_TEST_CASE(destructorAUROCOp) 
{
  AUROCOp<double>* ptrAUROC = nullptr;
	ptrAUROC = new AUROCOp<double>();
  delete ptrAUROC;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersAUROCOp)
{
  AUROCOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "AUROCOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  AUROCOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  MCCBCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMCCBCOp)
{
  MCCBCOp<double>* ptrMCC = nullptr;
  MCCBCOp<double>* nullPointerMCC = nullptr;
  BOOST_CHECK_EQUAL(ptrMCC, nullPointerMCC);
}

BOOST_AUTO_TEST_CASE(destructorMCCBCOp)
{
  MCCBCOp<double>* ptrMCC = nullptr;
  ptrMCC = new MCCBCOp<double>();
  delete ptrMCC;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMCCBCOp)
{
  MCCBCOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MCCBCOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  MCCBCOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  MCCMCMicroOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMCCMCMicroOp)
{
  MCCMCMicroOp<double>* ptrMCC = nullptr;
  MCCMCMicroOp<double>* nullPointerMCC = nullptr;
  BOOST_CHECK_EQUAL(ptrMCC, nullPointerMCC);
}

BOOST_AUTO_TEST_CASE(destructorMCCMCMicroOp)
{
  MCCMCMicroOp<double>* ptrMCC = nullptr;
  ptrMCC = new MCCMCMicroOp<double>();
  delete ptrMCC;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMCCMCMicroOp)
{
  MCCMCMicroOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MCCMCMicroOp");
}

/**
  MAEOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMAEOp)
{
  MAEOp<double>* ptrMAE = nullptr;
  MAEOp<double>* nullPointerMAE = nullptr;
  BOOST_CHECK_EQUAL(ptrMAE, nullPointerMAE);
}

BOOST_AUTO_TEST_CASE(destructorMAEOp)
{
  MAEOp<double>* ptrMAE = nullptr;
  ptrMAE = new MAEOp<double>();
  delete ptrMAE;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMAEOp)
{
  MAEOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MAEOp");
}

BOOST_AUTO_TEST_SUITE_END()