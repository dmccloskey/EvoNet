/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE MetricFunction test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/ml/MetricFunction.h>

#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(MetricFunction1)

/**
  ClassificationAccuracyOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorClassificationAccuracyOp) 
{
  ClassificationAccuracyOp<double>* ptrMetFunc = nullptr;
  ClassificationAccuracyOp<double>* nullPointerMetFunc = nullptr;
  BOOST_CHECK_EQUAL(ptrMetFunc, nullPointerMetFunc);
}

BOOST_AUTO_TEST_CASE(destructorClassificationAccuracyOp) 
{
  ClassificationAccuracyOp<double>* ptrMetFunc = nullptr;
	ptrMetFunc = new ClassificationAccuracyOp<double>();
  delete ptrMetFunc;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersClassificationAccuracyOp)
{
  ClassificationAccuracyOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "ClassificationAccuracyOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  ClassificationAccuracyOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  BCAccuracyOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCAccuracyOp)
{
  BCAccuracyOp<double>* ptrMetFunc = nullptr;
  BCAccuracyOp<double>* nullPointerMetFunc = nullptr;
  BOOST_CHECK_EQUAL(ptrMetFunc, nullPointerMetFunc);
}

BOOST_AUTO_TEST_CASE(destructorBCAccuracyOp)
{
  BCAccuracyOp<double>* ptrMetFunc = nullptr;
  ptrMetFunc = new BCAccuracyOp<double>();
  delete ptrMetFunc;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersBCAccuracyOp)
{
  BCAccuracyOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCAccuracyOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  BCAccuracyOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
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
  F1ScoreOp Tests
*/ 
BOOST_AUTO_TEST_CASE(constructorF1ScoreOp) 
{
  F1ScoreOp<double>* ptrF1Score = nullptr;
  F1ScoreOp<double>* nullPointerF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrF1Score, nullPointerF1Score);
}

BOOST_AUTO_TEST_CASE(destructorF1ScoreOp) 
{
  F1ScoreOp<double>* ptrF1Score = nullptr;
	ptrF1Score = new F1ScoreOp<double>();
  delete ptrF1Score;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersF1ScoreOp)
{
  F1ScoreOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "F1ScoreOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  F1ScoreOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  BCF1ScoreOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCF1ScoreOp)
{
  BCF1ScoreOp<double>* ptrBCF1Score = nullptr;
  BCF1ScoreOp<double>* nullPointerBCF1Score = nullptr;
  BOOST_CHECK_EQUAL(ptrBCF1Score, nullPointerBCF1Score);
}

BOOST_AUTO_TEST_CASE(destructorBCF1ScoreOp)
{
  BCF1ScoreOp<double>* ptrBCF1Score = nullptr;
  ptrBCF1Score = new BCF1ScoreOp<double>();
  delete ptrBCF1Score;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersBCF1ScoreOp)
{
  BCF1ScoreOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCF1ScoreOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  BCF1ScoreOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
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
  BCAUROCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorBCAUROCOp)
{
  BCAUROCOp<double>* ptrBCAUROC = nullptr;
  BCAUROCOp<double>* nullPointerBCAUROC = nullptr;
  BOOST_CHECK_EQUAL(ptrBCAUROC, nullPointerBCAUROC);
}

BOOST_AUTO_TEST_CASE(destructorBCAUROCOp)
{
  BCAUROCOp<double>* ptrBCAUROC = nullptr;
  ptrBCAUROC = new BCAUROCOp<double>();
  delete ptrBCAUROC;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersBCAUROCOp)
{
  BCAUROCOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "BCAUROCOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  BCAUROCOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
}

/**
  MCCOp Tests
*/
BOOST_AUTO_TEST_CASE(constructorMCCOp)
{
  MCCOp<double>* ptrMCC = nullptr;
  MCCOp<double>* nullPointerMCC = nullptr;
  BOOST_CHECK_EQUAL(ptrMCC, nullPointerMCC);
}

BOOST_AUTO_TEST_CASE(destructorMCCOp)
{
  MCCOp<double>* ptrMCC = nullptr;
  ptrMCC = new MCCOp<double>();
  delete ptrMCC;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersMCCOp)
{
  MCCOp<float> operation;
  BOOST_CHECK_EQUAL(operation.getName(), "MCCOp");
  BOOST_CHECK_EQUAL(operation.getParameters().at(0), 0.5);
  BOOST_CHECK_CLOSE(operation.getClassificationThreshold(), 0.5, 1e-3);

  MCCOp<float> operation2(0.1);
  BOOST_CHECK_CLOSE(operation2.getClassificationThreshold(), 0.1, 1e-3);
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