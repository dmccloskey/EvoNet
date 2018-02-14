/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PeakSimulator test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/algorithm/PeakSimulator.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(peaksimulator)

// class PeakSimulator_test: public PeakSimulator
// {
// public:
// };

BOOST_AUTO_TEST_CASE(constructor) 
{
  PeakSimulator* ptr = nullptr;
  PeakSimulator* nullPointer = nullptr;
	ptr = new PeakSimulator();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(constructor2) 
{  
  PeakSimulator psim(500.0, 1.0, 0.0, 10.0, 2.0, 1.0, 5.0, 10.0, 1e6);

  BOOST_CHECK_EQUAL(psim.getStepSizeMu(), 500.0);
  BOOST_CHECK_EQUAL(psim.getStepSizeSigma(), 1.0);
  BOOST_CHECK_EQUAL(psim.getWindowStart(), 0.0);
  BOOST_CHECK_EQUAL(psim.getWindowEnd(), 10.0);
  BOOST_CHECK_EQUAL(psim.getNoiseMu(), 2.0);
  BOOST_CHECK_EQUAL(psim.getNoiseSigma(), 1.0);
  BOOST_CHECK_EQUAL(psim.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(psim.getBaselineRight(), 10.0);
  BOOST_CHECK_EQUAL(psim.getSaturationLimit(), 1e6);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  PeakSimulator* ptr = nullptr;
	ptr = new PeakSimulator();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  PeakSimulator psim;
  psim.setStepSizeMu(500.0);
  psim.setStepSizeSigma(1.0);
  psim.setWindowStart(0.0);
  psim.setWindowEnd(10.0);
  psim.setNoiseMu(2.0);
  psim.setNoiseSimga(1.0);
  psim.setBaselineLeft(5.0);
  psim.setBaselineRight(10.0);
  psim.setSaturationLimit(1e6);

  BOOST_CHECK_EQUAL(psim.getStepSizeMu(), 500.0);
  BOOST_CHECK_EQUAL(psim.getStepSizeSigma(), 1.0);
  BOOST_CHECK_EQUAL(psim.getWindowStart(), 0.0);
  BOOST_CHECK_EQUAL(psim.getWindowEnd(), 10.0);
  BOOST_CHECK_EQUAL(psim.getNoiseMu(), 2.0);
  BOOST_CHECK_EQUAL(psim.getNoiseSigma(), 1.0);
  BOOST_CHECK_EQUAL(psim.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(psim.getBaselineRight(), 10.0);
  BOOST_CHECK_EQUAL(psim.getSaturationLimit(), 1e6);
}

BOOST_AUTO_TEST_CASE(generateRangeWithNoise) 
{ 
  PeakSimulator psim;

  // no noise
  std::vector<double> range = psim.generateRangeWithNoise(0.0, 1.0, 0.0, 10.0);
  BOOST_CHECK_EQUAL(range.size(), 11);
  BOOST_CHECK_EQUAL(range[0], 0.0);
  BOOST_CHECK_EQUAL(range[10], 10.0);

  // with noise
  range = psim.generateRangeWithNoise(0.0, 1.0, 0.1, 10.0);
  BOOST_CHECK_EQUAL(range[0], 0.0);
  BOOST_CHECK_NE(range[10], 10.0);
}

BOOST_AUTO_TEST_CASE(addNoise) 
{ 
  PeakSimulator psim;

  // no noise
  std::vector<double> range = {0, 1, 2, 3, 4, 5};
  std::vector<double> noise_range = psim.addNoise(range, 0.0, 0.0);
  for (int i=0; i<range.size(); ++i)
  {
    BOOST_CHECK_EQUAL(range[i], noise_range[i]);
  }

  // with noise
  noise_range = psim.addNoise(range, 0.0, 1.0);
  for (int i=0; i<range.size(); ++i)
  {
    BOOST_CHECK_NE(range[i], noise_range[i]);
  }

  // with noise
  noise_range = psim.addNoise(range, 1.0, 0.0);
  for (int i=0; i<range.size(); ++i)
  {
    BOOST_CHECK_EQUAL(range[i] + 1.0, noise_range[i]);
  }
}

BOOST_AUTO_TEST_CASE(addBaseline) 
{ 
  PeakSimulator psim;

  // toy peak
  std::vector<double> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> y = {0, 0, 1, 3, 7, 10, 7, 3, 1, 0, 0};
  
  // no baseline
  std::vector<double> y_baseline = psim.addBaseline(x, y, 
    3, 1, 5);

  // no noise
  std::vector<double> range = psim.generateRangeWithNoise(0.0, 1.0, 0.0, 10.0);
  BOOST_CHECK_EQUAL(range.size(), 11);
  BOOST_CHECK_EQUAL(range[0], 0.0);
  BOOST_CHECK_EQUAL(range[10], 10.0);

  // with noise
  range = psim.generateRangeWithNoise(0.0, 1.0, 0.1, 10.0);
  BOOST_CHECK_EQUAL(range[0], 0.0);
  BOOST_CHECK_NE(range[10], 10.0);
}

BOOST_AUTO_TEST_SUITE_END()