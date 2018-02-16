/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ChromatogramSimulator test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/algorithm/ChromatogramSimulator.h>
#include <SmartPeak/algorithm/PeakSimulator.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(chromatogramsimulator)

// class ChromatogramSimulator_test: public ChromatogramSimulator
// {
// public:
// };

BOOST_AUTO_TEST_CASE(constructor) 
{
  ChromatogramSimulator* ptr = nullptr;
  ChromatogramSimulator* nullPointer = nullptr;
	ptr = new ChromatogramSimulator();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ChromatogramSimulator* ptr = nullptr;
	ptr = new ChromatogramSimulator();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(joinPeakWindows) 
{
  ChromatogramSimulator chromsimulator;
  PeakSimulator peak_left, peak_right;

  // Perfect overlap; no differences in baseline
  peak_left = PeakSimulator(1.0, 0.0, 
    0.0, 10.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  peak_right = PeakSimulator(1.0, 0.0, 
    10.0, 20.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  chromsimulator.joinPeakWindows(peak_left, peak_right);
  BOOST_CHECK_EQUAL(peak_left.getBaselineRight(), 1.0);
  BOOST_CHECK_EQUAL(peak_right.getBaselineLeft(), 1.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowEnd(), 10.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowStart(), 10.0);

  // Perfect overlap; no differences in baseline
  // swapped peaks
  peak_right = PeakSimulator(1.0, 0.0, 
    0.0, 10.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  peak_left = PeakSimulator(1.0, 0.0, 
    10.0, 20.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  chromsimulator.joinPeakWindows(peak_left, peak_right);
  BOOST_CHECK_EQUAL(peak_left.getBaselineRight(), 1.0);
  BOOST_CHECK_EQUAL(peak_right.getBaselineLeft(), 1.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowStart(), 0.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowEnd(), 10.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowStart(), 10.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowEnd(), 20.0);

 // Non overlapping windows; Left baseline is higher
  peak_left = PeakSimulator(1.0, 0.0, 
    0.0, 8.0, 
    0.0, 0.0,
    1.0, 5.0, //bl, br
    15);
  peak_right = PeakSimulator(1.0, 0.0, 
    12.0, 20.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  chromsimulator.joinPeakWindows(peak_left, peak_right);
  BOOST_CHECK_EQUAL(peak_left.getBaselineRight(), 5.0);
  BOOST_CHECK_EQUAL(peak_right.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowEnd(), 12.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowStart(), 12.0);

 // Non overlapping windows; Right baseline is higher
  peak_left = PeakSimulator(1.0, 0.0, 
    0.0, 8.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  peak_right = PeakSimulator(1.0, 0.0, 
    12.0, 20.0, 
    0.0, 0.0,
    5.0, 1.0, //bl, br
    15);
  chromsimulator.joinPeakWindows(peak_left, peak_right);
  BOOST_CHECK_EQUAL(peak_left.getBaselineRight(), 5.0);
  BOOST_CHECK_EQUAL(peak_right.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowEnd(), 12.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowStart(), 12.0);

 // Overlapping windows; Left baseline is higher
  peak_left = PeakSimulator(1.0, 0.0, 
    0.0, 12.0, 
    0.0, 0.0,
    1.0, 5.0, //bl, br
    15);
  peak_right = PeakSimulator(1.0, 0.0, 
    8.0, 20.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  chromsimulator.joinPeakWindows(peak_left, peak_right);
  BOOST_CHECK_EQUAL(peak_left.getBaselineRight(), 5.0);
  BOOST_CHECK_EQUAL(peak_right.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowEnd(), 12.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowStart(), 12.0);

 // Overlapping windows; Right baseline is higher
  peak_left = PeakSimulator(1.0, 0.0, 
    0.0, 12.0, 
    0.0, 0.0,
    1.0, 1.0, //bl, br
    15);
  peak_right = PeakSimulator(1.0, 0.0, 
    8.0, 20.0, 
    0.0, 0.0,
    5.0, 1.0, //bl, br
    15);
  chromsimulator.joinPeakWindows(peak_left, peak_right);
  BOOST_CHECK_EQUAL(peak_left.getBaselineRight(), 5.0);
  BOOST_CHECK_EQUAL(peak_right.getBaselineLeft(), 5.0);
  BOOST_CHECK_EQUAL(peak_left.getWindowEnd(), 12.0);
  BOOST_CHECK_EQUAL(peak_right.getWindowStart(), 12.0);
}

BOOST_AUTO_TEST_SUITE_END()