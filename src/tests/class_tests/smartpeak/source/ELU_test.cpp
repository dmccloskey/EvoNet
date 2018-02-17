/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ELU test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/ELU.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(elu)

BOOST_AUTO_TEST_CASE(constructor) 
{
  ELU* ptr = nullptr;
  ELU* nullPointer = nullptr;
	ptr = new ELU();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ELU* ptr = nullptr;
	ptr = new ELU();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  ELU elu(1.0);

  BOOST_CHECK_EQUAL(elu.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  ELU elu;
  elu.setAlpha(1.0);

  BOOST_CHECK_EQUAL(elu.getAlpha(), 1.0);
}

BOOST_AUTO_TEST_CASE(fx) 
{
  ELU elu;

  BOOST_CHECK_CLOSE(elu.fx(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(elu.fx(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(elu.fx(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(elu.fx(-1.0), -0.63212055882855767, 1e-6);
  BOOST_CHECK_CLOSE(elu.fx(-10.0), -0.99995460007023751, 1e-6);
}

BOOST_AUTO_TEST_CASE(dfx) 
{
  ELU elu;

  BOOST_CHECK_CLOSE(elu.dfx(0.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(elu.dfx(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(elu.dfx(10.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(elu.dfx(-1.0), 0.36787944117144233, 1e-6);
  BOOST_CHECK_CLOSE(elu.dfx(-10.0), 4.5399929762490743e-05, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()