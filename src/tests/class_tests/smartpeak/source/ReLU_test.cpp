/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ReLU test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/ReLU.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(relu)

template<typename Scalar>
class add{
  public:
    add(const Scalar& a) : a_(a){}
    const Scalar operator()(const Scalar& x) const { return x + a_; }
  private:
    Scalar a_;
};

BOOST_AUTO_TEST_CASE(constructor) 
{
  ReLU* ptr = nullptr;
  ReLU* nullPointer = nullptr;
	ptr = new ReLU();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  ReLU* ptr = nullptr;
	ptr = new ReLU();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(fx) 
{
  ReLU relu;

  BOOST_CHECK_CLOSE(relu.fx(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.fx(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.fx(10.0), 10.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.fx(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.fx(-10.0), 0.0, 1e-6);

  Eigen::Tensor<double, 2> a(2.0, 2.0);
  a.setConstant(1.0f);
  // Eigen::Tensor<double, 2> b = a.unaryExpr(&dub);
  Eigen::Tensor<double, 2> b = a.unaryExpr(add<double>(2));
  std::cout << a << std::endl;
  std::cout << b << std::endl;
}

BOOST_AUTO_TEST_CASE(dfx) 
{
  ReLU relu;

  BOOST_CHECK_CLOSE(relu.dfx(0.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.dfx(1.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.dfx(10.0), 1.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.dfx(-1.0), 0.0, 1e-6);
  BOOST_CHECK_CLOSE(relu.dfx(-10.0), 0.0, 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()