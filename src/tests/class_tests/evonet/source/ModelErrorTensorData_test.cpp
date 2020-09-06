/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE ModelErrorData test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/ml/ModelErrorData.h>

#include <iostream>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(ModelErrorData1)

BOOST_AUTO_TEST_CASE(constructor) 
{
	ModelErrorDataCpu<float>* ptr = nullptr;
	ModelErrorDataCpu<float>* nullPointer = nullptr;
	ptr = new ModelErrorDataCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	ModelErrorDataCpu<float>* ptr = nullptr;
	ptr = new ModelErrorDataCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
	ModelErrorDataCpu<float> error, error_test;
	BOOST_CHECK(error == error_test);
}

#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	ModelErrorDataGpu<float> error;
	error.setBatchSize(2);
	error.setMemorySize(3);
	error.setNMetrics(4);

  Eigen::Tensor<float, 2> error_tensor(2, 3), metric(4, 3);
  error_tensor.setConstant(3); metric.setConstant(4);

	error.setError(error_tensor);
	error.setMetric(metric);

	BOOST_CHECK_EQUAL(error.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(error.getMemorySize(), 3);
	BOOST_CHECK_EQUAL(error.getNMetrics(), 4);
	BOOST_CHECK_EQUAL(error.getError()(0, 0), 3);
	BOOST_CHECK(error.getErrorStatus().first);
	BOOST_CHECK(!error.getErrorStatus().second);
	BOOST_CHECK_EQUAL(error.getMetric()(0, 0), 4);
	BOOST_CHECK(error.getMetricStatus().first);
	BOOST_CHECK(!error.getMetricStatus().second);

	// Test mutability
	error.getError()(0, 0) = 8;
	error.getMetric()(0, 0) = 9;

	BOOST_CHECK_EQUAL(error.getError()(0, 0), 8);
	BOOST_CHECK_EQUAL(error.getMetric()(0, 0), 9);
}

BOOST_AUTO_TEST_CASE(syncHAndD2)
{
	ModelErrorDataGpu<float> error;
	error.setBatchSize(2);
	error.setMemorySize(3);
	error.setNMetrics(4);

  Eigen::Tensor<float, 2> error_tensor(2, 3), metric(4, 3);
  error_tensor.setConstant(3); metric.setConstant(4);

	error.setError(error_tensor);
	error.setMetric(metric);

	Eigen::GpuStreamDevice stream_device;
	Eigen::GpuDevice device(&stream_device);
	error.syncHAndDError(device);
	error.syncHAndDMetric(device);

	BOOST_CHECK(!error.getErrorStatus().first);
	BOOST_CHECK(error.getErrorStatus().second);
	BOOST_CHECK(!error.getMetricStatus().first);
	BOOST_CHECK(error.getMetricStatus().second);

	error.syncHAndDError(device);
	error.syncHAndDMetric(device);

	BOOST_CHECK(error.getErrorStatus().first);
	BOOST_CHECK(!error.getErrorStatus().second);
	BOOST_CHECK(error.getMetricStatus().first);
	BOOST_CHECK(!error.getMetricStatus().second);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	ModelErrorDataCpu<float> error;
	error.setBatchSize(2);
	error.setMemorySize(3);
	error.setNMetrics(4);
	size_t test_error = 2 * 3 * sizeof(float);
  size_t test_metric = 3 * 4 * sizeof(float);
	BOOST_CHECK_EQUAL(error.getErrorTensorSize(), test_error);
  BOOST_CHECK_EQUAL(error.getMetricTensorSize(), test_metric);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	ModelErrorDataCpu<float> error;
	error.setBatchSize(2);
	error.setMemorySize(3);
	error.setNMetrics(4);

	Eigen::Tensor<float, 2> error_tensor(2, 3), metric(4, 3);
  error_tensor.setConstant(3); metric.setConstant(4);

	error.setError(error_tensor);
	error.setMetric(metric);

	BOOST_CHECK_EQUAL(error.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(error.getMemorySize(), 3);
	BOOST_CHECK_EQUAL(error.getNMetrics(), 4);
	BOOST_CHECK_EQUAL(error.getError()(0, 0), 3);
	BOOST_CHECK(error.getErrorStatus().first);
	BOOST_CHECK(error.getErrorStatus().second);
	BOOST_CHECK_EQUAL(error.getMetric()(0, 0), 4);
	BOOST_CHECK(error.getMetricStatus().first);
	BOOST_CHECK(error.getMetricStatus().second);

	// Test mutability
	error.getError()(0, 0) = 8;
	error.getMetric()(0, 0) = 9;

	BOOST_CHECK_EQUAL(error.getError()(0, 0), 8);
	BOOST_CHECK_EQUAL(error.getMetric()(0, 0), 9);
}

BOOST_AUTO_TEST_CASE(syncHAndD)
{
	ModelErrorDataCpu<float> error;
	error.setBatchSize(2);
	error.setMemorySize(3);
	error.setNMetrics(4);

  Eigen::Tensor<float, 2> error_tensor(2, 3), metric(4, 3);
  error_tensor.setConstant(3); metric.setConstant(4);

	error.setError(error_tensor);
	error.setMetric(metric);

	Eigen::DefaultDevice device;;
	error.syncHAndDError(device);
	error.syncHAndDMetric(device);

	BOOST_CHECK(error.getErrorStatus().first);
	BOOST_CHECK(error.getErrorStatus().second);
	BOOST_CHECK(error.getMetricStatus().first);
	BOOST_CHECK(error.getMetricStatus().second);

	error.syncHAndDError(device);
	error.syncHAndDMetric(device);

	BOOST_CHECK(error.getErrorStatus().first);
	BOOST_CHECK(error.getErrorStatus().second);
	BOOST_CHECK(error.getMetricStatus().first);
	BOOST_CHECK(error.getMetricStatus().second);
}

BOOST_AUTO_TEST_CASE(initModelErrorData)
{
	ModelErrorDataCpu<float> error;
	error.initModelErrorData(2, 5, 4);

	// Test the batch and memory sizes
	BOOST_CHECK_EQUAL(error.getBatchSize(), 2);
	BOOST_CHECK_EQUAL(error.getMemorySize(), 5);
	BOOST_CHECK_EQUAL(error.getNMetrics(), 4);

	BOOST_CHECK_EQUAL(error.getError()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(error.getError()(1, 4), 0.0);
	BOOST_CHECK_EQUAL(error.getMetric()(0, 0), 0.0);
	BOOST_CHECK_EQUAL(error.getMetric()(3, 4), 0.0);
}

BOOST_AUTO_TEST_SUITE_END()