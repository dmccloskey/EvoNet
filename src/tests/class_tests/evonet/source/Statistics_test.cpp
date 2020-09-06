/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE Statistics test suite 
#include <boost/test/included/unit_test.hpp>
#include <EvoNet/core/Statistics.h>

using namespace EvoNet;
using namespace std;

BOOST_AUTO_TEST_SUITE(statistics)

BOOST_AUTO_TEST_CASE(S_getConfidenceIntervals)
{
	std::vector<float> data = { 0, 2, 9, 8, 5, 3, 1, 7, 6, 4 };
	std::pair<float,float> result = confidence(data, 0.1f);
	BOOST_CHECK_CLOSE(result.first, 0, 1e-3);
	BOOST_CHECK_CLOSE(result.second, 9, 1e-3);
}

BOOST_AUTO_TEST_CASE(S_moment)
{
	float data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int n = 10;
	float ave, adev, sdev, var, skew, curt;
	moment(data, n, ave, adev, sdev, var, skew, curt);
	BOOST_CHECK_CLOSE(ave, 4.5, 1e-3);
	BOOST_CHECK_CLOSE(adev, 2.5, 1e-3);
	BOOST_CHECK_CLOSE(sdev, 3.02765, 1e-3);
	BOOST_CHECK_CLOSE(var, 9.1666667, 1e-3);
	BOOST_CHECK_CLOSE(skew, 0.0, 1e-3);
	BOOST_CHECK_CLOSE(curt, -1.56163645, 1e-3);
}

BOOST_AUTO_TEST_CASE(S_kstwo)
{
	float data1[] = { 0.55370819,-1.45963199,-1.29458514,-1.50967395,1.5718749,-0.97569619,0.48069879,0.62561431,0.72235302,0.91032644 };
	float data2[] = { 0.57870121,-1.60018641,0.25349027,-0.5041274,1.56796895,1.78298162,0.76469507,2.10362939,1.25984919,1.57030662,1.50733272,2.0732344 };
	float d, prob;
	kstwo(data1, 10, data2, 12, d, prob);
	BOOST_CHECK_CLOSE(d, 0.466666669, 1e-5);
	BOOST_CHECK_CLOSE(prob, 0.130679056, 1e-5);
	//BOOST_CHECK_CLOSE(d, 0.48333333333333334, 1e-5); //python3.6
	//BOOST_CHECK_CLOSE(prob, 0.10718344778577717, 1e-5); //python3.6
}

BOOST_AUTO_TEST_CASE(S_fisherExactTest)
{
	double prob = fisherExactTest<double>(1982, 3018, 2056, 2944);
	BOOST_CHECK_CLOSE(prob, 0.1367998254147284, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()