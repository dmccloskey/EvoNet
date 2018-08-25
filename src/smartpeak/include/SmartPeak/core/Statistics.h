#ifndef SMARTPEAK_STATISTICS_H
#define SMARTPEAK_STATISTICS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include <vector>
#include <iostream>

// Numerical Recipes C definitions
#define EPS1 1.0e-24
#define EPS2 1.0e-24

namespace SmartPeak
{
	/*
	@brief Methods for statistics
	*/

	/*
	@brief Calculate the Confidence Intervals for a distribution of data

	@param[in] data Distribution of data
	@param[in] alpha Confidence level

	@return pair of lower bound and upper bounds for the data
	*/
	template<typename T>
	std::pair<T, T> confidence(const std::vector<T>& data, T alpha=0.05)
	{
		std::vector<T> data_sorted = data;
		std::sort(data_sorted.begin(), data_sorted.end());
		int n = data_sorted.size();
		T lb = data_sorted[int((alpha / 2.0)*n)];
		T ub = data_sorted[int((1 - alpha / 2.0)*n)];
		return std::make_pair(lb, ub);
	}

	/*
	@brief Moments of a distribution

	Given an array of data[1..n], this routine returns its mean ave, average deviation adev,
	standard deviation sdev, variance var, skewness skew, and kurtosis curt.

	References:
	Numerical Recipes in C pg 613
	*/
	template<typename T>
	void moment(T data[], int n, T &ave, T &adev, T &sdev,
		T &var, T &skew, T &curt)
	{
		int j;
		T ep = 0.0, s, p;
		if (n <= 1) {
			std::cout << "n must be at least 2 in moment" << std::endl;
			return;
		}
		s = 0.0; 
		for (j = 0; j < n; j++) s += data[j];
		ave = s / n;
		adev = (var) = (skew) = (curt) = 0.0; 
		for (j = 0; j < n; j++) {
			adev += fabs(s = data[j] - (ave));
			ep += s;
			var += (p = s * s);
			skew += (p *= s);
			curt += (p *= s);
		}
		adev /= n;
		var = (var - ep * ep / n) / (n - 1); 
			sdev = sqrt(var); 
		if (var) {
			skew /= (n*(var)*(sdev));
			curt = (curt) / (n*(var)*(var)) - 3.0;
		}
		else {
			std::cout << "No skew/kurtosis when variance = 0 (in moment)" << std::endl;
			return;
		}
	}

	/*
	@brief Kolmogorov - Smirnov probability function.

	References:
	Numerical Recipes in C pg 626
	*/
	template<typename T>
	T probks(T alam)
	{
		int j;
		T a2, fac = 2.0, sum = 0.0, term, termbf = 0.0;
		a2 = -2.0*alam*alam;
		for (j = 1; j <= 100; j++) {
			term = fac * exp(a2*j*j);
			sum += term;
			if (fabs(term) <= EPS1 * termbf || fabs(term) <= EPS2 * sum) return sum;
			fac = -fac;
			termbf = fabs(term);
		}
		return 1.0; 
	}

	/*
	@brief Kolmogorov-Smirnov Test two way

	Given an array data1[1..n1], and an array data2[1..n2], this routine returns the K–
	S statistic d, and the significance level prob for the null hypothesis that the data sets are
	drawn from the same distribution.Small values of prob showthat the cumulative distribution
	function of data1 is significantly different from that of data2.The arrays data1 and data2
	are modified by being sorted into ascending order.

	References:
	Numerical Recipes in C pg 625
	*/
	template<typename T>
	void kstwo(T data1[], unsigned long n1, T data2[], unsigned long n2,
		T &d, T &prob)
	{
		unsigned long j1 = 0, j2 = 0;
		T d1, d2, dt, en1, en2, en, fn1 = 0.0, fn2 = 0.0;
		std::sort(data1, data1+n1);
		std::sort(data2, data2+n2);
		en1 = n1;
		en2 = n2;
		d = 0.0;
		while (j1 < n1 && j2 < n2) {
			if ((d1 = data1[j1]) <= (d2 = data2[j2])) fn1 = j1++ / en1;
			if (d2 <= d1) fn2 = j2++ / en2;
			if ((dt = fabs(fn2 - fn1)) > d) d = dt;
		}
		en = sqrt(en1*en2 / (en1 + en2));
		prob = probks((en + 0.12 + 0.11 / en)*(d));
	}
}
#endif //SMARTPEAK_STATISTICS_H