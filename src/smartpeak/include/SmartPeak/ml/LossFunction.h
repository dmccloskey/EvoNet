/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LOSSFUNCTION_H
#define SMARTPEAK_LOSSFUNCTION_H

#include <SmartPeak/ml/SharedFunctions.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <random>
#include <iostream>

namespace SmartPeak
{
	/**
	@brief Base class loss function.
	*/
	template<typename T>
	class LossFunctionOp
	{
	public:
		LossFunctionOp() {};
		~LossFunctionOp() {};
		virtual Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const = 0;
		virtual Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const = 0;
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename T>
	class LossFunctionGradOp
	{
	public:
		LossFunctionGradOp() {};
		~LossFunctionGradOp() {};
		virtual Eigen::Tensor<T, 2> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const = 0;
		virtual Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const = 0;
	};

  /**
    @brief EuclideanDistance loss function.
  */
  template<typename T>
  class EuclideanDistanceOp : public LossFunctionOp<T>
  {
public: 
    EuclideanDistanceOp(){}; 
    ~EuclideanDistanceOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims({1}); // sum along nodes
      return (y_true - y_pred).pow(2).sum(dims).sqrt();
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return ((y_true - y_pred).pow(2).sqrt()).unaryExpr(std::ptr_fun(substituteNanInf<T>));
		};
  };

  /**
    @brief EuclideanDistance loss function gradient.
  */
  template<typename T>
  class EuclideanDistanceGradOp : public LossFunctionGradOp<T>
  {
public: 
    EuclideanDistanceGradOp(){}; 
    ~EuclideanDistanceGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      //Eigen::Tensor<T, 1> a = (y_true - y_pred).pow(2).sum(dims1).sqrt();  [TODO: delete]
      const Eigen::array<int, 2> new_dims = {(int)y_pred.dimensions()[0], 1}; // reshape to a column vector of size batch_size
      const Eigen::array<int, 2> bcast = {1, (int)y_pred.dimensions()[1]}; // broadcast along the number of nodes
	  //const Eigen::array<int, 2> new_dims({ (int)y_pred.dimensions()[0], 1 }); // reshape to a column vector of size batch_size
	  //const Eigen::array<int, 2> bcast({ 1, (int)y_pred.dimensions()[1] }); // broadcast along the number of nodes
      return (y_true - y_pred)/(
        (y_true - y_pred).pow(2).sum(dims1).sqrt().eval()
          .reshape(new_dims).broadcast(bcast));
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return ((y_true - y_pred) / ((y_true - y_pred).pow(2).sqrt())).unaryExpr(std::ptr_fun(substituteNanInf<T>));
		};
  };

  /**
    @brief L2Norm loss function.
  */
  template<typename T>
  class L2NormOp : public LossFunctionOp<T>
  {
public: 
    L2NormOp(){}; 
    ~L2NormOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 1> c(y_pred.dimensions()[0]);
      c.setConstant(0.5);
      return (y_true - y_pred).pow(2).sum(dims1) * c; // modified to simplify the derivative
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(0.5);
			return ((y_true - y_pred).pow(2) * c).unaryExpr(std::ptr_fun(substituteNanInf<T>)); // modified to simplify the derivative
		};
  };

  /**
    @brief L2Norm loss function gradient.
  */
  template<typename T>
  class L2NormGradOp : public LossFunctionGradOp<T>
  {
public: 
    L2NormGradOp(){}; 
    ~L2NormGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      return y_true - y_pred; // modified to exclude the 0.5
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return (y_true - y_pred).unaryExpr(std::ptr_fun(substituteNanInf<T>)); // modified to exclude the 0.5
		};
  };

  /**
    @brief CrossEntropy loss function.
  */
  template<typename T>
  class CrossEntropyOp : public LossFunctionOp<T>
  {
public: 
    CrossEntropyOp(){}; 
    ~CrossEntropyOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      // // traditional
      // Eigen::Tensor<T, 0> n;
      // n.setValues({y_pred.dimensions()[0]});
      // Eigen::Tensor<T, 0> one;
      // one.setValues({1.0});
      // Eigen::Tensor<T, 1> ones(y_pred.dimensions()[0]);
      // ones.setConstant(1.0);
      // return -(y_true * y_pred.log() + (ones - y_true) * (ones - y_pred).log()).sum() * one / n;
      // simplified
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 2> ones(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      ones.setConstant(1.0);
      return -(y_true * y_pred.log() + (ones - y_true) * (ones - y_pred).log()).sum(dims1);
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> ones((int)y_pred.size());
			ones.setConstant(1.0);
			return (-(y_true * y_pred.log() + (ones - y_true) * (ones - y_pred).log())).unaryExpr(std::ptr_fun(substituteNanInf<T>));
		};
  };

  /**
    @brief CrossEntropy loss function gradient.
  */
  template<typename T>
  class CrossEntropyGradOp : public LossFunctionGradOp<T>
  {
public: 
    CrossEntropyGradOp(){}; 
    ~CrossEntropyGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      // simplified
      Eigen::Tensor<T, 2> ones(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      ones.setConstant(1.0);
      return -(y_true / y_pred + (ones - y_true) / (ones - y_pred));
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> ones((int)y_pred.size());
			ones.setConstant(1.0);
			return (-(y_true / y_pred + (ones - y_true) / (ones - y_pred))).unaryExpr(std::ptr_fun(substituteNanInf<T>));
		};
  };

  /**
    @brief NegativeLogLikelihood loss function.

		NOTES: implemented as the following:
		def CrossEntropy(yHat, y):
			if y == 1:
				return -log(yHat)
			else:
				return -log(1 - yHat)
  */
  template<typename T>
  class NegativeLogLikelihoodOp : public LossFunctionOp<T>
  {
public: 
    NegativeLogLikelihoodOp() = default;
		NegativeLogLikelihoodOp(const T& n) { setN(n); };
		void setN(const T& n) { n_ = n; }
    ~NegativeLogLikelihoodOp() = default;
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      return -(y_true * y_pred.log()).sum(dims1);
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return -y_true * (y_pred.unaryExpr(ClipOp<T>(1e-12, 0, 1)).log());
		};
	private:
		T n_ = 1; ///< the number of total classifiers
  };

  /**
    @brief NegativeLogLikelihood loss function gradient.
  */
  template<typename T>
  class NegativeLogLikelihoodGradOp : public LossFunctionGradOp<T>
  {
public: 
    NegativeLogLikelihoodGradOp(){};
		NegativeLogLikelihoodGradOp(const T& n) { setN(n); };
		void setN(const T& n) { n_ = n; }
    ~NegativeLogLikelihoodGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      return -(y_true / y_pred);
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return (-y_true / y_pred.unaryExpr(ClipOp<T>(1e-12, 0, 1))).unaryExpr(std::ptr_fun(checkNanInf<T>));
		};
	private:
		T n_ = 1; ///< the number of total classifiers
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename T>
  class MSEOp : public LossFunctionOp<T>
  {
public: 
    MSEOp(){}; 
    ~MSEOp(){};
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 1> n(y_pred.dimensions()[0]);
      n.setConstant(y_pred.dimensions()[0]);
      Eigen::Tensor<T, 1> c(y_pred.dimensions()[0]);
      c.setConstant(0.5);
      return (y_true - y_pred).pow(2).sum(dims1) * c / n;
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> n((int)y_pred.size());
			n.setConstant((int)y_pred.size());
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(0.5);
			return ((y_true - y_pred).pow(2) * c / n).unaryExpr(std::ptr_fun(substituteNanInf<T>));
		};
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename T>
  class MSEGradOp : public LossFunctionGradOp<T>
  {
public: 
    MSEGradOp(){}; 
    ~MSEGradOp(){};
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      Eigen::Tensor<T, 2> n(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      n.setConstant(y_pred.dimensions()[0]);
      return (y_true - y_pred) / n;
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> n((int)y_pred.size());
			n.setConstant((int)y_pred.size());
			Eigen::Tensor<T, 1> result = (y_true - y_pred) / n;
			return result.unaryExpr(std::ptr_fun(substituteNanInf<T>));
		};
  };



	/**
		@brief Hinge loss function.  

		Typically used for classification

		NOTES: implemented as the following:
		def Hinge(yHat, y):
			return np.max(0, 1 - yHat * y)
	*/


	/**
		@brief CrossEntropy loss function.

		Typically used for regression

		https://en.wikipedia.org/wiki/Huber_loss
	*/
}
#endif //SMARTPEAK_LOSSFUNCTION_H