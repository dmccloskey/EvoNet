/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LOSSFUNCTION_H
#define SMARTPEAK_LOSSFUNCTION_H

#include <SmartPeak/core/Preprocessing.h>
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
		LossFunctionOp() = default;
		LossFunctionOp(const T& eps) : eps_(eps) {};
		~LossFunctionOp() = default;
		virtual std::string getName() = 0;
		virtual Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const = 0;
		virtual Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const = 0;
	protected:
		T eps_ = 1e-6;
	};

	/**
	@brief Base class loss function gradient.
	*/
	template<typename T>
	class LossFunctionGradOp
	{
	public:
		LossFunctionGradOp() = default;
		LossFunctionGradOp(const T& eps) : eps_(eps) {};
		~LossFunctionGradOp() = default;
		virtual std::string getName() = 0;
		virtual Eigen::Tensor<T, 2> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const = 0;
		virtual Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const = 0;
	protected:
		T eps_ = 1e-6;
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
		std::string getName() {	return "EuclideanDistanceOp"; };
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
			return ((y_true - y_pred).pow(2).sqrt()).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
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
		std::string getName() { return "EuclideanDistanceGradOp"; };
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
			return ((y_true - y_pred) / ((y_true - y_pred).pow(2).sqrt())).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
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
		std::string getName() { return "L2NormOp"; };
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
			return ((y_true - y_pred).pow(2) * c).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9)); // modified to simplify the derivative
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
		std::string getName() { return "L2NormGradOp"; };
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
			return (y_true - y_pred).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9)); // modified to exclude the 0.5
		};
  };

  /**
    @brief Binary Cross Entropy loss function.
  */
  template<typename T>
  class BCEOp : public LossFunctionOp<T>
  {
public: 
    BCEOp(){}; 
    ~BCEOp(){};
		std::string getName() { return "BCEOp"; };
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
      return -(y_true * (y_pred + this->eps_).log() + (ones - y_true) * (ones - (y_pred - this->eps_)).log()).sum(dims1);
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> ones((int)y_pred.size());
			ones.setConstant(1.0);
			return (-(y_true * (y_pred + this->eps_).log() + (ones - y_true) * (ones - (y_pred - this->eps_)).log())).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
  };

  /**
    @brief Binary Cross Entropy loss function gradient.

	The derivative of -(z * log(x) + (1 - z)*log(1-x)) is the following
		= (1-z)/(1-x) - z/x
		= -(x-z)/((x-1)*x)
  */
  template<typename T>
  class BCEGradOp : public LossFunctionGradOp<T>
  {
public: 
    BCEGradOp(){}; 
    ~BCEGradOp(){};
		std::string getName() { return "BCEGradOp"; };
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
			//return (-(y_true / y_pred + (ones - y_true) / (ones - y_pred))).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
			return (-(y_pred - y_true) / ((y_pred - this->eps_ - ones) * y_pred)).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
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
		std::string getName() { return "NegativeLogLikelihoodOp"; };
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
			Eigen::Tensor<T, 1> n((int)y_pred.size());
			n.setConstant(n_);
			return -y_true * (y_pred.unaryExpr(ClipOp<T>(1e-6, 0, 1)).log()) / n;
		};
	private:
		T n_ = 1.0; ///< the number of total classifiers
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
		std::string getName() { return "NegativeLogLikelihoodGradOp"; };
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
			Eigen::Tensor<T, 1> n((int)y_pred.size());
			n.setConstant(n_);
			//return (-y_true / (y_pred + this->eps_) / n).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
			return (y_true / (y_pred + this->eps_) / n).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
	private:
		T n_ = 1.0; ///< the number of total classifiers
  };

  /**
    @brief MSE Mean Squared Error loss function.
  */
  template<typename T>
  class MSEOp : public LossFunctionOp<T>
  {
public: 
    MSEOp(){};
		MSEOp(const T& n): n_(n){};
    ~MSEOp(){};
		std::string getName() { return "MSEOp"; };
    Eigen::Tensor<T, 1> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      const Eigen::Tensor<float, 1>::Dimensions dims1({1}); // sum along nodes
      Eigen::Tensor<T, 1> n(y_pred.dimensions()[0]);
      n.setConstant(n_);
      Eigen::Tensor<T, 1> c(y_pred.dimensions()[0]);
      c.setConstant(0.5);
      return (y_true - y_pred).pow(2).sum(dims1) * c / n;
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> n((int)y_pred.size());
			n.setConstant(n_);
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(0.5);
			return ((y_true - y_pred).pow(2) * c / n).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
private:
		T n_ = 1;
  };

  /**
    @brief MSE Mean Squared Error loss function gradient.
  */
  template<typename T>
  class MSEGradOp : public LossFunctionGradOp<T>
  {
public: 
    MSEGradOp(){};
		MSEGradOp(const T& n) : n_(n) {};
    ~MSEGradOp(){};
		std::string getName() { return "MSEGradOp"; };
    Eigen::Tensor<T, 2> operator()(
      const Eigen::Tensor<T, 2>& y_pred, 
      const Eigen::Tensor<T, 2>& y_true) const 
    {
      Eigen::Tensor<T, 2> n(y_pred.dimensions()[0], y_pred.dimensions()[1]);
      n.setConstant(n_);
      return (y_true - y_pred) / n;
    };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> n((int)y_pred.size());
			n.setConstant(n_);
			Eigen::Tensor<T, 1> result = (y_true - y_pred) / n;
			return result.unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
	private:
		T n_ = 1;
  };

	/**
		@brief KLDivergenceMu loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename T>
	class KLDivergenceMuOp : public LossFunctionOp<T>
	{
	public:
		KLDivergenceMuOp() {};
		~KLDivergenceMuOp() {};
		std::string getName() { return "KLDivergenceMuOp"; };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 1>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(0.5);
			return (-c + c*y_pred.pow(2)).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
	};

	/**
		@brief KLDivergenceMu  loss function gradient.
	*/
	template<typename T>
	class KLDivergenceMuGradOp : public LossFunctionGradOp<T>
	{
	public:
		KLDivergenceMuGradOp() {};
		~KLDivergenceMuGradOp() {};
		std::string getName() { return "KLDivergenceMuGradOp"; };
		Eigen::Tensor<T, 2> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 2>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(2.0);
			return c * y_pred;
		};
	};

	/**
		@brief KLDivergenceLogVar loss function.

	References
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
		0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	*/
	template<typename T>
	class KLDivergenceLogVarOp : public LossFunctionOp<T>
	{
	public:
		KLDivergenceLogVarOp() {};
		~KLDivergenceLogVarOp() {};
		std::string getName() { return "KLDivergenceLogVarOp"; };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 1>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(0.5);
			return (-c -c*y_pred + c*y_pred.exp()).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
	};

	/**
		@brief KLDivergenceLogVar  loss function gradient.
	*/
	template<typename T>
	class KLDivergenceLogVarGradOp : public LossFunctionGradOp<T>
	{
	public:
		KLDivergenceLogVarGradOp() {};
		~KLDivergenceLogVarGradOp() {};
		std::string getName() { return "KLDivergenceLogVarGradOp"; };
		Eigen::Tensor<T, 2> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 2>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> c((int)y_pred.size());
			c.setConstant(0.5);
			return (-c + c*y_pred.exp()).unaryExpr(ClipOp<T>(1e-6, -1e9, 1e9));
		};
	};

	/**
	@brief BCEWithLogits loss function.

	Binary Cross Entropy with integrated sigmoid layer
	z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
	= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
	= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
	= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
	= (1 - z) * x + log(1 + exp(-x))
	= x - x * z + log(1 + exp(-x))


	References:
	https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss

	PyTorch implementation:
	max_val = (-input).clamp(min=0)
	loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

	TensorFlow implementation:
	max(x, 0) - x * z + log(1 + exp(-abs(x)))
	*/
	template<typename T>
	class BCEWithLogitsOp : public LossFunctionOp<T>
	{
	public:
		BCEWithLogitsOp() {};
		~BCEWithLogitsOp() {};
		std::string getName() { return "BCEWithLogitsOp"; };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 1>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			// Step 1
			Eigen::Tensor<T, 1> zero((int)y_pred.size());
			zero.setConstant(0.0);
			auto max_values = (-y_pred).cwiseMax(zero);
			// Step 2
			Eigen::Tensor<T, 1> ones((int)y_pred.size());
			ones.setConstant(1.0);
			return y_pred - y_pred * y_true + max_values + ((-max_values).exp() + (-y_pred - max_values).exp()).log();
		};
	};

	/**
	@brief BCEWithLogits loss function gradient.

	Starting from the following BCEWithLogits formula
	x - x * z + log(1 + exp(-x))

	The derivative with respect to x can be formulated as
	1 - z + 1/(1 + exp(-x))*(-exp(-x))
	= -((z - 1)*exp(x) + z)/(exp(x) + 1)
	*/
	template<typename T>
	class BCEWithLogitsGradOp : public LossFunctionGradOp<T>
	{
	public:
		BCEWithLogitsGradOp() {};
		~BCEWithLogitsGradOp() {};
		std::string getName() { return "BCEWithLogitsGradOp"; };
		Eigen::Tensor<T, 2> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 2>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			Eigen::Tensor<T, 1> ones((int)y_pred.size());
			ones.setConstant(1.0);
			return -((y_true - ones)*y_pred.exp().unaryExpr(ClipOp<T>(1e-6, 0, 1e9)) + y_true)/(y_pred.exp().unaryExpr(ClipOp<T>(1e-6, 0, 1e9)) + ones);
		};
	};

	/**
		@brief CrossEntropyWithLogits loss function.
	*/
	template<typename T>
	class CrossEntropyWithLogitsOp : public LossFunctionOp<T>
	{
	public:
		CrossEntropyWithLogitsOp() = default;
		~CrossEntropyWithLogitsOp() = default;
		std::string getName() { return "CrossEntropyWithLogitsOp"; };
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 1>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return Eigen::Tensor<T, 1>();
		};
	};

	/**
		@brief CrossEntropyWithLogits loss function gradient.
	*/
	template<typename T>
	class CrossEntropyWithLogitsGradOp : public LossFunctionGradOp<T>
	{
	public:
		CrossEntropyWithLogitsGradOp() {};
		~CrossEntropyWithLogitsGradOp() {};
		std::string getName() { return "CrossEntropyWithLogitsGradOp"; };
		Eigen::Tensor<T, 2> operator()(
			const Eigen::Tensor<T, 2>& y_pred,
			const Eigen::Tensor<T, 2>& y_true) const
		{
			return Eigen::Tensor<T, 2>();
		};
		Eigen::Tensor<T, 1> operator()(
			const Eigen::Tensor<T, 1>& y_pred,
			const Eigen::Tensor<T, 1>& y_true) const
		{
			return Eigen::Tensor<T, 1>();
		};
	private:
		T n_ = 1.0; ///< the number of total classifiers
	};

	/**
		@brief Hinge loss function.  

		Typically used for classification

		NOTES: implemented as the following:
		def Hinge(yHat, y):
			return np.max(0, 1 - yHat * y)
	*/
}
#endif //SMARTPEAK_LOSSFUNCTION_H