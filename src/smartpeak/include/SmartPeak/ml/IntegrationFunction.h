/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTEGRATIONFUNCTION_H
#define SMARTPEAK_INTEGRATIONFUNCTION_H

#include <SmartPeak/ml/SharedFunctions.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Base class for all integration functions.
  */
	template<typename T>
  class IntegrationOp
  {
	public: 
    IntegrationOp() = default;
    ~IntegrationOp() = default;
    void setEps(const T& eps) { eps_ = eps; }
    T getEps() const { return eps_; };
    virtual std::string getName() const = 0;
    virtual IntegrationOp<T>* copy() const = 0;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(eps_);
		}
		T eps_ = 1e-6;
  };

  /**
    @brief Sum integration function
  */
  template<typename T>
  class SumOp: public IntegrationOp<T>
  {
public: 
		using IntegrationOp<T>::IntegrationOp;
    std::string getName() const{return "SumOp";};
    IntegrationOp<T>* copy() const { return new SumOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
  };

	/**
	@brief Product integration function
	*/
	template<typename T>
	class ProdOp : public IntegrationOp<T>
	{
	public:
		using IntegrationOp<T>::IntegrationOp;
		std::string getName() const { return "ProdOp"; };
    IntegrationOp<T>* copy() const { return new ProdOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
	};

  /**
  @brief Product singly connected integration function
  */
  template<typename T>
  class ProdSCOp : public IntegrationOp<T>
  {
  public:
    using IntegrationOp<T>::IntegrationOp;
    std::string getName() const { return "ProdSCOp"; };
    IntegrationOp<T>* copy() const { return new ProdSCOp<T>(*this); }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<IntegrationOp<T>>(this));
    }
  };

	/**
	@brief Max integration function
	*/
	template<typename T>
	class MaxOp : public IntegrationOp<T>
	{
	public:
		using IntegrationOp<T>::IntegrationOp;
		std::string getName() const { return "MaxOp"; };
    IntegrationOp<T>* copy() const { return new MaxOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
	};

  /**
  @brief Max integration function
  */
  template<typename T>
  class MinOp : public IntegrationOp<T>
  {
  public:
    using IntegrationOp<T>::IntegrationOp;
    std::string getName() const { return "MinOp"; };
    IntegrationOp<T>* copy() const { return new MinOp<T>(*this); }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<IntegrationOp<T>>(this));
    }
  };

	/**
		@brief Mean integration function
	*/
	template<typename T>
	class MeanOp : public IntegrationOp<T>
	{
	public:
		using IntegrationOp<T>::IntegrationOp;
		std::string getName() const { return "MeanOp"; };
    IntegrationOp<T>* copy() const { return new MeanOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
	};

	/**
		@brief Variance integration function

		References:
		T.F.Chan, G.H. Golub and R.J. LeVeque (1983). ""Algorithms for computing the sample variance: Analysis and recommendations", The American Statistician, 37": 242–247.
	*/
	template<typename T>
	class VarianceOp : public IntegrationOp<T>
	{
	public:
		using IntegrationOp<T>::IntegrationOp;
		//Eigen::Tensor<T, 1> getNetNodeInput() const { 
		//	Eigen::Tensor<T, 1> n(this->net_node_input_.dimension(0));
		//	n.setConstant(this->n_); 
		//	return (this->net_node_input_  - (ex_ * ex_)/ n)/n; }
		//void operator()(const Eigen::Tensor<T, 1>& weight, const Eigen::Tensor<T, 1>&source_output) {
		//	auto input = weight * source_output;
		//	if (this->n_ == 0)
		//		k_ = input;
		//	auto input_k = input - k_;
		//	ex_ += input_k;
		//	this->n_ += 1;
		//	this->net_node_input_ += (input_k * input_k);
		//};
		std::string getName() const { return "VarianceOp"; };
    IntegrationOp<T>* copy() const { return new VarianceOp<T>(*this); }
	//private:
	//	Eigen::Tensor<T, 1> k_ = 0;
	//	Eigen::Tensor<T, 1> ex_ = 0;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
	};

	/**
		@brief VarMod integration function

		Modified variance integration function: 1/n Sum[0 to n](Xi)^2
		where Xi = xi - u (u: mean, xi: single sample)
	*/
	template<typename T>
	class VarModOp : public IntegrationOp<T>
	{
	public:
		using IntegrationOp<T>::IntegrationOp;
		std::string getName() const { return "VarModOp"; };
    IntegrationOp<T>* copy() const { return new VarModOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
	};

	/**
		@brief Count integration function
	*/
	template<typename T>
	class CountOp : public IntegrationOp<T>
	{
	public:
		using IntegrationOp<T>::IntegrationOp;
		std::string getName() const { return "CountOp"; };
    IntegrationOp<T>* copy() const { return new CountOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationOp<T>>(this));
		}
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename T>
	class IntegrationErrorOp
	{
	public:
		IntegrationErrorOp() = default;
		~IntegrationErrorOp() = default;
    void setEps(const T& eps) { eps_ = eps; }
    T getEps() const { return eps_; };
		virtual std::string getName() const = 0;
    virtual IntegrationErrorOp<T>* copy() const = 0;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(eps_);
		}
		T eps_ = 1e-6;
	};

	/**
	@brief Sum integration error function
	*/
	template<typename T>
	class SumErrorOp : public IntegrationErrorOp<T>
	{
	public:
		using IntegrationErrorOp<T>::IntegrationErrorOp;
		std::string getName() const { return "SumErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new SumErrorOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationErrorOp<T>>(this));
		}
	};

	/**
	@brief Product integration error function
	*/
	template<typename T>
	class ProdErrorOp : public IntegrationErrorOp<T>
	{
	public:
		using IntegrationErrorOp<T>::IntegrationErrorOp;
		std::string getName() const { return "ProdErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new ProdErrorOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationErrorOp<T>>(this));
		}
	};

	/**
	@brief Max integration error function
	*/
	template<typename T>
	class MaxErrorOp : public IntegrationErrorOp<T>
	{
	public:
		using IntegrationErrorOp<T>::IntegrationErrorOp;
		std::string getName() const { return "MaxErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new MaxErrorOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationErrorOp<T>>(this));
		}
	};

  /**
  @brief Min integration error function
  */
  template<typename T>
  class MinErrorOp : public IntegrationErrorOp<T>
  {
  public:
    using IntegrationErrorOp<T>::IntegrationErrorOp;
    std::string getName() const { return "MinErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new MinErrorOp<T>(*this); }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<IntegrationErrorOp<T>>(this));
    }
  };

	/**
	@brief Mean integration error function
	*/
	template<typename T>
	class MeanErrorOp : public IntegrationErrorOp<T>
	{
	public:
		using IntegrationErrorOp<T>::IntegrationErrorOp;
		std::string getName() const { return "MeanErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new MeanErrorOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationErrorOp<T>>(this));
		}
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename T>
	class VarModErrorOp : public IntegrationErrorOp<T>
	{
	public:
		using IntegrationErrorOp<T>::IntegrationErrorOp;
		std::string getName() const { return "VarModErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new VarModErrorOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationErrorOp<T>>(this));
		}
	};

	/**
	@brief Count integration error function
	*/
	template<typename T>
	class CountErrorOp : public IntegrationErrorOp<T>
	{
	public:
		using IntegrationErrorOp<T>::IntegrationErrorOp;
		std::string getName() const { return "CountErrorOp"; };
    IntegrationErrorOp<T>* copy() const { return new CountErrorOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationErrorOp<T>>(this));
		}
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename T>
	class IntegrationWeightGradOp
	{
	public:
		IntegrationWeightGradOp() = default;
		~IntegrationWeightGradOp() = default;
    void setEps(const T& eps) { eps_ = eps; }
    T getEps() const { return eps_; };
		virtual std::string getName() const = 0;
    virtual IntegrationWeightGradOp<T>* copy() const = 0;
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(eps_);
		}
		T eps_ = 1e-6;
	};

	/**
	@brief Sum integration error function
	*/
	template<typename T>
	class SumWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
		std::string getName() const { return "SumWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new SumWeightGradOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
		}
	};

	/**
	@brief Product integration error function
	*/
	template<typename T>
	class ProdWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
		std::string getName() const { return "ProdWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new ProdWeightGradOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
		}
	};

	/**
	@brief Max integration error function
	*/
	template<typename T>
	class MaxWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
		std::string getName() const { return "MaxWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new MaxWeightGradOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
		}
	};

  /**
  @brief Min integration error function
  */
  template<typename T>
  class MinWeightGradOp : public IntegrationWeightGradOp<T>
  {
  public:
    using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
    std::string getName() const { return "MinWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new MinWeightGradOp<T>(*this); }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
    }
  };

	/**
	@brief Count integration error function
	*/
	template<typename T>
	class CountWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
		std::string getName() const { return "CountWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new CountWeightGradOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
		}
	};

	/**
	@brief Mean integration error function
	*/
	template<typename T>
	class MeanWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
		std::string getName() const { return "MeanWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new MeanWeightGradOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
		}
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename T>
	class VarModWeightGradOp : public IntegrationWeightGradOp<T>
	{
	public:
		using IntegrationWeightGradOp<T>::IntegrationWeightGradOp;
		std::string getName() const { return "VarModWeightGradOp"; };
    IntegrationWeightGradOp<T>* copy() const { return new VarModWeightGradOp<T>(*this); }
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<IntegrationWeightGradOp<T>>(this));
		}
	};
}

CEREAL_REGISTER_TYPE(SmartPeak::SumOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::ProdOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::ProdSCOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MaxOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MinOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MeanOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::VarianceOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::VarModOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::CountOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::SumErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::ProdErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MaxErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MinErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MeanErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::VarModErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::CountErrorOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::SumWeightGradOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::ProdWeightGradOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MaxWeightGradOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MinWeightGradOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::CountWeightGradOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::MeanWeightGradOp<float>);
CEREAL_REGISTER_TYPE(SmartPeak::VarModWeightGradOp<float>);

//CEREAL_REGISTER_TYPE(SmartPeak::SumOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdSCOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MinOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarianceOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MinErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountErrorOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumWeightGradOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdWeightGradOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxWeightGradOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MinWeightGradOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountWeightGradOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanWeightGradOp<double>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModWeightGradOp<double>);
//
//CEREAL_REGISTER_TYPE(SmartPeak::SumOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdSCOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MinOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarianceOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MinErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountErrorOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumWeightGradOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdWeightGradOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxWeightGradOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MinWeightGradOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountWeightGradOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanWeightGradOp<int>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModWeightGradOp<int>);
#endif //SMARTPEAK_INTEGRATIONFUNCTION_H