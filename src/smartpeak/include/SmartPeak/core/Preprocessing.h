#ifndef SMARTPEAK_PREPROCESSING_H
#define SMARTPEAK_PREPROCESSING_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <algorithm>
#include <map>
#include <random>

#define _USE_MATH_DEFINES
#include <math.h>

#define maxFunc(a,b)            (((a) > (b)) ? (a) : (b))
#define minFunc(a,b)            (((a) < (b)) ? (a) : (b))

namespace SmartPeak
{
	/*
	@brief Methods for data preprocessing, normalization, and random sampling
	*/

	template<typename T>
	T selectRandomElement(const std::vector<T>& elements)
	{
		try
		{
			// select a random node
			// based on https://www.rosettacode.org/wiki/Pick_random_element
			std::random_device seed;
			std::mt19937 engine(seed());
			std::uniform_int_distribution<int> choose(0, elements.size() - 1);
			return elements.at(choose(engine));
		}
		catch (std::exception& e)
		{
			printf("Exception in selectRandomElement: %s", e.what());
		}
	}

	/*
	@brief Scale by magnitude of the data
	*/
	template<typename T>
	class UnitScaleFunctor
	{
	public:
		UnitScaleFunctor() {};
		UnitScaleFunctor(const Eigen::Tensor<T, 2>& data) { setUnitScale(data); };
		~UnitScaleFunctor() {};
		void setUnitScale(const Eigen::Tensor<T, 2>& data)
		{
			const Eigen::Tensor<T, 0> max_value = data.maximum();
			const Eigen::Tensor<T, 0> min_value = data.minimum();
			unit_scale_ = 1 / sqrt(pow(max_value(0) - min_value(0), 2));
		}
		T getUnitScale() { return unit_scale_; }
		T operator()(const T& x_I) const { return x_I * unit_scale_; };
	private:
		T unit_scale_;
	};

	/*
	@brief Project the data onto a specific range
	*/
	template<typename T>
	class LinearScaleFunctor
	{
	public:
		LinearScaleFunctor() = default;
		LinearScaleFunctor(const T& domain_min, const T& domain_max, const T& range_min, const T& range_max):
			domain_min_(domain_min), domain_max_(domain_max), range_min_(range_min), range_max_(range_max){};
		~LinearScaleFunctor() = default;
		T operator()(const T& x_I) const { 
			T t = (x_I - domain_min_) / (domain_max_ - domain_min_);
			return (range_min_ + (range_max_ - range_min_) * t); 
		};
	private:
		T domain_min_;
		T domain_max_;
		T range_min_;
		T range_max_;
	};

  /*
  @brief Project the data onto a specific range
  */
  template<typename T, int N>
  class LinearScale
  {
  public:
    LinearScale() = default;
    LinearScale(const Eigen::Tensor<T, N>& data, const T& range_min, const T& range_max) :
      range_min_(range_min), range_max_(range_max) { setDomain(data); }
    LinearScale(const T& range_min, const T& range_max) :
      range_min_(range_min), range_max_(range_max) {}
    LinearScale(const T& domain_min, const T& domain_max, const T& range_min, const T& range_max) :
      domain_min_(domain_min), domain_max_(domain_max), range_min_(range_min), range_max_(range_max) {}
    ~LinearScale() = default;
    void setDomain(const T& domain_min, const T& domain_max) {
      domain_min_ = domain_min;
      domain_max_ = domain_max;
    }
    void setDomain(const Eigen::Tensor<T, N>& data) {
      const Eigen::Tensor<T, 0> max_value = data.maximum();
      const Eigen::Tensor<T, 0> min_value = data.minimum();
      domain_max_ = max_value(0);
      domain_min_ = min_value(0);
    }
    Eigen::Tensor<T, N> operator()(const Eigen::Tensor<T, N>& data) const {
      auto t = (data - data.constant(domain_min_)) / data.constant(domain_max_ - domain_min_);
      const Eigen::Tensor<T, N> data_linear = data.constant(range_min_) + data.constant(range_max_ - range_min_) * t;
      return data_linear;
    };
  private:
    T domain_min_;
    T domain_max_;
    T range_min_;
    T range_max_;
  };

  /*
  @brief Standardize the data using the Mean and Standard Deviation
  */
  template<typename T, int N>
  class Standardize
  {
  public:
    Standardize() = default;
    Standardize(const Eigen::Tensor<T, N>& data) { setMeanAndVar(data); };
    Standardize(const T& mean, const T& var): mean_(mean), var_(var) {};
    ~Standardize() = default;
    void setMeanAndVar(const T& mean, const T& var) {
      mean_ = mean;
      var_ = var;
    }
    void setMeanAndVar(const Eigen::Tensor<T, N>& data)
    {
      // calculate the total dimensions
      int dim_size_tot = 1;
      for (int i = 0; i < N; ++i) {
        dim_size_tot *= data.dimension(i);
      }

      // calculate the mean
      Eigen::Tensor<T, 0> mean_0d = data.mean();
      mean_ = mean_0d(0);

      // calculate the var
      auto residuals = data - data.constant(mean_);
      auto ssr = residuals.pow(2).sum();
      Eigen::Tensor<T, 0> var_0d = ssr / ssr.constant(dim_size_tot - 1);
      var_ = var_0d(0);
    }
    T getMean() { return mean_; }
    T getVar() { return var_; }
    Eigen::Tensor<T, N> operator()(const Eigen::Tensor<T, N>& data) const {
      const Eigen::Tensor<T, N> data_stand = (data - data.constant(mean_)) / 
        data.constant(var_).pow(T(0.5));
      return data_stand;
    };
  private:
    T mean_; ///< data set mean
    T var_; ///< data set var
  };

  /*
  @brief Generate a permutation Matrix that will randomly shuffle the order of the
    columns (Data * Permut) or rows (Permut * Data) when applied to the original Matrix
  */
  template<typename T>
  class MakeShuffleMatrix
  {
  public:
    MakeShuffleMatrix() = default;
    MakeShuffleMatrix(const std::vector<int>& indices, const bool& shuffle_cols) : indices_(indices) { setShuffleMatrix(shuffle_cols); };
    MakeShuffleMatrix(const int& shuffle_dim_size, const bool& shuffle_cols) { setIndices(shuffle_dim_size); setShuffleMatrix(shuffle_cols); };
    ~MakeShuffleMatrix() = default;
    void setIndices(const int& shuffle_dim_size) {
      // initialize the indices
      indices_.clear();
      indices_.reserve(shuffle_dim_size);
      for (int i = 0; i < shuffle_dim_size; ++i) indices_.push_back(i);

      // randomize the indices
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(indices_), std::end(indices_), rng);
    }
    std::vector<int> getIndices() const { return indices_; }
    void setShuffleMatrix(const bool& shuffle_cols) {
      // initialize the shuffle matrix
      assert(indices_.size() > 0);
      shuffle_matrix_.resize(int(indices_.size()), int(indices_.size()));
      shuffle_matrix_.setZero();

      // specify the ones in the shuffle matrix
      for (int dim_iter = 0; dim_iter < indices_.size(); ++dim_iter) {
        if (shuffle_cols) shuffle_matrix_(indices_.at(dim_iter), dim_iter) = T(1);
        else shuffle_matrix_(dim_iter, indices_.at(dim_iter)) = T(1); 
      }
    };
    Eigen::Tensor<T, 2> getShuffleMatrix() const { return shuffle_matrix_; };
    template<typename TT = T, std::enable_if_t<std::is_same<TT, float>::value && std::is_same<TT, T>::value, int> = 0>
    void operator()(Eigen::Tensor<TT, 2>& data, const bool& shuffle_cols) {
      if (shuffle_cols) data = data.contract(shuffle_matrix_, Eigen::array<Eigen::IndexPair<int>, 1>({ Eigen::IndexPair<int>(1, 0) })).eval();
      else data = shuffle_matrix_.contract(data, Eigen::array<Eigen::IndexPair<int>, 1>({ Eigen::IndexPair<int>(1, 0) })).eval();
    };
  private:
    std::vector<int> indices_; ///< indices used to crate the permutation matrix
    Eigen::Tensor<T, 2> shuffle_matrix_;
  };

	/*
	@brief "Smooth" binary labels 0 and 1 by a certain offset
	*/
	template<typename T>
	class LabelSmoother
	{
	public:
		LabelSmoother() = default;
		LabelSmoother(const T& zero_offset, const T& one_offset) :
			zero_offset_(zero_offset), one_offset_(one_offset) {};
		~LabelSmoother() = default;
		T operator()(const T& x_I) const {
			const T eps = 1e-3;
			if (x_I < eps) return x_I + zero_offset_;
			else if (x_I > 1 - eps) return x_I - one_offset_;
			else return x_I;
		};
	private:
		T zero_offset_;
		T one_offset_;
	};

	/*
	@brief One hot encoder

	@param[in] data Tensor of input labels in a single column vector
	@param{in] all_possible_values

	@returns an integer tensor where all rows have been expanded
		across the columns with the one hot encoding 
	*/
	template<typename Ta, typename Tb>
	Eigen::Tensor<Tb, 2> OneHotEncoder(const Eigen::Tensor<Ta, 2>& data, const std::vector<Ta>& all_possible_values)
	{
		// integer encode input data
		std::map<Ta, int> T_to_int;
		for (int i = 0; i<all_possible_values.size(); ++i)
			T_to_int.emplace(all_possible_values[i], i);

		// convert to 1 hot vector
		Eigen::Tensor<Tb, 2> onehot_encoded(data.dimension(0), (int)T_to_int.size());
		onehot_encoded.setZero();
		for (int i = 0; i<data.dimension(0); ++i)
			onehot_encoded(i, T_to_int.at(data(i, 0))) = 1;

		return onehot_encoded;
	}
	
	/*
	@brief One hot encoder

	@param[in] data input label
	@param{in] all_possible_values

	@returns an integer tensor with the one hot encoding
	*/
	template<typename Ta, typename Tb>
	Eigen::Tensor<Tb, 1> OneHotEncoder(const Ta& data, const std::vector<Ta>& all_possible_values)
	{
		// integer encode input data
		std::map<Ta, int> T_to_int;
		for (int i = 0; i<all_possible_values.size(); ++i)
			T_to_int.emplace(all_possible_values[i], i);

		// convert to 1 hot vector
		Eigen::Tensor<Tb, 1> onehot_encoded(T_to_int.size());
		onehot_encoded.setConstant(0);
		onehot_encoded(T_to_int.at(data)) = 1;

		return onehot_encoded;
	}

	/*
	@brief One hot categorical sampler

	@param[in] n_labels the number of categorical labels

	@returns an integer tensor with the one hot encoding
	*/
	template<typename Ta>
	Eigen::Tensor<Ta, 1> OneHotCategorical(const int& n_labels)
	{
		std::random_device seed;
		std::mt19937 engine(seed());
		std::uniform_int_distribution<int> choose(0, n_labels - 1);

		Eigen::Tensor<Ta, 1> onehot_encoded(n_labels);
		onehot_encoded.setZero();
		onehot_encoded(choose(engine)) = 1;

		return onehot_encoded;
	}

	template<typename Ta>
	std::pair<Ta, Ta> GaussianMixtureSampler(const Ta& x, const Ta& y, const int& label, const int& n_labels) {
		const Ta shift = 1.4;
		const Ta r = 2.0 * M_PI / Ta(n_labels) * Ta(label);
		Ta new_x = x * std::cos(r) - y * std::sin(r);
		Ta new_y = x * std::sin(r) + y * std::cos(r);
		new_x += shift * std::cos(r);
		new_y += shift * std::sin(r);
		return std::make_pair(new_x, new_y);
	};
	/*
	@brief 2D Gaussian mixture sampler

	@param[in] n_dims the number of categorical labels
	@param[in] n_labels the number of categorical labels

	@returns a Tensor of gaussian mixture samples
	*/
	template<typename Ta>
	Eigen::Tensor<Ta, 1> GaussianMixture(const int& n_dims, const int& n_labels, int label = -1)
	{
		assert(n_dims % 2 == 0);

		std::random_device rd{};
		std::mt19937 gen{ rd() };
		
		// make the gaussian mixture tensor
		Eigen::Tensor<Ta, 1> gaussian_mixture(n_dims);
		gaussian_mixture.setZero();
		const Ta x_var = 0.5;
		const Ta y_var = 0.05;
		int i = 0;
		while (i < n_dims) {
			// random integer
			if (label == -1) {
				std::random_device seed;
				std::mt19937 engine(seed());
				std::uniform_int_distribution<int> choose(0, n_labels - 1);
				label = choose(engine);
			}

			// sample from the mixture
			std::normal_distribution<> dx{ 0.0f, x_var };
			std::normal_distribution<> dy{ 0.0f, y_var };
			std::pair<Ta, Ta> samples = GaussianMixtureSampler<Ta>(dx(gen), dy(gen), label, n_labels);
			gaussian_mixture(i) = samples.first;
			gaussian_mixture(i + 1) = samples.second;

			i += 2;
		}

		return gaussian_mixture;
	}

	template<typename Ta>
	std::pair<Ta, Ta> SwissRollSampler(const int& label, const int& n_labels) {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> dist{ 0, 1 };
		const Ta uni = Ta(dist(gen)) / Ta(n_labels) + Ta(label) / Ta(n_labels);
		const Ta r = std::sqrt(uni) * 3.0;
		const Ta rad = M_PI * 4.0 * sqrt(uni);
		Ta new_x = r * std::cos(rad);
		Ta new_y = r * std::sin(rad);
		return std::make_pair(new_x, new_y);
	};
	/*
	@brief 2D Swiss roll sampler

	@param[in] n_dims the number of categorical labels
	@param[in] n_labels the number of categorical labels

	@returns a Tensor of gaussian mixture samples
	*/
	template<typename Ta>
	Eigen::Tensor<Ta, 1> SwissRoll(const int& n_dims, const int& n_labels, int label = -1)
	{
		assert(n_dims % 2 == 0);

		// make the gaussian mixture tensor
		Eigen::Tensor<Ta, 1> swiss_roll(n_dims);
		swiss_roll.setZero();
		int i = 0;
		while (i < n_dims) {
			// random integer
			if (label == -1) {
				std::random_device seed;
				std::mt19937 engine(seed());
				std::uniform_int_distribution<int> choose(0, n_labels - 1);
				label = choose(engine);
			}

			// sample from the mixture
			std::pair<Ta, Ta> samples = SwissRollSampler<Ta>(label, n_labels);
			swiss_roll(i) = samples.first;
			swiss_roll(i + 1) = samples.second;

			i += 2;
		}

		return swiss_roll;
	}

	/*
	@brief 1D Gumbel sampler

	where the Gumbel(0; 1) distribution can be sampled using inverse transform sampling by drawing u 
		Uniform(0; 1) and computing g = -log(-log(u)).

	@param[in] n_dims the number of categorical labels
	@param[in] n_labels the number of categorical labels

	@returns a Tensor of Gumbel samples
	*/
	template<typename Ta>
	Eigen::Tensor<Ta, 2> GumbelSampler(const int& n_dims, const int& n_labels) {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::uniform_real_distribution<> dist{ 0, 1 };
		Eigen::Tensor<Ta, 2> gumbel_dist(n_dims, n_labels);
		gumbel_dist = -(-(gumbel_dist.unaryExpr([&gen, &dist](const Ta& elem) {
			return Ta(dist(gen));
		})).log()).log();
		return gumbel_dist;
	};

	/*
	@brief 1D Gaussian sampler

	@param[in] n_dims the number of gaussian labels
	@param[in] n_labels the number of gaussian labels

	@returns a Tensor of Gaussian samples
	*/
	template<typename Ta>
	Eigen::Tensor<Ta, 2> GaussianSampler(const int& n_dims, const int& n_labels) {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> dist{ 0.0f, 1.0f };
		Eigen::Tensor<Ta, 2> gaussian_dist(n_dims, n_labels);
		gaussian_dist = gaussian_dist.unaryExpr([&gen, &dist](const Ta& elem) {
			return Ta(dist(gen)); 
		});
		return gaussian_dist;
	};

  /*
  @brief 1D Gaussian sampler

  @param[in] batch_size
  @param[in] memory_size
  @param[in] encoding_size
  @param[in] n_epochs

  @returns a Tensor of Gaussian samples
  */
  template<typename Ta>
  Eigen::Tensor<Ta, 4> GaussianSampler(const int& batch_size, const int& memory_size, const int& encoding_size, const int& n_epochs) {
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> dist{ 0.0f, 1.0f };
    Eigen::Tensor<Ta, 4> gaussian_dist(batch_size, memory_size, encoding_size, n_epochs);
    gaussian_dist = gaussian_dist.unaryExpr([&gen, &dist](const Ta& elem) {
      return Ta(dist(gen));
    });
    return gaussian_dist;
  };

	/**
	@brief Replaces NaN and Inf with 0
	*/
	template<typename T>
	T checkNan( const T& x)	{
		if (std::isnan(x))
			return T(0);
		else
			return x;
	}

	/**
	@brief Replaces NaN and Inf with 0 or 1e9 respectively
	*/
	template<typename T>
	T substituteNanInf(const T& x)
	{
		if (x == std::numeric_limits<T>::infinity())
			return T(1e9);
		else if (x == -std::numeric_limits<T>::infinity())
			return T(-1e9);
		else if (std::isnan(x))
			return T(0);
		else
			return x;
	}

	/**
	@brief Clip
	*/
	template<typename T>
	class ClipOp
	{
	public:
		ClipOp() = default;
		ClipOp(const T& eps, const T& min, const T& max) : eps_(eps), min_(min), max_(max) {};
		~ClipOp() = default;
		T operator()(const T& x) const {
			if (x < min_ + eps_)
				return min_ + eps_;
			else if (x > max_ - eps_)
				return max_ - eps_;
			else
				return x;
		}
	private:
		T eps_ = 1e-12; ///< threshold to clip between min and max
		T min_ = 0;
		T max_ = 1;
	};

	/**
	@brief return x or 0 with a specified probability
	*/
	template<typename T>
	class RandBinaryOp
	{
	public:
		RandBinaryOp() = default;
		RandBinaryOp(const T& p) : p_(p) {};
		~RandBinaryOp() = default;
		T operator()(const T& x) const {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::discrete_distribution<> distrib({ p_, 1 - p_ });
			return x * (T)distrib(gen);
		}
	private:
		T p_ = (T)1; ///< probablity of 0
	};

	/**
	@brief Scale a tensor by a specified value
	*/
	template<typename TensorT>
	class ScaleOp
	{
	public:
		ScaleOp() = default;
		ScaleOp(const TensorT& scale) : scale_(scale) {};
		~ScaleOp() = default;
		TensorT operator()(const TensorT& x_I) const {
			return x_I * scale_;
		}
	private:
		TensorT scale_ = 1;
	};

	/**
	@brief Offset a tensor by a specified value implemented as new_value = value + offset
	*/
	template<typename TensorT>
	class OffsetOp
	{
	public:
		OffsetOp() = default;
		OffsetOp(const TensorT& offset) : offset_(offset) {};
		~OffsetOp() = default;
		TensorT operator()(const TensorT& x_I) const {
			return x_I + offset_;
		}
	private:
		TensorT offset_ = 0;
	};

	/*
	@brief Test absolute and relative closeness of values

	References: http://realtimecollisiondetection.net/blog/?p=89

	@param: lhs Left Hand Side to compare
	@param: rhs Right Hand Side to Compare
	@param: rel_tol Relative Tolerance threshold
	@param: abs_tol Absolute Tolerance threshold

	@returns True or False
	*/
	template<typename T>
	bool assert_close(const T& lhs, const T& rhs, T rel_tol = 1e-4, T abs_tol = 1e-4) {
		return (std::fabs(lhs - rhs) <= maxFunc(abs_tol, rel_tol * maxFunc(fabs(lhs), fabs(rhs)) ));
	}

	///**
	//@brief Functor for use with base classes.
	//*/
	//template<typename TensorT, typename FunctorT>
	//class BaseClassFunctor
	//{
	//public:
	//	BaseClassFunctor() {};
	//	BaseClassFunctor(FunctorT<TensorT>* functor) : functor_(functor) {};
	//	~BaseClassFunctor() {};
	//	TensorT operator()(const TensorT& x_I) const {
	//		return (*functor_)(x_I);
	//	}
	//private:
	//	FunctorT<TensorT>* functor_;
	//};
}

#endif //SMARTPEAK_PREPROCESSING_H