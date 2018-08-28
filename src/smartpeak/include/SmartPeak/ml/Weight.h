/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHT_H
#define SMARTPEAK_WEIGHT_H

#include <SmartPeak/ml/Solver.h>
#include <SmartPeak/ml/WeightInit.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <tuple>
#include <string>

namespace SmartPeak
{

  /**
    @brief Directed Network Weight
  */
  class Weight
  {
public:
    Weight(); ///< Default constructor
    Weight(const Weight& other); ///< Copy constructor // [TODO: add test]
    Weight(const int& id); ///< Explicit constructor 
    Weight(const std::string& name); ///< Explicit constructor 
    Weight(const int& id, const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver); ///< Explicit constructor 
    Weight(const std::string& name, const std::shared_ptr<WeightInitOp>& weight_init, const std::shared_ptr<SolverOp>& solver); ///< Explicit constructor 
    ~Weight(); ///< Default destructor

    inline bool operator==(const Weight& other) const
    {
      return
        std::tie(
          id_,
          name_,
					module_id_,
					module_name_
        ) == std::tie(
          other.id_,
          other.name_,
					other.module_id_,
					other.module_name_
        )
      ;
    }

    inline bool operator!=(const Weight& other) const
    {
      return !(*this == other);
    }

    inline Weight& operator=(const Weight& other)
    { // [TODO: add test]
      id_  = other.id_;
      name_  = other.name_;
			module_id_ = other.module_id_;
			module_name_ = other.module_name_;
      weight_  = other.weight_;
      weight_init_ = other.weight_init_;
      solver_ = other.solver_;
      weight_min_ = other.weight_min_;
      weight_max_ = other.weight_max_;
			drop_probability_ = other.drop_probability_;
      return *this;
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< naem setter
    std::string getName() const; ///< name getter

    void setWeight(const float& weight); ///< weight setter
    float getWeight() const; ///< weight getter
		float* getWeightMutable(); ///< weight getter

    void setWeightInitOp(const std::shared_ptr<WeightInitOp>& weight_init); ///< weight initialization operator setter
    WeightInitOp* getWeightInitOp() const; ///< weight initialization operator getter

    void setSolverOp(const std::shared_ptr<SolverOp>& solver); ///< weight update operator setter
    SolverOp* getSolverOp() const; ///< weight update operator getter

    void setWeightMin(const float& weight_min); ///< min weight setter
    void setWeightMax(const float& weight_max); ///< max weight setter

		void setModuleId(const int& module_id); ///< module id setter
		int getModuleId() const; ///< module id getter

		void setModuleName(const std::string& module_name); ///< module name setter
		std::string getModuleName() const; ///< module name getter

		void setDropProbability(const float& drop_probability); ///< drop_probability setter
		float getDropProbability() const; ///< drop_probability getter

		void setDrop(const float& drop); ///< drop setter
		float getDrop() const; ///< drop getter

    /**
      @brief Initializes the weight.  
    */ 
    void initWeight();
 
    /**
      @brief Update the weight.

      @param[in] errpr Weight error   
    */ 
    void updateWeight(const float& error);
 
    /**
      @brief Check if the weight is within the min/max.  
    */ 
    void checkWeight();

private:
    int id_ = -1; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
		int module_id_ = -1; ///< Module ID
		std::string module_name_ = ""; ///<Module Name
    float weight_ = 1.0; ///< Weight weight
    std::shared_ptr<WeightInitOp> weight_init_; ///< weight initialization operator
    std::shared_ptr<SolverOp> solver_; ///< weight update operator

    float weight_min_ = -1.0e6;
    float weight_max_ = 1.0e6;
		float drop_probability_ = 0.0;
		float drop_ = 1.0;
  };
}

#endif //SMARTPEAK_WEIGHT_H