/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LINK_H
#define SMARTPEAK_LINK_H

#include <tuple>

namespace SmartPeak
{

  enum class WeightInitMethod
  {
    RandWeightInit = 0,
    ConstWeightInit = 1,
  };

  /**
    @brief Directed Network Link
  */
  class Link
  {
public:
    Link(); ///< Default constructor
    Link(const int& id, const int& source_node_id,
      const int& sink_node_id); ///< Explicit constructor  
    Link(const int& id, const int& source_node_id,
      const int& sink_node_id,
      const SmartPeak::WeightInitMethod& weight_init); ///< Explicit constructor 
    ~Link(); ///< Default destructor

    inline bool operator==(const Link& other) const
    {
      return
        std::tie(
          id_,
          source_node_id_,
          sink_node_id_,
          weight_
        ) == std::tie(
          other.id_,
          other.source_node_id_,
          other.sink_node_id_,
          other.weight_
        )
      ;
    }

    inline bool operator!=(const Link& other) const
    {
      return !(*this == other);
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setSourceNodeId(const int& source_node_id); ///< source_node_id setter
    int getSourceNodeId() const; ///< source_node_id getter

    void setSinkNodeId(const int& sink_node_id); ///< sink_node_id setter
    int getSinkNodeId() const; ///< sink_node_id getter

    void setWeight(const double& weight); ///< weight setter
    double getWeight() const; ///< weight getter

    void setWeightInitMethod(const SmartPeak::WeightInitMethod& weight_init); ///< weight_init_ setter
    SmartPeak::WeightInitMethod getWeightInitMethod() const; ///< weight_init_ getter

private:
    int id_; ///< Link ID
    int source_node_id_; ///< Link source node
    int sink_node_id_; ///< Link sink node
    double weight_ = 1.0; ///< Link weight
    SmartPeak::WeightInitMethod weight_init_ = SmartPeak::WeightInitMethod::ConstWeightInit; ///< Weight Init method
  };
}

#endif //SMARTPEAK_LINK_H