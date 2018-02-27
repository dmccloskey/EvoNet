/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LINK_H
#define SMARTPEAK_LINK_H

namespace SmartPeak
{

  /**
    @brief Directed Network Link
  */
  class Link
  {
public:
    Link(); ///< Default constructor
    Link(const int& id, const int& source_node_id,
      const int& sink_node_id); ///< Explicit constructor  
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
    int getSourceNode() const; ///< source_node_id getter

    void setSinkNodeId(const int& sink_node_id); ///< sink_node_id setter
    int getSinkNode() const; ///< sink_node_id getter

    void setWeight(const double& weight); ///< weight setter
    double getWeight() const; ///< weight getter

private:
    int id_; ///< Link ID
    int source_node_id_; ///< Link source node
    int sink_node_id_; ///< Link sink node
    double weight_ = 1.0; ///< Link weight

  };
}

#endif //SMARTPEAK_LINK_H