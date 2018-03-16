/**TODO:  Add copyright*/

#ifndef SMARTPEAK_LINK_H
#define SMARTPEAK_LINK_H

#include <tuple>

namespace SmartPeak
{

  /**
    @brief Directed Network Link
  */
  class Link
  {
public:
    Link(); ///< Default constructor
    Link(const int& id); ///< Explicit constructor
    Link(const int& id,
      const int& source_node_id,
      const int& sink_node_id,
      const int& weight_id); ///< Explicit constructor
    ~Link(); ///< Default destructor

    inline bool operator==(const Link& other) const
    {
      return
        std::tie(
          id_,
          source_node_id_,
          sink_node_id_,
          weight_id_
        ) == std::tie(
          other.id_,
          other.source_node_id_,
          other.sink_node_id_,
          other.weight_id_
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

    void setWeightId(const int& weight_id); ///< weight_id setter
    int getWeightId() const; ///< weight_id getter

private:
    int id_; ///< Link ID
    int source_node_id_; ///< Link source node
    int sink_node_id_; ///< Link sink node
    int weight_id_; ///< Link weight
  };
}

#endif //SMARTPEAK_LINK_H