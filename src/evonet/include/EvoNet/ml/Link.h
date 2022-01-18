/**TODO:  Add copyright*/

#ifndef EVONET_LINK_H
#define EVONET_LINK_H

#include <tuple>
#include <string>

#include <cereal/cereal.hpp>

namespace EvoNet
{

  /**
    @brief Directed Network Link
  */
  class Link
  {
public:
    Link(); ///< Default constructor
    Link(const Link& other); ///< Copy constructor // [TODO: add test]
    Link(const int& id); ///< Explicit constructor
    Link(const std::string& name); ///< Explicit constructor
    Link(const std::string& name,
      const std::string& source_node_name,
      const std::string& sink_node_name,
      const std::string& weight_name); ///< Explicit constructor
    ~Link(); ///< Default destructor

    inline bool operator==(const Link& other) const
    {
      return
        std::tie(
          id_,
          source_node_name_,
          sink_node_name_,
          weight_name_,
          name_,
					module_id_,
					module_name_
        ) == std::tie(
          other.id_,
          other.source_node_name_,
          other.sink_node_name_,
          other.weight_name_,
          other.name_,
					other.module_id_,
					other.module_name_
        )
      ;
    }

    inline bool operator!=(const Link& other) const
    {
      return !(*this == other);
    }

    inline Link& operator=(const Link& other)
    { // [TODO: add test]
      id_ = other.id_;
      name_ = other.name_;
			module_id_ = other.module_id_;
			module_name_ = other.module_name_;
      source_node_name_ = other.source_node_name_;
      sink_node_name_ = other.sink_node_name_;
      weight_name_ = other.weight_name_;
      return *this;
    }

    void setId(const int& id); ///< id setter
    int getId() const; ///< id getter

    void setName(const std::string& name); ///< naem setter
    std::string getName() const; ///< name getter

    void setSourceNodeName(const std::string& source_node_name); ///< source_node_name setter
    std::string getSourceNodeName() const; ///< source_node_name getter

    void setSinkNodeName(const std::string& sink_node_name); ///< sink_node_name setter
    std::string getSinkNodeName() const; ///< sink_node_name getter

    void setWeightName(const std::string& weight_name); ///< weight_name setter
    std::string getWeightName() const; ///< weight_name getter

		void setModuleId(const int& module_id); ///< module id setter
		int getModuleId() const; ///< module id getter

		void setModuleName(const std::string& module_name); ///< module name setter
		std::string getModuleName() const; ///< module name getter

private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(id_, name_, module_id_, module_name_, source_node_name_, sink_node_name_,	weight_name_);
		}
    int id_ = -1; ///< Weight ID
    std::string name_ = ""; ///< Weight Name
		int module_id_ = -1; ///< Module ID
		std::string module_name_ = ""; ///<Module Name
    std::string source_node_name_; ///< Link source node
    std::string sink_node_name_; ///< Link sink node
    std::string weight_name_; ///< Link weight
  };
}

#endif //EVONET_LINK_H