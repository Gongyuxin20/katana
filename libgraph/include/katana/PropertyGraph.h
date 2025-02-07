#ifndef KATANA_LIBGRAPH_KATANA_PROPERTYGRAPH_H_
#define KATANA_LIBGRAPH_KATANA_PROPERTYGRAPH_H_

#include <memory>
#include <utility>

#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/type_traits.h>

#include "katana/ArrowInterchange.h"
#include "katana/Details.h"
#include "katana/EntityIndex.h"
#include "katana/EntityTypeManager.h"
#include "katana/ErrorCode.h"
#include "katana/GraphTopology.h"
#include "katana/Iterators.h"
#include "katana/Logging.h"
#include "katana/NUMAArray.h"
#include "katana/Properties.h"
#include "katana/RDG.h"
#include "katana/RDGTopology.h"
#include "katana/Result.h"
#include "katana/config.h"

namespace katana {

// TODO(amber): find a better place to put this
template <
    typename T,
    typename __Unused = std::enable_if_t<std::is_arithmetic<T>::value>>
auto
ProjectAsArrowArray(const T* buf, const size_t len) noexcept {
  using ArrowDataType = typename arrow::CTypeTraits<T>::ArrowType;
  using ArrowArrayType = arrow::NumericArray<ArrowDataType>;
  return std::make_shared<ArrowArrayType>(len, arrow::Buffer::Wrap(buf, len));
}

template <typename Props>
std::vector<std::string>
DefaultPropertyNames() {
  auto num_tuple_elem = std::tuple_size_v<Props>;
  std::vector<std::string> names(num_tuple_elem);

  for (size_t i = 0; i < names.size(); ++i) {
    names[i] = "Column_" + std::to_string(i);
  }
  return names;
}

/// A property graph is a graph that has properties associated with its nodes
/// and edges. A property has a name and value. Its value may be a primitive
/// type, a list of values or a composition of properties.
///
/// A PropertyGraph is a representation of a property graph that is backed
/// by persistent storage, and it may be a subgraph of a larger, global property
/// graph. Another way to view a PropertyGraph is as a container for node
/// and edge properties that can be serialized.
///
/// The main way to load and store a property graph is via an RDG. An RDG
/// manages the serialization of the various partitions and properties that
/// comprise the physical representation of the logical property graph.
class KATANA_EXPORT PropertyGraph {
  friend class PGViewCache;

  // Regular methods
public:
  using node_iterator = GraphTopology::node_iterator;
  using edge_iterator = GraphTopology::edge_iterator;
  using nodes_range = GraphTopology::nodes_range;
  using edges_range = GraphTopology::edges_range;
  using iterator = GraphTopology::iterator;
  using Node = GraphTopology::Node;
  using Edge = GraphTopology::Edge;

  using EntityTypeIDArray = katana::NUMAArray<EntityTypeID>;

  PropertyGraph(PropertyGraph&& other) = default;
  PropertyGraph& operator=(PropertyGraph&& other) = default;
  virtual ~PropertyGraph();

  /// PropertyView provides a uniform interface when you don't need to
  /// distinguish operating on edge or node properties
  struct ReadOnlyPropertyView {
    const PropertyGraph* const_g;

    std::shared_ptr<arrow::Schema> (PropertyGraph::*loaded_schema_fn)() const;
    std::shared_ptr<arrow::Schema> (PropertyGraph::*full_schema_fn)() const;
    std::shared_ptr<arrow::ChunkedArray> (PropertyGraph::*property_fn_int)(
        int i) const;
    Result<std::shared_ptr<arrow::ChunkedArray>> (
        PropertyGraph::*property_fn_str)(const std::string& str) const;
    Result<URI> (PropertyGraph::*property_storage_fn_str)(
        const std::string& str) const;
    int32_t (PropertyGraph::*property_num_fn)() const;

    std::shared_ptr<arrow::Schema> loaded_schema() const {
      return (const_g->*loaded_schema_fn)();
    }

    std::shared_ptr<arrow::Schema> full_schema() const {
      return (const_g->*full_schema_fn)();
    }

    std::shared_ptr<arrow::ChunkedArray> GetProperty(int i) const {
      return (const_g->*property_fn_int)(i);
    }

    Result<std::shared_ptr<arrow::ChunkedArray>> GetProperty(
        const std::string& str) const {
      return (const_g->*property_fn_str)(str);
    }

    Result<URI> GetPropertyStorageLocation(const std::string& str) const {
      return (const_g->*property_storage_fn_str)(str);
    }

    int32_t GetNumProperties() const { return (const_g->*property_num_fn)(); }

    uint64_t ApproxMemUse() const {
      uint64_t total_mem_use = 0;
      for (int32_t i = 0; i < GetNumProperties(); ++i) {
        const auto& chunked_array = GetProperty(i);
        for (const auto& array : chunked_array->chunks()) {
          total_mem_use += katana::ApproxArrayMemUse(array);
        }
      }
      return total_mem_use;
    }
  };

  struct MutablePropertyView {
    ReadOnlyPropertyView ropv;
    PropertyGraph* g;

    Result<void> (PropertyGraph::*add_properties_fn)(
        const std::shared_ptr<arrow::Table>& props,
        katana::TxnContext* txn_ctx);
    Result<void> (PropertyGraph::*upsert_properties_fn)(
        const std::shared_ptr<arrow::Table>& props,
        katana::TxnContext* txn_ctx);
    Result<void> (PropertyGraph::*remove_property_int)(
        int i, katana::TxnContext* txn_ctx);
    Result<void> (PropertyGraph::*remove_property_str)(
        const std::string& str, katana::TxnContext* txn_ctx);
    Result<void> (PropertyGraph::*ensure_loaded_property_fn)(
        const std::string& str);
    Result<void> (PropertyGraph::*unload_property_fn)(const std::string& str);

    std::shared_ptr<arrow::Schema> loaded_schema() const {
      return ropv.loaded_schema();
    }
    std::shared_ptr<arrow::Schema> full_schema() const {
      return ropv.full_schema();
    }

    std::shared_ptr<arrow::ChunkedArray> GetProperty(int i) const {
      return ropv.GetProperty(i);
    }

    Result<std::shared_ptr<arrow::ChunkedArray>> GetProperty(
        const std::string& str) const {
      return ropv.GetProperty(str);
    }

    int32_t GetNumProperties() const { return ropv.GetNumProperties(); }

    uint64_t ApproxMemUse() const { return ropv.ApproxMemUse(); }

    Result<void> AddProperties(
        const std::shared_ptr<arrow::Table>& props,
        katana::TxnContext* txn_ctx) const {
      return (g->*add_properties_fn)(props, txn_ctx);
    }

    Result<void> UpsertProperties(
        const std::shared_ptr<arrow::Table>& props,
        katana::TxnContext* txn_ctx) const {
      return (g->*upsert_properties_fn)(props, txn_ctx);
    }

    Result<void> RemoveProperty(int i, katana::TxnContext* txn_ctx) const {
      return (g->*remove_property_int)(i, txn_ctx);
    }
    Result<void> RemoveProperty(
        const std::string& str, katana::TxnContext* txn_ctx) const {
      return (g->*remove_property_str)(str, txn_ctx);
    }

    Result<void> EnsurePropertyLoaded(const std::string& str) const {
      return (g->*ensure_loaded_property_fn)(str);
    }

    Result<void> UnloadProperty(const std::string& str) const {
      return (g->*unload_property_fn)(str);
    }
  };

  PropertyGraph() = default;

  // XXX: WARNING: do not add new constructors. Add Make Functions
  PropertyGraph(
      std::unique_ptr<RDGFile>&& rdg_file, RDG&& rdg, GraphTopology&& topo,
      EntityTypeIDArray&& node_entity_type_ids,
      EntityTypeIDArray&& edge_entity_type_ids,
      EntityTypeManager&& node_type_manager,
      EntityTypeManager&& edge_type_manager) noexcept
      : rdg_(std::make_shared<RDG>(std::move(rdg))),
        file_(std::move(rdg_file)),
        node_entity_type_manager_(
            std::make_shared<EntityTypeManager>(std::move(node_type_manager))),
        edge_entity_type_manager_(
            std::make_shared<EntityTypeManager>(std::move(edge_type_manager))),
        node_entity_type_ids_(std::make_shared<EntityTypeIDArray>(
            std::move(node_entity_type_ids))),
        node_entity_data_(node_entity_type_ids_->data()),
        edge_entity_type_ids_(std::make_shared<EntityTypeIDArray>(
            std::move(edge_entity_type_ids))),
        edge_entity_data_(edge_entity_type_ids_->data()),
        pg_view_cache_(std::move(topo)) {
    KATANA_LOG_DEBUG_VASSERT(
        node_entity_type_ids_->size() == NumNodes(),
        "type array size: {}, num nodes: {}", node_entity_type_ids_->size(),
        NumNodes());
    KATANA_LOG_DEBUG_VASSERT(
        edge_entity_type_ids_->size() == NumEdges(),
        "type array size: {}, num edges: {}", edge_entity_type_ids_->size(),
        NumEdges());
  }

  template <typename PGView>
  PGView BuildView() noexcept {
    return pg_view_cache_.BuildView<PGView>(this);
  }

  /// Make a property graph from a constructed RDG. Take ownership of the RDG
  /// and its underlying resources.
  static Result<std::unique_ptr<PropertyGraph>> Make(
      std::unique_ptr<katana::RDGFile> rdg_file, katana::RDG&& rdg,
      katana::TxnContext* txn_ctx);

  /// Make a property graph from an RDG name.
  static Result<std::unique_ptr<PropertyGraph>> Make(
      const std::string& rdg_name, katana::TxnContext* txn_ctx,
      const katana::RDGLoadOptions& opts = katana::RDGLoadOptions());

  /// Make a property graph from an RDG handle
  static Result<std::unique_ptr<PropertyGraph>> Make(
      std::unique_ptr<RDGFile> rdg_handle, katana::TxnContext* txn_ctx,
      const katana::RDGLoadOptions& opts = katana::RDGLoadOptions());

  /// Make a property graph from topology
  // [deprecated("please provide a storage prefix")]
  static Result<std::unique_ptr<PropertyGraph>> Make(
      GraphTopology&& topo_to_assign);

  /// Make a property graph from topology
  static Result<std::unique_ptr<PropertyGraph>> Make(
      const URI& rdg_dir, GraphTopology&& topo_to_assign);

  /// Make a property graph from topology and type arrays
  // [deprecated("please provide a storage prefix")]
  static Result<std::unique_ptr<PropertyGraph>> Make(
      GraphTopology&& topo_to_assign, EntityTypeIDArray&& node_entity_type_ids,
      EntityTypeIDArray&& edge_entity_type_ids,
      EntityTypeManager&& node_type_manager,
      EntityTypeManager&& edge_type_manager);

  /// Make a property graph from topology and type arrays
  static Result<std::unique_ptr<PropertyGraph>> Make(
      const URI& rdg_dir, GraphTopology&& topo_to_assign,
      EntityTypeIDArray&& node_entity_type_ids,
      EntityTypeIDArray&& edge_entity_type_ids,
      EntityTypeManager&& node_type_manager,
      EntityTypeManager&& edge_type_manager);

  static Result<std::unique_ptr<katana::PropertyGraph>> Make(
      const katana::RDGManifest& rdg_manifest,
      const katana::RDGLoadOptions& opts, katana::TxnContext* txn_ctx);

  [[deprecated]] static std::unique_ptr<PropertyGraph> MakeProjectedGraph(
      PropertyGraph& pg, const std::vector<std::string>& node_types,
      const std::vector<std::string>& edge_types);

  /// Make a projected graph from a property graph. Shares state with
  /// the original graph.
  static Result<std::unique_ptr<PropertyGraph>> MakeProjectedGraph(
      PropertyGraph& pg, std::optional<std::vector<std::string>> node_types,
      std::optional<std::vector<std::string>> edge_types);

  /// Make a projected graph from a property graph. Shares state with
  /// the original graph.
  static Result<std::unique_ptr<PropertyGraph>> MakeProjectedGraph(
      PropertyGraph& pg, std::optional<SetOfEntityTypeIDs> node_types,
      std::optional<SetOfEntityTypeIDs> edge_types);

  /// \return A copy of this with the same set of properties. The copy shares no
  ///       state with this.
  Result<std::unique_ptr<PropertyGraph>> Copy(
      katana::TxnContext* txn_ctx) const;

  /// \param node_properties The node properties to copy.
  /// \param edge_properties The edge properties to copy.
  /// \return A copy of this with a subset of the properties. The copy shares no
  ///        state with this.
  Result<std::unique_ptr<PropertyGraph>> Copy(
      const std::vector<std::string>& node_properties,
      const std::vector<std::string>& edge_properties,
      katana::TxnContext* txn_ctx) const;

  /// Construct node & edge EntityTypeIDs from node & edge properties
  /// Also constructs metadata to convert between atomic types and EntityTypeIDs
  /// Assumes all boolean or uint8 properties are atomic types
  /// TODO(roshan) move this to be a part of Make()
  Result<void> ConstructEntityTypeIDs(katana::TxnContext* txn_ctx);

  /// This is an unfortunate hack. Due to some technical debt, we need a way to
  /// modify these arrays in place from outside this class. This style mirrors a
  /// similar hack in GraphTopology and hopefully makes it clear that these
  /// functions should not be used lightly.
  const EntityTypeID* node_type_data() const noexcept {
    return node_entity_data_;
  }
  /// This is an unfortunate hack. Due to some technical debt, we need a way to
  /// modify these arrays in place from outside this class. This style mirrors a
  /// similar hack in GraphTopology and hopefully makes it clear that these
  /// functions should not be used lightly.
  const EntityTypeID* edge_type_data() const noexcept {
    return edge_entity_data_;
  }

  const EntityTypeManager& GetNodeTypeManager() const {
    return *node_entity_type_manager_;
  }

  const EntityTypeManager& GetEdgeTypeManager() const {
    return *edge_entity_type_manager_;
  }

  Result<std::optional<RDKLSHIndexPrimitive>> LoadRDKLSHIndexPrimitive() {
    return rdg_->LoadRDKLSHIndexPrimitive();
  }

  Result<void> WriteRDKLSHIndexPrimitive(RDKLSHIndexPrimitive& index) {
    return rdg_->WriteRDKLSHIndexPrimitive(index);
  }

  Result<std::optional<RDKSubstructureIndexPrimitive>>
  LoadRDKSubstructureIndexPrimitive() {
    return rdg_->LoadRDKSubstructureIndexPrimitive();
  }

  Result<void> WriteRDKSubstructureIndexPrimitive(
      RDKSubstructureIndexPrimitive& index) {
    return rdg_->WriteRDKSubstructureIndexPrimitive(index);
  }

  const std::string& rdg_dir() const { return rdg_->rdg_dir().string(); }

  uint32_t partition_id() const { return rdg_->partition_id(); }

  uint32_t partition_policy_id() const {
    return rdg_->part_metadata().policy_id_;
  }

  /// \returns the current version of the graph (does not access storage)
  Result<uint64_t> CurrentVersion() {
    if (file_ == nullptr) {
      return KATANA_ERROR(katana::ErrorCode::AssertionFailed, "no RDG handle");
    }
    return rdg_->CurrentVersion(*file_);
  }

  /// Create a new storage location for a graph and write everything into it.
  ///
  /// \returns io_error if, for instance, a file already exists
  Result<void> Write(
      const std::string& rdg_name, const std::string& command_line,
      katana::TxnContext* txn_ctx);

  /// Commit updates modified state and re-uses graph components already in storage.
  ///
  /// Like \ref Write(const std::string&, const std::string&) but can only update
  /// parts of the original read location of the graph.
  Result<void> Commit(
      const std::string& command_line, katana::TxnContext* txn_ctx);
  Result<void> WriteView(
      const std::string& command_line, katana::TxnContext* txn_ctx);

  /// Determine if two PropertyGraphs are Equal
  /// THIS IS A TESTING ONLY FUNCTION, DO NOT EXPOSE THIS TO THE USER
  /// when comparing PG in Equals we directly compare all tables in properties
  /// this is potentially buggy. If for example we have:
  /// 1) an "old" graph, modified in "old" software to add type information which is then stored in properties
  /// 2) a "new" graph, modified in "new" software to add identical type information which is then stored in entity type arrays
  ///    and entity type managers
  /// if we then loaded both graphs (1) and (2) in "new" software and compared them, their type information would look identical
  /// but their properties information would differ as the old software added the type information to properties while the new
  /// software did not. The two graphs would be functionally Equal, but this function would say this are not equal
  /// TODO(unknown):(emcginnis) consider breaking the function down into: topology comparison, type comparison, and property comparison. Move pitfall described above alone with the property comparison function
  bool Equals(const PropertyGraph* other) const;
  /// Report the differences between two graphs
  /// THIS IS A TESTING ONLY FUNCTION, DO NOT EXPOSE THIS TO THE USER
  std::string ReportDiff(const PropertyGraph* other) const;

  /// get the schema for loaded node properties
  std::shared_ptr<arrow::Schema> loaded_node_schema() const {
    return rdg_->node_properties()->schema();
  }

  /// get the schema for all node properties (includes unloaded properties)
  std::shared_ptr<arrow::Schema> full_node_schema() const {
    return rdg_->full_node_schema();
  }

  /// get the schema for loaded edge properties
  std::shared_ptr<arrow::Schema> loaded_edge_schema() const {
    return rdg_->edge_properties()->schema();
  }

  /// get the schema for all edge properties (includes unloaded properties)
  std::shared_ptr<arrow::Schema> full_edge_schema() const {
    return rdg_->full_edge_schema();
  }

  /// \returns the number of node atomic types
  size_t GetNumNodeAtomicTypes() const {
    return GetNodeTypeManager().GetNumAtomicTypes();
  }

  /// \returns the number of edge atomic types
  size_t GetNumEdgeAtomicTypes() const {
    return GetEdgeTypeManager().GetNumAtomicTypes();
  }

  /// \returns the number of node entity types (including kUnknownEntityType)
  size_t GetNumNodeEntityTypes() const {
    return GetNodeTypeManager().GetNumEntityTypes();
  }

  /// \returns the number of edge entity types (including kUnknownEntityType)
  size_t GetNumEdgeEntityTypes() const {
    return GetEdgeTypeManager().GetNumEntityTypes();
  }

  /// \returns true iff a node atomic type @param name exists
  /// NB: no node may have a type that intersects with this atomic type
  /// TODO(roshan) build an index for the number of nodes with the type
  bool HasAtomicNodeType(const std::string& name) const {
    return GetNodeTypeManager().HasAtomicType(name);
  }

  /// \returns all atomic node types
  std::vector<std::string> ListAtomicNodeTypes() const {
    return GetNodeTypeManager().ListAtomicTypes();
  }

  /// \returns true iff an edge atomic type with @param name exists
  /// NB: no edge may have a type that intersects with this atomic type
  /// TODO(roshan) build an index for the number of edges with the type
  bool HasAtomicEdgeType(const std::string& name) const {
    return GetEdgeTypeManager().HasAtomicType(name);
  }

  /// \returns all atomic edge types
  std::vector<std::string> ListAtomicEdgeTypes() const {
    return GetEdgeTypeManager().ListAtomicTypes();
  }

  /// \returns true iff a node entity type @param node_entity_type_id exists
  /// NB: even if it exists, it may not be the most specific type for any node
  /// (returns true for kUnknownEntityType)
  bool HasNodeEntityType(EntityTypeID node_entity_type_id) const {
    return GetNodeTypeManager().HasEntityType(node_entity_type_id);
  }

  /// \returns true iff an edge entity type @param node_entity_type_id exists
  /// NB: even if it exists, it may not be the most specific type for any edge
  /// (returns true for kUnknownEntityType)
  bool HasEdgeEntityType(EntityTypeID edge_entity_type_id) const {
    return GetEdgeTypeManager().HasEntityType(edge_entity_type_id);
  }

  /// \returns the node EntityTypeID for an atomic node type with name
  /// @param name
  /// (assumes that the node type exists)
  EntityTypeID GetNodeEntityTypeID(const std::string& name) const {
    return GetNodeTypeManager().GetEntityTypeID(name);
  }

  /// \returns the edge EntityTypeID for an atomic edge type with name
  /// @param name
  /// (assumes that the edge type exists)
  EntityTypeID GetEdgeEntityTypeID(const std::string& name) const {
    return GetEdgeTypeManager().GetEntityTypeID(name);
  }

  /// \returns the name of the atomic type if the node EntityTypeID
  /// @param node_entity_type_id is an atomic type,
  /// nullopt otherwise
  std::optional<std::string> GetNodeAtomicTypeName(
      EntityTypeID node_entity_type_id) const {
    return GetNodeTypeManager().GetAtomicTypeName(node_entity_type_id);
  }

  /// \returns the name of the atomic type if the edge EntityTypeID
  /// @param edge_entity_type_id is an atomic type,
  /// nullopt otherwise
  std::optional<std::string> GetEdgeAtomicTypeName(
      EntityTypeID edge_entity_type_id) const {
    return GetEdgeTypeManager().GetAtomicTypeName(edge_entity_type_id);
  }

  /// \returns the set of node entity types that intersect
  /// the node atomic type @param node_entity_type_id
  /// (assumes that the node atomic type exists)
  const SetOfEntityTypeIDs& GetNodeSupertypes(
      EntityTypeID node_entity_type_id) const {
    return GetNodeTypeManager().GetSupertypes(node_entity_type_id);
  }

  /// \returns the set of edge entity types that intersect
  /// the edge atomic type @param edge_entity_type_id
  /// (assumes that the edge atomic type exists)
  const SetOfEntityTypeIDs& GetEdgeSupertypes(
      EntityTypeID edge_entity_type_id) const {
    return GetEdgeTypeManager().GetSupertypes(edge_entity_type_id);
  }

  /// \returns the set of atomic node types that are intersected
  /// by the node entity type @param node_entity_type_id
  /// (assumes that the node entity type exists)
  const SetOfEntityTypeIDs& GetNodeAtomicSubtypes(
      EntityTypeID node_entity_type_id) const {
    return GetNodeTypeManager().GetAtomicSubtypes(node_entity_type_id);
  }

  /// \returns the set of atomic edge types that are intersected
  /// by the edge entity type @param edge_entity_type_id
  /// (assumes that the edge entity type exists)
  const SetOfEntityTypeIDs& GetEdgeAtomicSubtypes(
      EntityTypeID edge_entity_type_id) const {
    return GetEdgeTypeManager().GetAtomicSubtypes(edge_entity_type_id);
  }

  /// \returns true iff the node type @param sub_type is a
  /// sub-type of the node type @param super_type
  /// (assumes that the sub_type and super_type EntityTypeIDs exists)
  bool IsNodeSubtypeOf(EntityTypeID sub_type, EntityTypeID super_type) const {
    return GetNodeTypeManager().IsSubtypeOf(sub_type, super_type);
  }

  /// \returns true iff the edge type @param sub_type is a
  /// sub-type of the edge type @param super_type
  /// (assumes that the sub_type and super_type EntityTypeIDs exists)
  bool IsEdgeSubtypeOf(EntityTypeID sub_type, EntityTypeID super_type) const {
    return GetEdgeTypeManager().IsSubtypeOf(sub_type, super_type);
  }

  /// \return returns the most specific node entity type for @param node
  EntityTypeID GetTypeOfNode(Node node) const {
    auto idx = GetNodePropertyIndex(node);
    return node_entity_data_[idx];
  }

  /// \return returns the most specific node entity type for @param node
  EntityTypeID GetTypeOfNodeFromPropertyIndex(
      GraphTopology::PropertyIndex prop_index) const {
    return node_entity_data_[prop_index];
  }

  /// \return returns the most specific edge entity type for @param edge
  EntityTypeID GetTypeOfEdgeFromTopoIndex(Edge edge) const {
    auto idx = GetEdgePropertyIndexFromOutEdge(edge);
    return GetTypeOfEdgeFromPropertyIndex(idx);
  }

  /// \return returns the most specific edge entity type for @param edge
  EntityTypeID GetTypeOfEdgeFromPropertyIndex(
      GraphTopology::PropertyIndex prop_index) const {
    return edge_entity_data_[prop_index];
  }

  /// \return true iff the node @param node has the given entity type
  /// @param node_entity_type_id (need not be the most specific type)
  /// (assumes that the node entity type exists)
  bool DoesNodeHaveType(Node node, EntityTypeID node_entity_type_id) const {
    return IsNodeSubtypeOf(node_entity_type_id, GetTypeOfNode(node));
  }

  /// \return true iff the edge @param edge has the given entity type
  /// @param edge_entity_type_id (need not be the most specific type)
  /// (assumes that the edge entity type exists)
  bool DoesEdgeHaveTypeFromTopoIndex(
      Edge edge, EntityTypeID edge_entity_type_id) const {
    return IsEdgeSubtypeOf(
        edge_entity_type_id, GetTypeOfEdgeFromTopoIndex(edge));
  }

  bool DoesEdgeHaveTypeFromPropertyIndex(
      Edge edge, EntityTypeID edge_entity_type_id) const {
    return IsEdgeSubtypeOf(
        edge_entity_type_id, GetTypeOfEdgeFromPropertyIndex(edge));
  }

  // Return type dictated by arrow
  /// Returns the number of node properties
  /// Does not include types managed by the EntityTypeManager
  int32_t GetNumNodeProperties() const {
    return loaded_node_schema()->num_fields();
  }

  /// Returns the number of edge properties
  /// Does not include types managed by the EntityTypeManager
  int32_t GetNumEdgeProperties() const {
    return loaded_edge_schema()->num_fields();
  }

  // num_rows() == NumNodes() (all local nodes)
  std::shared_ptr<arrow::ChunkedArray> GetNodeProperty(int i) const {
    if (i >= rdg_->node_properties()->num_columns()) {
      return nullptr;
    }
    return rdg_->node_properties()->column(i);
  }

  // num_rows() == num_edges() (all local edges)
  std::shared_ptr<arrow::ChunkedArray> GetEdgeProperty(int i) const {
    if (i >= rdg_->edge_properties()->num_columns()) {
      return nullptr;
    }
    return rdg_->edge_properties()->column(i);
  }

  /// \returns true if a node property/type with @param name exists
  bool HasNodeProperty(const std::string& name) const {
    return loaded_node_schema()->GetFieldIndex(name) != -1;
  }

  /// \returns true if an edge property/type with @param name exists
  bool HasEdgeProperty(const std::string& name) const {
    return loaded_edge_schema()->GetFieldIndex(name) != -1;
  }

  /// Get a node property by name.
  ///
  /// \param name The name of the property to get.
  /// \return The property data or NULL if the property is not found.
  Result<std::shared_ptr<arrow::ChunkedArray>> GetNodeProperty(
      const std::string& name) const;

  Result<URI> GetNodePropertyStorageLocation(const std::string& name) const;

  std::string GetNodePropertyName(int32_t i) const {
    return loaded_node_schema()->field(i)->name();
  }

  Result<std::shared_ptr<arrow::ChunkedArray>> GetEdgeProperty(
      const std::string& name) const;

  std::string GetEdgePropertyName(int32_t i) const {
    return loaded_edge_schema()->field(i)->name();
  }

  Result<URI> GetEdgePropertyStorageLocation(const std::string& name) const;

  /// Get a node property by name and cast it to a type.
  ///
  /// \tparam T The type of the property.
  /// \param name The name of the property.
  /// \return The property array or an error if the property does not exist or has a different type.
  template <typename T>
  Result<std::shared_ptr<typename arrow::CTypeTraits<T>::ArrayType>>
  GetNodePropertyTyped(const std::string& name) {
    // TODO(amp): Use KATANA_CHECKED once that doesn't cause CUDA builds to fail.
    auto chunked_array_result = GetNodeProperty(name);
    if (!chunked_array_result) {
      return chunked_array_result.assume_error();
    }
    auto chunked_array = chunked_array_result.assume_value();
    KATANA_LOG_ASSERT(chunked_array);

    auto array =
        std::dynamic_pointer_cast<typename arrow::CTypeTraits<T>::ArrayType>(
            chunked_array->chunk(0));
    if (!array) {
      return KATANA_ERROR(
          katana::ErrorCode::TypeError, "Incorrect arrow::Array type: {}",
          chunked_array->type()->ToString());
    }
    return MakeResult(std::move(array));
  }

  /// Get an edge property by name and cast it to a type.
  ///
  /// \tparam T The type of the property.
  /// \param name The name of the property.
  /// \return The property array or an error if the property does not exist or has a different type.
  template <typename T>
  Result<std::shared_ptr<typename arrow::CTypeTraits<T>::ArrayType>>
  GetEdgePropertyTyped(const std::string& name) {
    // TODO(amp): Use KATANA_CHECKED once that doesn't cause CUDA builds to fail.
    auto chunked_array_result = GetEdgeProperty(name);
    if (!chunked_array_result) {
      return chunked_array_result.assume_error();
    }
    auto chunked_array = chunked_array_result.assume_value();
    KATANA_LOG_ASSERT(chunked_array);

    auto array =
        std::dynamic_pointer_cast<typename arrow::CTypeTraits<T>::ArrayType>(
            chunked_array->chunk(0));
    if (!array) {
      return KATANA_ERROR(
          katana::ErrorCode::TypeError, "Incorrect arrow::Array type: {}",
          chunked_array->type()->ToString());
    }
    return MakeResult(std::move(array));
  }

  //TODO(yan): Add fine-grained control for dropping specific topologies.
  void DropAllTopologies() noexcept {
    return pg_view_cache_.DropAllTopologies();
  }

  const GraphTopology& topology() const noexcept {
    return pg_view_cache_.GetDefaultTopologyRef();
  }

  GraphTopology::PropertyIndex GetEdgePropertyIndexFromOutEdge(
      const Edge& eid) const noexcept;

  GraphTopology::PropertyIndex GetNodePropertyIndex(
      const Node& nid) const noexcept;

  template <typename NodeProps>
  Result<void> ConstructNodeProperties(
      katana::TxnContext* txn_ctx, const std::vector<std::string>& names =
                                       DefaultPropertyNames<NodeProps>()) {
    auto num_nodes = NumOriginalNodes();
    auto res_table =
        IsTransformed()
            ? katana::AllocateTable<NodeProps>(num_nodes, names, node_bitmask_)
            : katana::AllocateTable<NodeProps>(num_nodes, names);
    if (!res_table) {
      return res_table.error();
    }

    return AddNodeProperties(res_table.value(), txn_ctx);
  }

  template <typename EdgeProps>
  Result<void> ConstructEdgeProperties(
      katana::TxnContext* txn_ctx, const std::vector<std::string>& names =
                                       DefaultPropertyNames<EdgeProps>()) {
    auto num_edges = NumOriginalEdges();
    auto res_table =
        IsTransformed()
            ? katana::AllocateTable<EdgeProps>(num_edges, names, edge_bitmask_)
            : katana::AllocateTable<EdgeProps>(num_edges, names);
    if (!res_table) {
      return res_table.error();
    }

    return AddEdgeProperties(res_table.value(), txn_ctx);
  }

  /// Add Node properties that do not exist in the current graph
  Result<void> AddNodeProperties(
      const std::shared_ptr<arrow::Table>& props, katana::TxnContext* txn_ctx);
  /// Add Edge properties that do not exist in the current graph
  Result<void> AddEdgeProperties(
      const std::shared_ptr<arrow::Table>& props, katana::TxnContext* txn_ctx);
  /// If property name exists, replace it, otherwise insert it
  Result<void> UpsertNodeProperties(
      const std::shared_ptr<arrow::Table>& props, katana::TxnContext* txn_ctx);
  /// If property name exists, replace it, otherwise insert it
  Result<void> UpsertEdgeProperties(
      const std::shared_ptr<arrow::Table>& props, katana::TxnContext* txn_ctx);

  Result<void> RemoveNodeProperty(int i, katana::TxnContext* txn_ctx);
  Result<void> RemoveNodeProperty(
      const std::string& prop_name, katana::TxnContext* txn_ctx);

  Result<void> RemoveEdgeProperty(int i, katana::TxnContext* txn_ctx);
  Result<void> RemoveEdgeProperty(
      const std::string& prop_name, katana::TxnContext* txn_ctx);

  /// Write a node property column out to storage and de-allocate the memory
  /// it was using
  Result<void> UnloadNodeProperty(const std::string& prop_name);

  /// Write an edge property column out to storage and de-allocate the
  /// memory it was using
  Result<void> UnloadEdgeProperty(const std::string& prop_name);

  /// Load a node property by name put it in the table at index i
  /// if i is not a valid index, append the column to the end of the table
  Result<void> LoadNodeProperty(const std::string& name, int i = -1);

  /// Load an edge property by name put it in the table at index i
  /// if i is not a valid index, append the column to the end of the table
  Result<void> LoadEdgeProperty(const std::string& name, int i = -1);

  /// Load a node property by name if it is absent and append its column to
  /// the table do nothing otherwise
  Result<void> EnsureNodePropertyLoaded(const std::string& name);

  /// Load an edge property by name if it is absent and append its column to
  /// the table do nothing otherwise
  Result<void> EnsureEdgePropertyLoaded(const std::string& name);

  std::vector<std::string> ListFullNodeProperties() const {
    return rdg_->ListFullNodeProperties();
  }
  std::vector<std::string> ListLoadedNodeProperties() const {
    return rdg_->ListLoadedNodeProperties();
  }
  std::vector<std::string> ListFullEdgeProperties() const {
    return rdg_->ListFullEdgeProperties();
  }
  std::vector<std::string> ListLoadedEdgeProperties() const {
    return rdg_->ListLoadedEdgeProperties();
  }

  /// Remove all node properties
  void DropNodeProperties() { rdg_->DropNodeProperties(); }
  /// Remove all edge properties
  void DropEdgeProperties() { rdg_->DropEdgeProperties(); }

  MutablePropertyView NodeMutablePropertyView() {
    return MutablePropertyView{
        .ropv =
            {
                .const_g = this,
                .loaded_schema_fn = &PropertyGraph::loaded_node_schema,
                .full_schema_fn = &PropertyGraph::full_node_schema,
                .property_fn_int = &PropertyGraph::GetNodeProperty,
                .property_fn_str = &PropertyGraph::GetNodeProperty,
                .property_storage_fn_str =
                    &PropertyGraph::GetNodePropertyStorageLocation,
                .property_num_fn = &PropertyGraph::GetNumNodeProperties,
            },
        .g = this,
        .add_properties_fn = &PropertyGraph::AddNodeProperties,
        .upsert_properties_fn = &PropertyGraph::UpsertNodeProperties,
        .remove_property_int = &PropertyGraph::RemoveNodeProperty,
        .remove_property_str = &PropertyGraph::RemoveNodeProperty,
        .ensure_loaded_property_fn = &PropertyGraph::EnsureNodePropertyLoaded,
        .unload_property_fn = &PropertyGraph::UnloadNodeProperty,
    };
  }

  ReadOnlyPropertyView NodeReadOnlyPropertyView() const {
    return ReadOnlyPropertyView{
        .const_g = this,
        .loaded_schema_fn = &PropertyGraph::loaded_node_schema,
        .full_schema_fn = &PropertyGraph::full_node_schema,
        .property_fn_int = &PropertyGraph::GetNodeProperty,
        .property_fn_str = &PropertyGraph::GetNodeProperty,
        .property_storage_fn_str =
            &PropertyGraph::GetNodePropertyStorageLocation,
        .property_num_fn = &PropertyGraph::GetNumNodeProperties,
    };
  }

  MutablePropertyView EdgeMutablePropertyView() {
    return MutablePropertyView{
        .ropv =
            {
                .const_g = this,
                .loaded_schema_fn = &PropertyGraph::loaded_edge_schema,
                .full_schema_fn = &PropertyGraph::full_edge_schema,
                .property_fn_int = &PropertyGraph::GetEdgeProperty,
                .property_fn_str = &PropertyGraph::GetEdgeProperty,
                .property_storage_fn_str =
                    &PropertyGraph::GetEdgePropertyStorageLocation,
                .property_num_fn = &PropertyGraph::GetNumEdgeProperties,
            },
        .g = this,
        .add_properties_fn = &PropertyGraph::AddEdgeProperties,
        .upsert_properties_fn = &PropertyGraph::UpsertEdgeProperties,
        .remove_property_int = &PropertyGraph::RemoveEdgeProperty,
        .remove_property_str = &PropertyGraph::RemoveEdgeProperty,
        .ensure_loaded_property_fn = &PropertyGraph::EnsureEdgePropertyLoaded,
        .unload_property_fn = &PropertyGraph::UnloadEdgeProperty,
    };
  }

  ReadOnlyPropertyView EdgeReadOnlyPropertyView() const {
    return ReadOnlyPropertyView{
        .const_g = this,
        .loaded_schema_fn = &PropertyGraph::loaded_edge_schema,
        .full_schema_fn = &PropertyGraph::full_edge_schema,
        .property_fn_int = &PropertyGraph::GetEdgeProperty,
        .property_fn_str = &PropertyGraph::GetEdgeProperty,
        .property_storage_fn_str =
            &PropertyGraph::GetEdgePropertyStorageLocation,
        .property_num_fn = &PropertyGraph::GetNumEdgeProperties,
    };
  }

  // Standard container concepts

  node_iterator begin() const { return topology().begin(); }

  node_iterator end() const { return topology().end(); }

  nodes_range Nodes() const noexcept { return topology().Nodes(); }

  edges_range OutEdges() const noexcept { return topology().OutEdges(); }

  /// Gets the edge range of some node.
  ///
  /// \param node node to get the edge range of
  /// \returns iterable edge range for node.
  edges_range OutEdges(Node node) const { return topology().OutEdges(node); }

  /// Return the number of local nodes
  size_t size() const { return topology().size(); }

  bool empty() const { return topology().empty(); }

  /// Return the number of local nodes
  ///  num_nodes in repartitioner is of type LocalNodeID
  uint64_t NumNodes() const { return topology().NumNodes(); }
  /// Return the number of local edges
  uint64_t NumEdges() const { return topology().NumEdges(); }

  /// Gets the destination for an edge.
  ///
  /// @param edge edge iterator to get the destination of
  /// @returns node iterator to the edge destination
  node_iterator OutEdgeDst(const edge_iterator& edge) const {
    auto node_id = topology().OutEdgeDst(*edge);
    return node_iterator(node_id);
  }

  // Creates an index over a node property.
  Result<void> MakeNodeIndex(const std::string& property_name);

  // Delete an existing index over a node property.
  Result<void> DeleteNodeIndex(const std::string& property_name);

  // Creates an index over an edge property.
  Result<void> MakeEdgeIndex(const std::string& property_name);

  // Delete an existing index over an edge property.
  Result<void> DeleteEdgeIndex(const std::string& property_name);

  // Returns the list of node indexes.
  const std::vector<std::shared_ptr<EntityIndex<GraphTopology::Node>>>&
  node_indexes() const {
    return node_indexes_;
  }

  // Returns the list of edge indexes.
  const std::vector<std::shared_ptr<EntityIndex<GraphTopology::Edge>>>&
  edge_indexes() const {
    return edge_indexes_;
  }

  /// Returns true of an index exists for the named node property
  bool HasNodeIndex(const std::string& property_name) const;

  /// Returns the index associated with the named node property.
  ///
  /// The graph retains ownership of the index.
  katana::Result<std::shared_ptr<katana::EntityIndex<GraphTopology::Node>>>
  GetNodeIndex(const std::string& property_name) const;

  /// Returns true of an index exists for the named edge property
  bool HasEdgeIndex(const std::string& property_name) const;

  /// Returns the index associated with the named edge property.
  ///
  /// The graph retains ownership of the index.
  katana::Result<std::shared_ptr<katana::EntityIndex<GraphTopology::Edge>>>
  GetEdgeIndex(const std::string& property_name) const;

protected:
  RDG& rdg() { return *rdg_; }
  const RDG& rdg() const { return *rdg_; }

  GraphTopology::Edge OriginalToTransformedEdgeID(
      GraphTopology::Edge edge) const {
    return IsTransformed() ? original_to_transformed_edges_[edge] : edge;
  }

  GraphTopology::Node OriginalToTransformedNodeID(
      GraphTopology::Node node) const {
    return IsTransformed() ? original_to_transformed_nodes_[node] : node;
  }

  // TODO(Rob): avoid exposing mutable versions of these
  //            members b/c they are NUMAArrays, and there
  //            is a potential issue of who owns and frees
  //            the (memory for) the array.
  //            Alternative solution: provide mutable access
  //            to the array buffer instead of the array
  EntityTypeManager& mutable_node_entity_type_manager() {
    return *node_entity_type_manager_;
  }
  EntityTypeManager& mutable_edge_entity_type_manager() {
    return *edge_entity_type_manager_;
  }
  EntityTypeIDArray& mutable_node_entity_type_ids() {
    return *node_entity_type_ids_;
  }
  EntityTypeIDArray& mutable_edge_entity_type_ids() {
    return *edge_entity_type_ids_;
  }

  const uint8_t* NodeBitmask() noexcept { return node_bitmask_data_.data(); }

  /// Call this constructor to populate projection data.
  PropertyGraph(
      PropertyGraph& parent, GraphTopology&& projected_topo,
      NUMAArray<Node>&& original_to_transformed_nodes,
      NUMAArray<Edge>&& original_to_transformed_edges,
      NUMAArray<uint8_t>&& node_bitmask_data,
      NUMAArray<uint8_t>&& edge_bitmask_data) noexcept
      : rdg_(parent.rdg_),
        file_(parent.file_),
        node_entity_type_manager_(parent.node_entity_type_manager_),
        edge_entity_type_manager_(parent.edge_entity_type_manager_),
        node_entity_type_ids_(parent.node_entity_type_ids_),
        node_entity_data_(node_entity_type_ids_->data()),
        edge_entity_type_ids_(parent.edge_entity_type_ids_),
        edge_entity_data_(edge_entity_type_ids_->data()),
        pg_view_cache_(std::move(projected_topo)),
        parent_{&parent},
        original_to_transformed_nodes_(
            std::move(original_to_transformed_nodes)),
        original_to_transformed_edges_(
            std::move(original_to_transformed_edges)),
        node_bitmask_data_(std::move(node_bitmask_data)),
        edge_bitmask_data_(std::move(edge_bitmask_data)) {
    auto n_nodes = original_to_transformed_nodes_.size();
    node_bitmask_ = std::make_shared<arrow::Buffer>(
        node_bitmask_data_.data(), arrow::BitUtil::BytesForBits(n_nodes));

    auto n_edges = original_to_transformed_edges_.size();
    edge_bitmask_ = std::make_shared<arrow::Buffer>(
        edge_bitmask_data_.data(), arrow::BitUtil::BytesForBits(n_edges));
  }

  bool IsTransformed() const { return parent_ != nullptr; }
  PropertyGraph* Parent() const { return parent_; }

private:
  /// this function creates an empty projection with num_new_nodes nodes
  static std::unique_ptr<PropertyGraph> MakeEmptyEdgeProjectedGraph(
      PropertyGraph& pg, uint32_t num_new_nodes,
      const DynamicBitset& nodes_bitset,
      NUMAArray<Node>&& original_to_projected_nodes_mapping,
      NUMAArray<GraphTopology::PropertyIndex>&&
          projected_to_original_nodes_mapping);

  /// this function creates an empty projection
  static std::unique_ptr<PropertyGraph> MakeEmptyProjectedGraph(
      PropertyGraph& pg, const DynamicBitset& bitset);

  /// Validate performs a sanity check on the the graph after loading
  Result<void> Validate();

  Result<void> DoWriteTopologies();

  Result<void> DoWrite(
      katana::RDGHandle handle, const std::string& command_line,
      katana::RDG::RDGVersioningPolicy versioning_action,
      katana::TxnContext* txn_ctx);

  Result<void> ConductWriteOp(
      const std::string& uri, const std::string& command_line,
      katana::RDG::RDGVersioningPolicy versioning_action,
      katana::TxnContext* txn_ctx);

  Result<void> WriteGraph(
      const std::string& uri, const std::string& command_line,
      katana::TxnContext* txn_ctx);

  Result<void> WriteView(
      const std::string& uri, const std::string& command_line,
      katana::TxnContext* txn_ctx);

  /// Return the number of nodes of the original property graph.
  uint64_t NumOriginalNodes() const {
    return IsTransformed() ? original_to_transformed_nodes_.size() : NumNodes();
  }

  /// Return the number of edges of the original property graph.
  uint64_t NumOriginalEdges() const {
    return IsTransformed() ? original_to_transformed_edges_.size() : NumEdges();
  }

  Result<RDGTopology*> LoadTopology(const RDGTopology& shadow);

  // Data
  std::shared_ptr<katana::RDG> rdg_{std::make_shared<katana::RDG>()};
  std::shared_ptr<katana::RDGFile> file_;

  /// Manages the relations between the node entity types
  std::shared_ptr<EntityTypeManager> node_entity_type_manager_{
      std::make_shared<EntityTypeManager>()};

  /// Manages the relations between the edge entity types
  std::shared_ptr<EntityTypeManager> edge_entity_type_manager_{
      std::make_shared<EntityTypeManager>()};

  /// The node EntityTypeID for each node's most specific type
  std::shared_ptr<EntityTypeIDArray> node_entity_type_ids_;

  // Optimization to avoid shared_ptr indirection when indexing into node_entity_type_ids_
  EntityTypeID* node_entity_data_;

  /// The edge EntityTypeID for each edge's most specific type
  std::shared_ptr<EntityTypeIDArray> edge_entity_type_ids_;

  // Optimization to avoid shared_ptr indirection when indexing into edge_entity_type_ids_
  EntityTypeID* edge_entity_data_;

  // List of node and edge indexes on this graph.
  std::vector<std::shared_ptr<EntityIndex<Node>>> node_indexes_;
  std::vector<std::shared_ptr<EntityIndex<Edge>>> edge_indexes_;

  PGViewCache pg_view_cache_;

  // Transformation related data.
  PropertyGraph* parent_{nullptr};

  NUMAArray<Node> original_to_transformed_nodes_{};
  NUMAArray<Edge> original_to_transformed_edges_{};

  NUMAArray<uint8_t> node_bitmask_data_{};
  std::shared_ptr<arrow::Buffer> node_bitmask_;
  NUMAArray<uint8_t> edge_bitmask_data_{};
  std::shared_ptr<arrow::Buffer> edge_bitmask_;
};

/// SortAllEdgesByDest sorts edges for each node by destination
/// IDs (ascending order).
///
/// Returns the permutation vector (mapping from old
/// indices to the new indices) which results due to  sorting.
KATANA_EXPORT Result<std::unique_ptr<katana::NUMAArray<uint64_t>>>
SortAllEdgesByDest(PropertyGraph* pg);

/// FindEdgeSortedByDest finds the "node_to_find" id in the
/// sorted edgelist of the "node" using binary search.
///
/// This returns the matched edge index if 'node_to_find' is present
/// in the edgelist of 'node' else edge end if 'node_to_find' is not found.
// TODO(amber): make this a method of a sorted topology class in the near future
// TODO(amber): this method should return an edge_iterator
KATANA_EXPORT GraphTopology::Edge FindEdgeSortedByDest(
    const PropertyGraph* graph, GraphTopology::Node node,
    GraphTopology::Node node_to_find);

/// Renumber all nodes in the graph by sorting in the descending
/// order by node degree.
// TODO(amber): this method should return a new sorted topology
KATANA_EXPORT Result<void> SortNodesByDegree(PropertyGraph* pg);

/// Creates in-memory symmetric (or undirected) graph.
///
/// This function creates an symmetric or undirected version of the
/// PropertyGraph topology by adding reverse edges in-memory.
///
/// For each edge (a, b) in the graph, this function will
/// add an additional edge (b, a) except when a == b, in which
/// case, no additional edge is added.
/// The generated symmetric graph may have duplicate edges.
/// \param pg The original property graph
/// \return The new symmetric property graph by adding reverse edges
// TODO(amber): this function should return a new topology
KATANA_EXPORT Result<std::unique_ptr<katana::PropertyGraph>>
CreateSymmetricGraph(PropertyGraph* pg);

/// Creates in-memory transpose graph.
///
/// This function creates transpose version of the
/// PropertyGraph topology by reversing the edges in-memory.
///
/// For each edge (a, b) in the graph, this function will
/// add edge (b, a) without retaining the original edge (a, b) unlike
/// CreateSymmetricGraph.
/// \param topology The original property graph topology
/// \return The new transposed property graph by reversing the edges
// TODO(lhc): hack for bfs-direct-opt
// TODO(amber): this function should return a new topology
KATANA_EXPORT Result<std::unique_ptr<PropertyGraph>>
CreateTransposeGraphTopology(const GraphTopology& topology);

}  // namespace katana

#endif
