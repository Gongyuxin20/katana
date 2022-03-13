/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "katana/analytics/gcn/gcn.h"

#include "katana/ArrowRandomAccessBuilder.h"
#include "katana/Statistics.h"
#include "katana/TypedPropertyGraph.h"
#include "katana/analytics/Utils.h"
#include "katana/PODVector.h"
#include "math.h"
#include "stdlib.h"

using namespace katana::analytics;

const int GCNPlan::kChunkSize = 64;

/*******************************************************************************
 * Functions for running the algorithm
 ******************************************************************************/
//! Node deadness can be derived from current degree and k value, so no field
//! necessary.
struct GCNNodeCurrentDegree : public katana::AtomicPODProperty<uint32_t> {};

struct GCNNodeAlive : public katana::PODProperty<uint32_t> {};

using NodeData = std::tuple<GCNNodeCurrentDegree>;
using EdgeData = std::tuple<>;

std::vector <std::vector < std::vector <double>>> W; // 权重矩阵
std::vector <std::vector < std::vector <double>>> dW; // W对应需要更新的梯度

std::vector <std::vector <double>> y; // label p维
std::vector <std::vector <double>> pred_y; // pred_result p维
std::vector <std::vector <double>> dH1; // H1 处loss m维
std::vector <std::vector <double>> dH2; // H2 处loss p维

std::vector <std::vector < std::vector <double>>> feature; // Hi
std::vector <std::vector < std::vector <double>>> agg_weight; // Zi
std::vector <std::vector < std::vector <double>>> agg; // agg
std::vector <std::vector < std::vector <double>>> w0agg; // g'/W0 after agg
std::vector <std::vector < std::vector <double>>> w0agg_agg; // g'/W0 after agg_agg

void AllocateMemory( uint64_t numNodes)
{
  y.resize(numNodes);
  pred_y.resize(numNodes);
  dH1.resize(numNodes);
  dH2.resize(numNodes);
  feature.resize(numNodes);
  agg_weight.resize(numNodes);
  agg.resize(numNodes);
  w0agg.resize(numNodes);
  w0agg_agg.resize(numNodes);
    
}

void FreeMemory()
{
  y.resize(0);
  pred_y.resize(0);
  dH1.resize(0);
  dH2.resize(0);
  feature.resize(0);
  agg_weight.resize(0);
  agg.resize(0);
  w0agg.resize(0);
  w0agg_agg.resize(0);
    
}

std::vector<double> dot( std::vector<double> v1, std::vector<std::vector<double> > v2)
{
  std::vector<double> v3(v2[0].size());
  double s = 0.0000;
  for (uint h = 0; h < v2[0].size(); h++){
    for (uint j = 0; j < v2.size(); j++){
      s += v1[j] * v2[j][h];
    }
    v3[h] = s;
    s =0.0000;
  }
  return v3;
}

std::vector<double> dot( std::vector<std::vector<double> > v2, std::vector<double> v1)
{
  std::vector<double> v3(v2.size());
  double s = 0.0000;
  for (uint h = 0; h < v2.size(); h++){
    for (uint j = 0; j < v2[0].size(); j++){
      s += v1[j] * v2[h][j];
    }
    v3[h] = s;
    s =0.0000;
  }
  return v3;
}

std::vector<double> ReLU( std::vector<double> v)
{
  // 实现relu激活函数
  std::vector<double> v3(v.size());
  for (uint i = 0; i < v.size(); i++){
    if (v[i] < 0){
      v3[i] = 0;
    }
    else{
      v3[i] = v[i];
    }
  }
  return v3;
}

std::vector<double> softmax(std::vector<double> v)
{
  // 实现softmax分类预测
  std::vector<double> v3(v.size());
  double sum = 0.0000;
  for (uint i = 0; i < v.size(); i++) {
    v3[i] =exp(v[i]);
    sum += exp(v[i]);
  }
  for (uint i = 0; i < v.size(); i++) {
    v3[i] = v3[i]/sum;
  }
  return v3;
}

double accuracy(std::vector<std::vector<double> > y_pred, std::vector<std::vector<double> > y_true)
{
  // 计算准确性
  double sum = y_pred.size();
  katana::GAccumulator<double> GAccumulator_result;
  katana::do_all(
      katana::iterate(static_cast<size_t>(0), y_pred.size()),
      [&](size_t i) {
        double max = 0;
        double max_idx = 0;
        double max_idy = 0;
        for (uint j = 0; j < y_pred[0].size(); j++){
          if (y_pred[i][j] > max){
            max = y_pred[i][j];
            max_idx = j;
          }
          if (y_true[i][j] == 1){
            max_idy = j;
          }
        }
        if (max_idx == max_idy){
          GAccumulator_result += 1.0;
        }
        
      },
      katana::loopname("Accuracy"), katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
    double result = GAccumulator_result.reduce();
    result = result/sum;
    return result;
}

/**
 * Initialize degree fields in graph with current degree. Since symmetric,
 * out edge count is equivalent to in-edge count.
 *
 * @param graph Graph to initialize degrees in
 */
template <typename GraphTy>
void
DegreeCounting(GraphTy* graph) {
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& node) {
        auto& node_current_degree =
            graph->template GetData<GCNNodeCurrentDegree>(node);
        node_current_degree.store(Degree(*graph, node));
      },
      katana::loopname("DegreeCounting"), 
      katana::no_stats());
}

/**
 * Initialize node state
 *
 * @param graph Graph to initialize in
 */
template <typename GraphTy>
void
InitializeNodeState(GraphTy* graph, const std::vector<double>& layers) {
  uint layer = layers.size() -1;
  uint sizeY = 2;
  W.resize(layer);
  dW.resize(layer);
  // using GNode = typename GraphTy::Node;
  // katana::do_all(
  //     katana::iterate(*graph),
  //     [&](const GNode& node) {
  //       auto& node_current_degree =
  //           graph->template GetData<GCNNodeCurrentDegree>(node);
  //       node_current_degree.store(Degree(*graph, node));
  //     },
  //     katana::loopname("DegreeCounting"), 
  //     katana::no_stats());
}

/**
 * Setup initial worklist of dead nodes.
 *
 * @param graph Graph to operate on
 * @param initial_worklist Empty worklist to be filled with dead nodes.
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number.
 */
template <typename GraphTy>
void
SetupInitialWorklist(
    const GraphTy& graph,
    katana::InsertBag<typename GraphTy::Node>& initial_worklist,
    uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(graph),
      [&](const GNode& node) {
        const auto& node_current_degree =
            graph.template GetData<GCNNodeCurrentDegree>(node);
        if (node_current_degree < k_core_number) {
          //! Dead node, add to initial_worklist for processing later.
          initial_worklist.emplace(node);
        }
      },
      katana::loopname("InitialWorklistSetup"), katana::no_stats());
}


/**
 * Starting with initial dead nodes as current worklist; decrement degree;
 * add to next worklist; switch next with current and repeat until worklist
 * is empty (i.e. no more dead nodes).
 *
 * @param graph Graph to operate on
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number
 */
template <typename GraphTy>
void
SyncCascadeGCN(GraphTy* graph, uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  auto current = std::make_unique<katana::InsertBag<GNode>>();
  auto next = std::make_unique<katana::InsertBag<GNode>>();

  //! Setup worklist.
  SetupInitialWorklist(*graph, *next, k_core_number);

  while (!next->empty()) {
    //! Make "next" into current.
    std::swap(current, next);
    next->clear();

    katana::do_all(
        katana::iterate(*current),
        [&](const GNode& dead_node) {
          //! Decrement degree of all neighbors.
          for (auto e : Edges(*graph, dead_node)) {
            auto dest = EdgeDst(*graph, e);
            auto& dest_current_degree =
                graph->template GetData<GCNNodeCurrentDegree>(dest);
            uint32_t old_degree = katana::atomicSub(dest_current_degree, 1u);

            if (old_degree == k_core_number) {
              //! This thread was responsible for putting degree of destination
              //! below threshold; add to worklist.
              next->emplace(dest);
            }
          }
        },
        katana::steal(), katana::chunk_size<GCNPlan::kChunkSize>(),
        katana::loopname("GCN Synchronous"));
  }
}

/**
 * Starting with initial dead nodes, decrement degree and add to worklist
 * as they drop below 'k' threshold until worklist is empty (i.e. no more dead
 * nodes).
 *
 * @param graph Graph to operate on
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number.
 */
template <typename GraphTy>
void
AsyncCascadeGCN(GraphTy* graph, uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  katana::InsertBag<GNode> initial_worklist;
  //! Setup worklist.
  SetupInitialWorklist(*graph, initial_worklist, k_core_number);

  katana::for_each(
      katana::iterate(initial_worklist),
      [&](const GNode& dead_node, auto& ctx) {
        //! Decrement degree of all neighbors.
        for (auto e : Edges(*graph, dead_node)) {
          auto dest = EdgeDst(*graph, e);
          auto& dest_current_degree =
              graph->template GetData<GCNNodeCurrentDegree>(dest);
          uint32_t old_degree = katana::atomicSub(dest_current_degree, 1u);

          if (old_degree == k_core_number) {
            //! This thread was responsible for putting degree of destination
            //! below threshold: add to worklist.
            ctx.push(dest);
          }
        }
      },
      katana::disable_conflict_detection(),
      katana::chunk_size<GCNPlan::kChunkSize>(),
      katana::loopname("GCN Asynchronous"));
}

/**
 * After computation is finished, the nodes left in the core
 * are marked as alive.
 *
 * @param graph Graph to operate on
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number.
 */
template <typename GraphTy>
katana::Result<void>
GCNMarkAliveNodes(GraphTy* graph, uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& node) {
        auto& node_current_degree =
            graph->template GetData<GCNNodeCurrentDegree>(node);
        auto& node_flag = graph->template GetData<GCNNodeAlive>(node);
        node_flag = 1;
        if (node_current_degree < k_core_number) {
          node_flag = 0;
        }
      },
      katana::loopname("GCN Mark Nodes in Core"));
  return katana::ResultSuccess();
}

template <typename GraphTy>
static katana::Result<void>
GCNImpl(GraphTy* graph, GCNPlan algo, uint32_t k_core_number) {
  size_t approxNodeData = 4 * (graph->NumNodes() + graph->NumEdges());
  katana::EnsurePreallocated(8, approxNodeData);
  katana::ReportPageAllocGuard page_alloc;

  //! Intialization of degrees.
  DegreeCounting(graph);

  //! Begins main computation.
  katana::StatTimer exec_time("GCN");

  exec_time.start();

  switch (algo.algorithm()) {
  case GCNPlan::kSynchronous:
    SyncCascadeGCN(graph, k_core_number);
    break;
  case GCNPlan::kAsynchronous:
    AsyncCascadeGCN(graph, k_core_number);
    break;
  default:
    return katana::ErrorCode::AssertionFailed;
  }
  exec_time.stop();

  return katana::ResultSuccess();
}

katana::Result<void>
katana::analytics::GCN(
    katana::PropertyGraph* pg, uint32_t k_core_number,
    const std::string& output_property_name, katana::TxnContext* txn_ctx,
    const bool& is_symmetric, GCNPlan plan) {
  katana::analytics::TemporaryPropertyGuard temporary_property{
      pg->NodeMutablePropertyView()};

  KATANA_CHECKED(
      pg->ConstructNodeProperties<std::tuple<GCNNodeCurrentDegree>>(
          txn_ctx, {temporary_property.name()}));

  if (is_symmetric) {
    using Graph = katana::TypedPropertyGraphView<
        katana::PropertyGraphViews::Default, NodeData, EdgeData>;
    Graph graph =
        KATANA_CHECKED(Graph::Make(pg, {temporary_property.name()}, {}));

    KATANA_CHECKED(GCNImpl(&graph, plan, k_core_number));
  } else {
    using Graph = katana::TypedPropertyGraphView<
        katana::PropertyGraphViews::Undirected, NodeData, EdgeData>;

    Graph graph =
        KATANA_CHECKED(Graph::Make(pg, {temporary_property.name()}, {}));

    KATANA_CHECKED(GCNImpl(&graph, plan, k_core_number));
  }
  // Post processing. Mark alive nodes.
  KATANA_CHECKED(pg->ConstructNodeProperties<std::tuple<GCNNodeAlive>>(
      txn_ctx, {output_property_name}));

  using GraphTy = katana::TypedPropertyGraph<
      std::tuple<GCNNodeAlive, GCNNodeCurrentDegree>, std::tuple<>>;
  auto graph_final = KATANA_CHECKED(
      GraphTy::Make(pg, {output_property_name, temporary_property.name()}, {}));

  return GCNMarkAliveNodes(&graph_final, k_core_number);
}

// Doxygen doesn't correctly handle implementation annotations that do not
// appear in the declaration.
/// \cond DO_NOT_DOCUMENT
// TODO (gill) Add a validity routine.
katana::Result<void>
katana::analytics::GCNAssertValid(
    [[maybe_unused]] katana::PropertyGraph* pg,
    [[maybe_unused]] uint32_t k_core_number,
    [[maybe_unused]] const std::string& property_name) {
  return katana::ResultSuccess();
}

katana::Result<GCNStatistics>
katana::analytics::GCNStatistics::Compute(
    katana::PropertyGraph* pg, [[maybe_unused]] uint32_t k_core_number,
    const std::string& property_name) {
  using Graph =
      katana::TypedPropertyGraph<std::tuple<GCNNodeAlive>, std::tuple<>>;
  using GNode = Graph::Node;
  auto pg_result = Graph::Make(pg, {property_name}, {});
  if (!pg_result) {
    return pg_result.error();
  }

  auto graph = pg_result.value();

  katana::GAccumulator<uint32_t> alive_nodes;
  alive_nodes.reset();

  katana::do_all(
      katana::iterate(graph),
      [&](const GNode& node) {
        auto& node_alive = graph.GetData<GCNNodeAlive>(node);
        if (node_alive) {
          alive_nodes += 1;
        }
      },
      katana::loopname("GCN sanity check"), katana::no_stats());

  return GCNStatistics{alive_nodes.reduce()};
}
/// \endcond DO_NOT_DOCUMENT

void
katana::analytics::GCNStatistics::Print(std::ostream& os) const {
  os << "Number of nodes in the core = " << number_of_nodes_in_kcore
     << std::endl;
}
