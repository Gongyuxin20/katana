/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause
 * BSD License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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

#include <iostream>

#include <katana/analytics/gcn/gcn.h>

#include "Lonestar/BoilerPlate.h"

using namespace katana::analytics;

constexpr static const char* const name = "gcn";
constexpr static const char* const desc =
    "calculate results with gcn algorithm ";
static const char* url = "gcn";
/*******************************************************************************
 * Declaration of command line arguments
 ******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<std::string> inputFile(
    cll::Positional, cll::desc("<input file>"), cll::Required);

//! Choose algorithm: worklist vs. sync.
static cll::opt<GCNPlan::Algorithm> algo(
    "algo", cll::desc("Choose an algorithm (default value SyncTile):"),
    cll::values(
        clEnumValN(
            GCNPlan::kSynchronous, "Synchronous", "Synchronous algorithm"),
        clEnumValN(
            GCNPlan::kAsynchronous, "Asynchronous",
            "Asynchronous algorithm")),
    cll::init(GCNPlan::kSynchronous));

//! Required k specification for k-core.
static cll::opt<uint32_t> kCoreNumber(
    "kCoreNumber",
    cll::desc("kCoreNumber value: Each node is expected to have out-degree >= "
              "kCoreNumber value (default value 10)"),
    cll::init(10));

std::string
AlgorithmName(GCNPlan::Algorithm algorithm) {
  switch (algorithm) {
  case GCNPlan::kSynchronous:
    return "Synchronous";
  case GCNPlan::kAsynchronous:
    return "Asynchronous";
  default:
    return "Unknown";
  }
}

int
main(int argc, char** argv) {
  std::unique_ptr<katana::SharedMemSys> G =
      LonestarStart(argc, argv, name, desc, url, &inputFile);

  katana::StatTimer total_timer("TimerTotal");
  total_timer.start();

  if (symmetricGraph) {
    KATANA_LOG_WARN(
        "This application requires a symmetric graph input;"
        " Using the -symmetricGraph flag "
        " indicates that the input is a symmetric graph and can be used as it "
        "is.");
  }

  std::cout << "Reading from file: " << inputFile << "\n";
  std::unique_ptr<katana::PropertyGraph> pg =
      MakeFileGraph(inputFile, edge_property_name);

  std::cout << "Read " << pg->topology().NumNodes() << " nodes, "
            << pg->topology().NumEdges() << " edges\n";

  std::cout << "Running " << AlgorithmName(algo) << "\n";

  std::unique_ptr<katana::PropertyGraph> pg_projected_view =
      ProjectPropertyGraphForArguments(pg);

  std::cout << "Projected graph has: "
            << pg_projected_view->topology().NumNodes() << " nodes, "
            << pg_projected_view->topology().NumEdges() << " edges\n";

  GCNPlan plan = GCNPlan();
  switch (algo) {
  case GCNPlan::kSynchronous:
    plan = GCNPlan::Synchronous();
    break;
  case GCNPlan::kAsynchronous:
    plan = GCNPlan::Asynchronous();
    break;
  default:
    KATANA_LOG_FATAL("Invalid algorithm");
  }

  katana::TxnContext txn_ctx;
  if (auto r = GCN(
          pg_projected_view.get(), kCoreNumber, "node-in-core", &txn_ctx,
          symmetricGraph, plan);
      !r) {
    KATANA_LOG_FATAL("Failed to compute k-core: {}", r.error());
  }

  auto stats_result = GCNStatistics::Compute(
      pg_projected_view.get(), kCoreNumber, "node-in-core");
  if (!stats_result) {
    KATANA_LOG_FATAL(
        "Failed to compute GCN statistics: {}", stats_result.error());
  }
  auto stats = stats_result.value();
  stats.Print();

  if (!skipVerify) {
    if (GCNAssertValid(
            pg_projected_view.get(), kCoreNumber, "node-in-core")) {
      std::cout << "Verification successful.\n";
    } else {
      KATANA_LOG_FATAL("verification failed");
    }
  }

  if (output) {
    auto r = pg_projected_view->GetNodePropertyTyped<uint32_t>("node-in-core");
    if (!r) {
      KATANA_LOG_FATAL("Failed to get node property {}", r.error());
    }
    auto results = r.value();
    KATANA_LOG_DEBUG_ASSERT(
        uint64_t(results->length()) ==
        pg_projected_view->topology().NumNodes());

    writeOutput(outputLocation, results->raw_values(), results->length());
  }

  total_timer.stop();

  return 0;
}
