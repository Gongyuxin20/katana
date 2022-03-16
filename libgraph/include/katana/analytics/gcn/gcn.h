#ifndef KATANA_LIBGRAPH_KATANA_ANALYTICS_GCN_GCN_H_
#define KATANA_LIBGRAPH_KATANA_ANALYTICS_GCN_GCN_H_

#include <iostream>

#include <katana/analytics/Plan.h>

#include "katana/AtomicHelpers.h"
#include "katana/analytics/Utils.h"

// API

namespace katana::analytics {

/// A computational plan to for gcn, specifying the algorithm and any
/// parameters associated with it.
class GCNPlan : public Plan {
public:
  /// Algorithm selectors for GCN
  enum Algorithm { kSynchronous, kAsynchronous };

  // Don't allow people to directly construct these, so as to have only one
  // consistent way to configure.
private:
  Algorithm algorithm_;

  GCNPlan(Architecture architecture, Algorithm algorithm)
      : Plan(architecture), algorithm_(algorithm) {}

public:
  // kChunkSize is a fixed const int (default value: 64)
  static const int kChunkSize;

  GCNPlan() : GCNPlan{kCPU, kSynchronous} {}

  Algorithm algorithm() const { return algorithm_; }

  /// Synchronous k-core algorithm.
  static GCNPlan Synchronous() { return {kCPU, kSynchronous}; }

  /// Asynchronous k-core algorithm.
  static GCNPlan Asynchronous() { return {kCPU, kAsynchronous}; }
};

/// Compute the gcn for pg. The pg must be symmetric.
/// The algorithm, and k_core_number parameters can be specified,
/// but have reasonable defaults.
/// The property named output_property_name is created by this function and may
/// not exist before the call.
KATANA_EXPORT Result<void> GCN(
    PropertyGraph* pg, std::string& content_file, uint32_t k_core_number,
    const std::string& output_property_name, katana::TxnContext* txn_ctx,
    const bool& is_symmetric = false, GCNPlan plan = GCNPlan());


KATANA_EXPORT Result<void> GCNAssertValid(
    PropertyGraph* pg, uint32_t k_core_number,
    const std::string& property_name);

struct KATANA_EXPORT GCNStatistics {
  /// Total number of node left in the core.
  uint64_t number_of_nodes_in_kcore;

  /// Print the statistics in a human readable form.
  void Print(std::ostream& os = std::cout) const;

  static katana::Result<GCNStatistics> Compute(
      katana::PropertyGraph* pg, uint32_t k_core_number,
      const std::string& property_name);
};

}  // namespace katana::analytics
#endif
