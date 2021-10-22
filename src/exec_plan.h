#pragma once
#include <iostream>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <parquet/file_reader.h>

#include "arrow/dataset/scanner.h"

#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/util/thread_pool.h"

#include "arrow/compute/api.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/cast.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/test_util.h"
#include "arrow/compute/exec/util.h"
#include "arrow/dataset/dataset.h"
#include "arrow/ipc/api.h"
#include "arrow/pretty_print.h"
#include "arrow/testing/random.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/async_util.h" //AsyncTaskGroup
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/logging.h"
#include "arrow/util/optional.h"
#include "arrow/util/range.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/unreachable.h"
#include "arrow/util/vector.h"

#include "arrow/testing/gtest_util.h"

#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/platform.h"
#include "parquet/properties.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <list>
#include <memory>

#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "exec_node.h"

namespace arrow {
using internal::checked_cast;

namespace compute {

class ExecPlanKernel : public std::enable_shared_from_this<ExecPlanKernel> {
public:
  using NodeVector = std::vector<ExecNodeKernel *>;

  virtual ~ExecPlanKernel() = default;

  ExecContext *exec_context() const { return exec_context_; }

  /// Make an empty exec plan
  static Result<std::shared_ptr<ExecPlanKernel>>
  Make(ExecContext * = default_exec_context());

  ExecNodeKernel *AddNode(std::unique_ptr<ExecNodeKernel> node);

  template <typename Node, typename... Args> Node *EmplaceNode(Args &&...args) {
    std::unique_ptr<Node> node{new Node{std::forward<Args>(args)...}};
    auto out = node.get();
    AddNode(std::move(node));
    return out;
  }

  /// The initial inputs
  const NodeVector &sources() const;

  /// The final outputs
  const NodeVector &sinks() const;

  Status Validate();

  /// \brief Start producing on all nodes
  ///
  /// Nodes are started in reverse topological order, such that any node
  /// is started before all of its inputs.
  Status StartProducing();

  /// \brief Stop producing on all nodes
  ///
  /// Nodes are stopped in topological order, such that any node
  /// is stopped before all of its outputs.
  void StopProducing();

  /// \brief A future which will be marked finished when all nodes have stopped
  /// producing.
  Future<> finished();

  std::string ToString() const;

protected:
  ExecContext *exec_context_;
  explicit ExecPlanKernel(ExecContext *exec_context)
      : exec_context_(exec_context) {}
};

namespace {

struct ExecPlanImpl : public ExecPlanKernel {
  explicit ExecPlanImpl(ExecContext *exec_context)
      : ExecPlanKernel(exec_context) {}

  ~ExecPlanImpl() override {
    std::cerr << "~~~~~~ExecPlanImpl";

    if (started_ && !finished_.is_finished()) {
      std::cerr << "Plan was destroyed before finishing";
      StopProducing();
      finished().Wait();
    }
  }

  ExecNodeKernel *AddNode(std::unique_ptr<ExecNodeKernel> node) {
    if (node->label().empty()) {
      node->SetLabel(std::to_string(auto_label_counter_++));
    }
    if (node->num_inputs() == 0) {
      sources_.push_back(node.get());
    }
    if (node->num_outputs() == 0) {
      sinks_.push_back(node.get());
    }
    nodes_.push_back(std::move(node));
    return nodes_.back().get();
  }

  Status Validate() const {
    if (nodes_.empty()) {
      return Status::Invalid("ExecPlanKernel has no node");
    }
    for (const auto &node : nodes_) {
      RETURN_NOT_OK(node->Validate());
    }
    return Status::OK();
  }

  Status StartProducing() {
    if (started_) {
      return Status::Invalid("restarted ExecPlanKernel");
    }
    started_ = true;

    // producers precede consumers
    sorted_nodes_ = TopoSort();

    std::vector<Future<>> futures;

    Status st = Status::OK();

    using rev_it = std::reverse_iterator<NodeVector::iterator>;
    for (rev_it it(sorted_nodes_.end()), end(sorted_nodes_.begin()); it != end;
         ++it) {
      auto node = *it;

      st = node->StartProducing();
      if (!st.ok()) {
        // Stop nodes that successfully started, in reverse order
        stopped_ = true;
        StopProducingImpl(it.base(), sorted_nodes_.end());
        break;
      }

      futures.push_back(node->finished());
    }

    finished_ = AllFinished(futures);
    return st;
  }

  void StopProducing() {
    DCHECK(started_) << "stopped an ExecPlanKernel which never started";
    stopped_ = true;

    StopProducingImpl(sorted_nodes_.begin(), sorted_nodes_.end());
  }

  template <typename It> void StopProducingImpl(It begin, It end) {
    for (auto it = begin; it != end; ++it) {
      auto node = *it;
      node->StopProducing();
    }
  }

  NodeVector TopoSort() const {
    struct Impl {
      const std::vector<std::unique_ptr<ExecNodeKernel>> &nodes;
      std::unordered_set<ExecNodeKernel *> visited;
      NodeVector sorted;

      explicit Impl(const std::vector<std::unique_ptr<ExecNodeKernel>> &nodes)
          : nodes(nodes) {
        visited.reserve(nodes.size());
        sorted.resize(nodes.size());

        for (const auto &node : nodes) {
          Visit(node.get());
        }

        DCHECK_EQ(visited.size(), nodes.size());
      }

      void Visit(ExecNodeKernel *node) {
        if (visited.count(node) != 0)
          return;

        for (auto input : node->inputs()) {
          // Ensure that producers are inserted before this consumer
          Visit(input);
        }

        sorted[visited.size()] = node;
        visited.insert(node);
      }
    };

    return std::move(Impl{nodes_}.sorted);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "ExecPlanKernel with " << nodes_.size() << " nodes:" << std::endl;
    for (const auto &node : TopoSort()) {
      ss << node->ToString() << std::endl;
    }
    return ss.str();
  }

  Future<> finished_ = Future<>::MakeFinished();
  bool started_ = false, stopped_ = false;
  std::vector<std::unique_ptr<ExecNodeKernel>> nodes_;
  NodeVector sources_, sinks_;
  NodeVector sorted_nodes_;
  uint32_t auto_label_counter_ = 0;
};

ExecPlanImpl *ToDerived(ExecPlanKernel *ptr) {
  return checked_cast<ExecPlanImpl *>(ptr);
}

const ExecPlanImpl *ToDerived(const ExecPlanKernel *ptr) {
  return checked_cast<const ExecPlanImpl *>(ptr);
}

} // namespace

Result<std::shared_ptr<ExecPlanKernel>> ExecPlanKernel::Make(ExecContext *ctx) {
  return std::shared_ptr<ExecPlanKernel>(new ExecPlanImpl{ctx});
}

ExecNodeKernel *ExecPlanKernel::AddNode(std::unique_ptr<ExecNodeKernel> node) {
  return ToDerived(this)->AddNode(std::move(node));
}

const ExecPlanKernel::NodeVector &ExecPlanKernel::sources() const {
  return ToDerived(this)->sources_;
}

const ExecPlanKernel::NodeVector &ExecPlanKernel::sinks() const {
  return ToDerived(this)->sinks_;
}

Status ExecPlanKernel::Validate() { return ToDerived(this)->Validate(); }

Status ExecPlanKernel::StartProducing() {
  return ToDerived(this)->StartProducing();
}

void ExecPlanKernel::StopProducing() { ToDerived(this)->StopProducing(); }

Future<> ExecPlanKernel::finished() { return ToDerived(this)->finished_; }

std::string ExecPlanKernel::ToString() const {
  return ToDerived(this)->ToString();
}

}
}
