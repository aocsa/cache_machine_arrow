#pragma once

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

namespace arrow {
using internal::checked_cast;

namespace compute {

class ExecPlanKernel;

class ExecNodeKernel {
public:
  using NodeVector = std::vector<ExecNodeKernel *>;

  virtual ~ExecNodeKernel() = default;

  virtual const char *kind_name() const = 0;

  // The number of inputs/outputs expected by this node
  int num_inputs() const { return static_cast<int>(inputs_.size()); }
  int num_outputs() const { return num_outputs_; }

  /// This node's predecessors in the exec plan
  const NodeVector &inputs() const { return inputs_; }

  /// \brief Labels identifying the function of each input.
  const std::vector<std::string> &input_labels() const { return input_labels_; }

  /// This node's successors in the exec plan
  const NodeVector &outputs() const { return outputs_; }

  /// The datatypes for batches produced by this node
  const std::shared_ptr<Schema> &output_schema() const {
    return output_schema_;
  }

  /// This node's exec plan
  ExecPlanKernel *plan() { return plan_; }

  /// \brief An optional label, for display and debugging
  ///
  /// There is no guarantee that this value is non-empty or unique.
  const std::string &label() const { return label_; }
  void SetLabel(std::string label) { label_ = std::move(label); }

  Status Validate() const;

  /// Upstream API:
  /// These functions are called by input nodes that want to inform this node
  /// about an updated condition (a new input batch, an error, an impeding
  /// end of stream).
  ///
  /// Implementation rules:
  /// - these may be called anytime after StartProducing() has succeeded
  ///   (and even during or after StopProducing())
  /// - these may be called concurrently
  /// - these are allowed to call back into PauseProducing(), ResumeProducing()
  ///   and StopProducing()

  /// Transfer input batch to ExecNodeKernel
  virtual void InputReceived(ExecNodeKernel *input, ExecBatch batch) = 0;

  //  virtual void SubmitTask(std::function<Result<ExecBatch>(ExecBatch)>
  //  map_fn, ExecBatch batch) = 0;

  /// Signal error to ExecNodeKernel
  virtual void ErrorReceived(ExecNodeKernel *input, Status error) = 0;

  /// Mark the inputs finished after the given number of batches.
  ///
  /// This may be called before all inputs are received.  This simply fixes
  /// the total number of incoming batches for an input, so that the
  /// ExecNodeKernel knows when it has received all input, regardless of order.
  virtual void InputFinished(ExecNodeKernel *input, int total_batches) = 0;

  /// Lifecycle API:
  /// - start / stop to initiate and terminate production
  /// - pause / resume to apply backpressure
  ///
  /// Implementation rules:
  /// - StartProducing() should not recurse into the inputs, as it is
  ///   handled by ExecPlanKernel::StartProducing()
  /// - PauseProducing(), ResumeProducing(), StopProducing() may be called
  ///   concurrently (but only after StartProducing() has returned successfully)
  /// - PauseProducing(), ResumeProducing(), StopProducing() may be called
  ///   by the downstream nodes' InputReceived(), ErrorReceived(),
  ///   InputFinished() methods
  /// - StopProducing() should recurse into the inputs
  /// - StopProducing() must be idempotent

  // XXX What happens if StartProducing() calls an output's InputReceived()
  // synchronously, and InputReceived() decides to call back into
  // StopProducing() (or PauseProducing()) because it received enough data?
  //
  // Right now, since synchronous calls happen in both directions (input to
  // output and then output to input), a node must be careful to be reentrant
  // against synchronous calls from its output, *and* also concurrent calls from
  // other threads.  The most reliable solution is to update the internal state
  // first, and notify outputs only at the end.
  //
  // Alternate rules:
  // - StartProducing(), ResumeProducing() can call synchronously into
  //   its ouputs' consuming methods (InputReceived() etc.)
  // - InputReceived(), ErrorReceived(), InputFinished() can call asynchronously
  //   into its inputs' PauseProducing(), StopProducing()
  //
  // Alternate API:
  // - InputReceived(), ErrorReceived(), InputFinished() return a ProductionHint
  //   enum: either None (default), PauseProducing, ResumeProducing,
  //   StopProducing
  // - A method allows passing a ProductionHint asynchronously from an output
  // node
  //   (replacing PauseProducing(), ResumeProducing(), StopProducing())

  /// \brief Start producing
  ///
  /// This must only be called once.  If this fails, then other lifecycle
  /// methods must not be called.
  ///
  /// This is typically called automatically by
  /// ExecPlanKernel::StartProducing().
  virtual Status StartProducing() = 0;

  /// \brief Pause producing temporarily
  ///
  /// This call is a hint that an output node is currently not willing
  /// to receive data.
  ///
  /// This may be called any number of times after StartProducing() succeeds.
  /// However, the node is still free to produce data (which may be difficult
  /// to prevent anyway if data is produced using multiple threads).
  virtual void PauseProducing(ExecNodeKernel *output) = 0;

  /// \brief Resume producing after a temporary pause
  ///
  /// This call is a hint that an output node is willing to receive data again.
  ///
  /// This may be called any number of times after StartProducing() succeeds.
  /// This may also be called concurrently with PauseProducing(), which suggests
  /// the implementation may use an atomic counter.
  virtual void ResumeProducing(ExecNodeKernel *output) = 0;

  /// \brief Stop producing definitively to a single output
  ///
  /// This call is a hint that an output node has completed and is not willing
  /// to receive any further data.
  virtual void StopProducing(ExecNodeKernel *output) = 0;

  /// \brief Stop producing definitively to all outputs
  virtual void StopProducing() = 0;

  /// \brief A future which will be marked finished when this node has stopped
  /// producing.
  virtual Future<> finished() = 0;

  std::string ToString() const;

  virtual void InputReceivedTask(ExecNodeKernel *input,
                                 std::function<Result<ExecBatch>(ExecBatch)> fn,
                                 ExecBatch batch) = 0;

protected:
  ExecNodeKernel(ExecPlanKernel *plan, NodeVector inputs,
                 std::vector<std::string> input_labels,
                 std::shared_ptr<Schema> output_schema, int num_outputs);

  // A helper method to send an error status to all outputs.
  // Returns true if the status was an error.
  bool ErrorIfNotOk(Status status);

  /// Provide extra info to include in the string representation.
  virtual std::string ToStringExtra() const;

  ExecPlanKernel *plan_;
  std::string label_;

  NodeVector inputs_;
  std::vector<std::string> input_labels_;

  std::shared_ptr<Schema> output_schema_;
  int num_outputs_;
  NodeVector outputs_;
};

util::optional<int> GetNodeIndex(const std::vector<ExecNodeKernel *> &nodes,
                                 const ExecNodeKernel *node) {
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    if (nodes[i] == node)
      return i;
  }
  return util::nullopt;
}

ExecNodeKernel::ExecNodeKernel(ExecPlanKernel *plan, NodeVector inputs,
                               std::vector<std::string> input_labels,
                               std::shared_ptr<Schema> output_schema,
                               int num_outputs)
    : plan_(plan), inputs_(std::move(inputs)),
      input_labels_(std::move(input_labels)),
      output_schema_(std::move(output_schema)), num_outputs_(num_outputs) {
  for (auto input : inputs_) {
    input->outputs_.push_back(this);
  }
}

Status ExecNodeKernel::Validate() const {
  if (inputs_.size() != input_labels_.size()) {
    return Status::Invalid("Invalid number of inputs for '", label(),
                           "' (expected ", num_inputs(), ", actual ",
                           input_labels_.size(), ")");
  }

  if (static_cast<int>(outputs_.size()) != num_outputs_) {
    return Status::Invalid("Invalid number of outputs for '", label(),
                           "' (expected ", num_outputs(), ", actual ",
                           outputs_.size(), ")");
  }

  for (auto out : outputs_) {
    auto input_index = GetNodeIndex(out->inputs(), this);
    if (!input_index) {
      return Status::Invalid("Node '", label(), "' outputs to node '",
                             out->label(), "' but is not listed as an input.");
    }
  }

  return Status::OK();
}

std::string ExecNodeKernel::ToString() const {
  std::stringstream ss;
  ss << kind_name() << "{\"" << label_ << '"';
  if (!inputs_.empty()) {
    ss << ", inputs=[";
    for (size_t i = 0; i < inputs_.size(); i++) {
      if (i > 0)
        ss << ", ";
      ss << input_labels_[i] << ": \"" << inputs_[i]->label() << '"';
    }
    ss << ']';
  }

  if (!outputs_.empty()) {
    ss << ", outputs=[";
    for (size_t i = 0; i < outputs_.size(); i++) {
      if (i > 0)
        ss << ", ";
      ss << "\"" << outputs_[i]->label() << "\"";
    }
    ss << ']';
  }

  const std::string extra = ToStringExtra();
  if (!extra.empty())
    ss << ", " << extra;

  ss << '}';
  return ss.str();
}

std::string ExecNodeKernel::ToStringExtra() const { return ""; }

bool ExecNodeKernel::ErrorIfNotOk(Status status) {
  if (status.ok())
    return false;

  for (auto out : outputs_) {
    out->ErrorReceived(this,
                       out == outputs_.back() ? std::move(status) : status);
  }
  return true;
}

}
}