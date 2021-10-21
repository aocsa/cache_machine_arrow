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

#include "async.h"
#include "memory_resource.h"

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
    if (started_ && !finished_.is_finished()) {
      ARROW_LOG(WARNING) << "Plan was destroyed before finishing";
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

////////////////////////////////////////////////////////////
using namespace std::chrono_literals;

enum class DataHolderType { CPU, LOCAL_FILE };

struct DataHolder {
  explicit DataHolder(std::shared_ptr<Schema> schema, DataHolderType type)
      : schema_(schema), type_(type) {}

  std::shared_ptr<Schema> schema() const { return schema_; };

  DataHolderType type() const { return type_; };

  virtual size_t SizeInBytes() const = 0;

  virtual Result<ExecBatch> Get() = 0;

private:
  DataHolderType type_;

  std::shared_ptr<Schema> schema_;
};

class CPUDataHolder : public DataHolder {
public:
  explicit CPUDataHolder(const std::shared_ptr<RecordBatch> &batch,
                         const std::shared_ptr<Schema> &schema)
      : DataHolder(schema, DataHolderType::CPU), batch_(std::move(batch)) {
    CPUMemoryResource::getInstance().allocate(SizeInBytes());
  }

  virtual ~CPUDataHolder() {
    CPUMemoryResource::getInstance().deallocate(SizeInBytes());
  }

  virtual size_t SizeInBytes() const override {
    auto record_batch = batch_;
    size_t size_in_bytes = 0;
    for (auto &&column : record_batch->columns()) {
      const auto &data = column->data();
      for (auto &&buffer : data->buffers) {
        if (buffer) {
          size_in_bytes += buffer->size();
        }
      }
    }
    return size_in_bytes;
  }

  Result<ExecBatch> Get() override { return ExecBatch(*batch_); }

private:
  const std::shared_ptr<RecordBatch> batch_;
};

std::string RandomString(std::size_t length) {
  const std::string characters =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<> distribution(0, characters.size() - 1);

  std::string random_string;

  for (std::size_t i = 0; i < length; ++i) {
    random_string += characters[distribution(generator)];
  }

  return random_string;
}

using WriterProperties = parquet::WriterProperties;
using ArrowWriterProperties = parquet::ArrowWriterProperties;

using FileReader = parquet::arrow::FileReader;
using ParquetFileReader = parquet::ParquetFileReader;

#define ABORT_ON_FAILURE(expr)                                                 \
  do {                                                                         \
    arrow::Status status_ = (expr);                                            \
    if (!status_.ok()) {                                                       \
      std::cerr << status_.message() << std::endl;                             \
      abort();                                                                 \
    }                                                                          \
  } while (0);

class DiskDataHolder : public DataHolder {
public:
  explicit DiskDataHolder(ExecBatch batch,
                          const std::shared_ptr<Schema> &schema)
      : DataHolder(schema, DataHolderType::CPU) {
    this->file_path_ = "/tmp/blazing-temp-" + RandomString(64) + ".parquet";

//    printf("DiskDataHolder:Write\n");

    std::shared_ptr<RecordBatch> record_batch =
        batch.ToRecordBatch(schema).ValueOrDie();
    auto table = Table::FromRecordBatches(schema, {record_batch}).ValueOrDie();
    std::shared_ptr<arrow::io::FileOutputStream> outfile;

    PARQUET_ASSIGN_OR_THROW(outfile,
                            arrow::io::FileOutputStream::Open(file_path_));
    ABORT_ON_FAILURE(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile, /*chunk_size=*/2048));
  }
  virtual size_t SizeInBytes() const override {
    struct stat st;

    if (stat(this->file_path_.c_str(), &st) == 0)
      return (st.st_size);
    else
      throw;
  }

  Result<ExecBatch> Get() override {
    printf("DiskDataHolder:Read\n");

    auto pool = ::arrow::default_memory_pool();
    std::unique_ptr<FileReader> arrow_reader;
    Status s = FileReader::Make(
        pool, ParquetFileReader::OpenFile(file_path_, false), &arrow_reader);
    if (s.ok()) {
      std::shared_ptr<::arrow::Table> table;
      arrow_reader->ReadTable(&table);
      arrow::RecordBatchVector filtered_batches;
      TableBatchReader batch_iter(*table);
      while (true) {
        ARROW_ASSIGN_OR_RAISE(auto batch, batch_iter.Next());
        if (batch == nullptr) {
          break;
        }
        return ExecBatch(*batch);
      }
    }
    return Status::IOError("Failed read in DiskDataHolder");
  }

private:
  std::string file_path_;
};

struct Message {
  Message(std::unique_ptr<DataHolder> data_holder, std::string id)
      : data_holder(std::move(data_holder)), id(id) {}
  std::unique_ptr<DataHolder> data_holder;
  std::string id;
};

class MessageQueue {
public:
  MessageQueue() : finished{false} {}
  ~MessageQueue() = default;
  static std::shared_ptr<MessageQueue> Instance();

public:
  void finish() {
    std::unique_lock<std::mutex> lock(mutex_);
    this->finished = true;
    condition_variable_.notify_all();
  }

  bool is_finished() { return this->finished.load(std::memory_order_seq_cst) && this->message_queue_.empty(); }

  std::shared_ptr<Message> pull() {
    std::unique_lock<std::mutex> lock(mutex_);

    while (!condition_variable_.wait_for(lock, 60000ms, [&, this] {
      bool done_waiting = this->finished.load(std::memory_order_seq_cst) or
                          !this->message_queue_.empty();
      return done_waiting;
    })) {

    }
    if (this->message_queue_.size() == 0) {
      return nullptr;
    }
    auto data = std::move(this->message_queue_.front());
    this->message_queue_.pop_front();
    return std::move(data);
  }
  std::shared_ptr<Message> pull(const std::string &id);

  void push(std::shared_ptr<Message> &message);

  size_t size() const { return message_queue_.size(); }

private:
  std::shared_ptr<Message> getMessageQueue(const std::string &id);

private:
  std::mutex mutex_;
  std::atomic<bool> finished;

  std::deque<std::shared_ptr<Message>> message_queue_;
  std::condition_variable condition_variable_;
};

std::shared_ptr<Message> MessageQueue::pull(const std::string &id) {
  std::unique_lock<std::mutex> lock(mutex_);

  while (!condition_variable_.wait_for(lock, 60000ms, [&, this] {
    bool got_the_message =
        std::any_of(this->message_queue_.cbegin(), this->message_queue_.cend(),
                    [&](const auto &e) { return e->id == id; });
    return got_the_message;
  })){}
  return getMessageQueue(id);
}

void MessageQueue::push(std::shared_ptr<Message> &message) {

  printf("\t**MessageQueue::push\n");
  std::unique_lock<std::mutex> lock(mutex_);
  this->message_queue_.push_back(message);
  lock.unlock();
  condition_variable_
      .notify_all(); // Note: Very important to notify all threads
}

std::shared_ptr<Message> MessageQueue::getMessageQueue(const std::string &id) {
  auto it = std::find_if(message_queue_.begin(), message_queue_.end(),
                         [&id](const auto &e) { return e->id == id; });
  assert(it != message_queue_.end());

  std::shared_ptr<Message> message = *it;
  message_queue_.erase(it, it + 1);
  return message;
}

size_t SizeInBytes(ExecBatch batch, std::shared_ptr<Schema> schema) {
  auto record_batch = batch.ToRecordBatch(schema).ValueOrDie();
  size_t size_in_bytes = 0;
  for (auto &&column : record_batch->columns()) {
    const auto &data = column->data();
    for (auto &&buffer : data->buffers) {
      if (buffer) {
        size_in_bytes += buffer->size();
      }
    }
  }
  return size_in_bytes;
}

class CacheMachine {
public:
  CacheMachine(ExecContext *context) : context_(context) {
    pool_ = std::make_unique<MessageQueue>();

    this->memory_resources_.push_back(&CPUMemoryResource::getInstance());
    this->memory_resources_.push_back(&DiskMemoryResource::getInstance());
  }

  void push(ExecBatch batch, std::shared_ptr<Schema> schema) {
    int cache_index = 0;
    std::string message_id = "";
    while (cache_index < this->memory_resources_.size()) {
      auto memory_to_use =
          this->memory_resources_[cache_index]->get_memory_used() +
          SizeInBytes(batch, schema);
      if (memory_to_use <
          this->memory_resources_[cache_index]->get_memory_limit()) {
        if (cache_index == 0) {
          //          printf("push>CPUDataHolder\n");
          auto data_holder = std::make_unique<CPUDataHolder>(
              batch.ToRecordBatch(schema).ValueOrDie(), schema);
          auto item =
              std::make_shared<Message>(std::move(data_holder), message_id);
          this->pool_->push(item);
        } else if (cache_index == 1) {
          //          printf("push>DiskDataHolder\n");
          auto disk_data_holder =
              std::make_unique<DiskDataHolder>(batch, schema);
          auto item = std::make_shared<Message>(std::move(disk_data_holder),
                                                message_id);
          this->pool_->push(item);
        }
      }
      cache_index++;
    }
    this->something_added_ = true;
  }

  std::unique_ptr<DataHolder> pull() {
    std::shared_ptr<Message> message_data = pool_->pull();
    if (message_data == nullptr) {
      return nullptr;
    }
    std::unique_ptr<DataHolder> output = std::move(message_data->data_holder);
    return std::move(output);
  }
  size_t size() const { return pool_->size(); }

  void finish() {
    printf("CacheMachine:finish() \n");
    pool_->finish();
  }
  bool is_finished() { return pool_->is_finished(); }

private:
  std::unique_ptr<MessageQueue> pool_;
  std::vector<MemoryResource *> memory_resources_;

  bool something_added_{false};

  ExecContext *context_;
};
/////////////////////////////////////////////////////////

class DataHolderNodeOptions : public ExecNodeOptions {
public:
  explicit DataHolderNodeOptions(bool async_mode = true)
      : async_mode(async_mode) {}

  bool async_mode;
};

class DataHolderNode : public ExecNodeKernel {
public:
  DataHolderNode(ExecPlanKernel *plan, NodeVector inputs,
                 std::vector<std::string> input_labels,
                 std::shared_ptr<Schema> output_schema, int num_outputs);

  void ErrorReceived(ExecNodeKernel *input, Status error) override {
    DCHECK_EQ(input, inputs_[0]);
    outputs_[0]->ErrorReceived(this, std::move(error));
  }

  void InputFinished(ExecNodeKernel *input, int total_batches) override;

  static Result<ExecNodeKernel *> Make(ExecPlanKernel *plan,
                                       std::vector<ExecNodeKernel *> inputs,
                                       const ExecNodeOptions &options) {

    const auto &data_holder_options =
        checked_cast<const DataHolderNodeOptions &>(options);
    auto schema = inputs[0]->output_schema(); // ???

    return plan->EmplaceNode<DataHolderNode>(
        plan, std::move(inputs), std::vector<std::string>{"target"},
        std::move(schema), /*num_outputs=*/1);
  }

  const char *kind_name() const override { return "DataHolderNode"; }

  void InputReceived(ExecNodeKernel *input, ExecBatch batch) {

    // NO INPUTS()
    throw;
  }

  Status StartProducing() override { return Status::OK(); }

  void PauseProducing(ExecNodeKernel *output) override {}

  void ResumeProducing(ExecNodeKernel *output) override {}

  void StopProducing(ExecNodeKernel *output) override {
    DCHECK_EQ(output, outputs_[0]);
    StopProducing();
  }

  void StopProducing() override {
    if (executor_) {
      this->stop_source_.RequestStop();
    }
    if (input_counter_.Cancel()) {
      this->Finish();
    }
    inputs_[0]->StopProducing(this);
  }

  Future<> finished() override { return finished_; }

  void InputReceivedTask(ExecNodeKernel *input,
                         std::function<Result<ExecBatch>(ExecBatch)> fn,
                         ExecBatch batch) override;

  std::string ToStringExtra() const override { return ""; }

protected:
  void Finish(Status finish_st = Status::OK());

  void Execute();

protected:
  // Counter for the number of batches received
  AtomicCounter input_counter_;

  // Future to sync finished
  Future<> finished_ = Future<>::Make();

  // The task group for the corresponding batches
  util::AsyncTaskGroup task_group_;

  ::arrow::internal::Executor *executor_;

  // Variable used to cancel remaining tasks in the executor
  StopSource stop_source_;

  std::thread run_thread_;

  std::unique_ptr<CacheMachine> cache_machine_;
};

DataHolderNode::DataHolderNode(ExecPlanKernel *plan, NodeVector inputs,
                               std::vector<std::string> input_labels,
                               std::shared_ptr<Schema> output_schema,
                               int num_outputs)
    : ExecNodeKernel(plan, std::move(inputs), input_labels,
                     std::move(output_schema),
                     /*num_outputs=*/num_outputs) {
  executor_ = plan->exec_context()->executor();
  assert(executor_ != nullptr);
  // todo run this into an executor!!
  cache_machine_ = std::make_unique<CacheMachine>(plan->exec_context());

  run_thread_ = std::thread([this] { this->Execute(); });
  this->run_thread_.detach();
}

void DataHolderNode::Execute() {
  while (!this->cache_machine_->is_finished()) {
    std::unique_ptr<DataHolder> data_holder = this->cache_machine_->pull();
    if (data_holder) {
      printf("$$$$$$$$$$$$[0]->InputReceivedTask\n");
      outputs_[0]->InputReceivedTask(
          this, [](ExecBatch batch) { return batch; },
          data_holder->Get().ValueOrDie());
    }
  }
  printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2\n");
  std::cerr << "pool->size() : " << cache_machine_->size() << std::endl;
}

void DataHolderNode::InputReceivedTask(
    ExecNodeKernel *input,
    std::function<Result<ExecBatch>(ExecBatch)> prev_task, ExecBatch batch) {
  Status status;
  if (finished_.is_finished()) {
    return;
  }

  //  auto task = [this, prev_task, batch]() {
  auto output_batch = prev_task(std::move(batch));
  if (ErrorIfNotOk(output_batch.status())) {
    status = output_batch.status();
  } else {
    auto output_batch_unsafe = output_batch.MoveValueUnsafe();
    cache_machine_->push(output_batch_unsafe, this->output_schema_);
    status = Status::OK();
  }
  //  };
  //  status = task();

  //  status = task_group_.AddTask([this, task]() -> Result<Future<>> {
  //    return this->executor_->Submit(this->stop_source_.token(), [this,
  //    task]() {
  //      auto status = task();
  //      if (this->input_counter_.Increment()) {
  //        this->Finish(status);
  //      }
  //      return status;
  //    });
  //  });

  if (this->input_counter_.Increment()) {
    this->Finish(status);
  }

  if (!status.ok()) {
    if (input_counter_.Cancel()) {
      this->Finish(status);
    }
    inputs_[0]->StopProducing(this);
    return;
  }
}

void DataHolderNode::InputFinished(ExecNodeKernel *input, int total_batches) {
  DCHECK_EQ(input, inputs_[0]);
  outputs_[0]->InputFinished(this, total_batches);
  if (input_counter_.SetTotal(total_batches)) {
    this->Finish();
  }
}

void DataHolderNode::Finish(Status finish_st /* = Status::OK()*/) {
  if (executor_) {
    //    task_group_.End().AddCallback([this, finish_st](const Status& st) {
    //      Status final_status = finish_st & st;
    //      printf("DataHolderNode:EndTaskGroup() \n");
    //
    //      this->finished_.MarkFinished(final_status);
    //    });
    if (finish_st.ok()) {
      printf("###cache_machine_->finish\n");
      this->cache_machine_->finish();
    }
    this->finished_.MarkFinished(finish_st);

  } else {
    throw;
  }
}

////////////////////////////////////////////////////////////////////////////////
static bool kEnableExecutablePipelines = true;

struct SourceNode : public ExecNodeKernel {
  SourceNode(ExecPlanKernel *plan, std::shared_ptr<Schema> output_schema,
             AsyncGenerator<util::optional<ExecBatch>> generator)
      : ExecNodeKernel(plan, {}, {}, std::move(output_schema),
                       /*num_outputs=*/1),
        generator_(std::move(generator)) {}

  static Result<ExecNodeKernel *> Make(ExecPlanKernel *plan,
                                       std::vector<ExecNodeKernel *> inputs,
                                       const ExecNodeOptions &options) {

    // RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 0, "SourceNode"));
    const auto &source_options =
        checked_cast<const SourceNodeOptions &>(options);
    return plan->EmplaceNode<SourceNode>(plan, source_options.output_schema,
                                         source_options.generator);
  }

  const char *kind_name() const override { return "SourceNode"; }
  [[noreturn]] static void NoInputs() {
    Unreachable("no inputs; this should never be called");
  }
  [[noreturn]] void InputReceived(ExecNodeKernel *, ExecBatch) override {
    NoInputs();
  }
  [[noreturn]] void ErrorReceived(ExecNodeKernel *, Status) override {
    NoInputs();
  }
  [[noreturn]] void
  InputReceivedTask(ExecNodeKernel *input,
                    std::function<Result<ExecBatch>(ExecBatch)> fn,
                    ExecBatch batch) override {
    NoInputs();
  }

  [[noreturn]] void InputFinished(ExecNodeKernel *, int) override {
    NoInputs();
  }

  Status StartProducing() override {
    DCHECK(!stop_requested_) << "Restarted SourceNode";

    CallbackOptions options;
    auto executor = plan()->exec_context()->executor();
    if (executor) {
      // These options will transfer execution to the desired Executor if
      // necessary. This can happen for in-memory scans where batches didn't
      // require any CPU work to decode. Otherwise, parsing etc should have
      // already been placed us on the desired Executor and no queues will be
      // pushed to.
      options.executor = executor;
      options.should_schedule = ShouldSchedule::IfDifferentExecutor;
    }
    finished_ =
        Loop([this, executor, options] {
          std::unique_lock<std::mutex> lock(mutex_);
          int total_batches = batch_count_++;
          if (stop_requested_) {
            return Future<ControlFlow<int>>::MakeFinished(Break(total_batches));
          }
          lock.unlock();

          return generator_().Then(
              [=](const util::optional<ExecBatch> &maybe_batch)
                  -> ControlFlow<int> {
                std::unique_lock<std::mutex> lock(mutex_);
                if (IsIterationEnd(maybe_batch) || stop_requested_) {
                  stop_requested_ = true;
                  return Break(total_batches);
                }
                lock.unlock();
                ExecBatch batch = std::move(*maybe_batch);

                if (executor) {
                  auto status = task_group_.AddTask(
                      [this, executor, batch]() -> Result<Future<>> {
                        return executor->Submit([=]() {
                          if (kEnableExecutablePipelines) {
                            outputs_[0]->InputReceivedTask(
                                this, [](ExecBatch b) { return b; },
                                std::move(batch));
                          } else {
                            outputs_[0]->InputReceived(this, std::move(batch));
                          }
                          return Status::OK();
                        });
                      });
                  if (!status.ok()) {
                    outputs_[0]->ErrorReceived(this, std::move(status));
                    return Break(total_batches);
                  }
                } else {
                  outputs_[0]->InputReceived(this, std::move(batch));
                }
                return Continue();
              },
              [=](const Status &error) -> ControlFlow<int> {
                // NB: ErrorReceived is independent of InputFinished, but
                // ErrorReceived will usually prompt StopProducing which will
                // prompt InputFinished. ErrorReceived may still be called from
                // a node which was requested to stop (indeed, the request to
                // stop may prompt an error).
                std::unique_lock<std::mutex> lock(mutex_);
                stop_requested_ = true;
                lock.unlock();
                outputs_[0]->ErrorReceived(this, error);
                return Break(total_batches);
              },
              options);
        }).Then([&](int total_batches) {
          outputs_[0]->InputFinished(this, total_batches);
          return task_group_.End();
        });

    return Status::OK();
  }

  void PauseProducing(ExecNodeKernel *output) override {}

  void ResumeProducing(ExecNodeKernel *output) override {}

  void StopProducing(ExecNodeKernel *output) override {
    DCHECK_EQ(output, outputs_[0]);
    StopProducing();
  }

  void StopProducing() override {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_requested_ = true;
  }

  Future<> finished() override { return finished_; }

private:
  std::mutex mutex_;
  bool stop_requested_{false};
  int batch_count_{0};
  Future<> finished_ = Future<>::MakeFinished();
  util::AsyncTaskGroup task_group_;
  AsyncGenerator<util::optional<ExecBatch>> generator_;
};

class FilterNode : public ExecNodeKernel {
public:
  FilterNode(ExecPlanKernel *plan, std::vector<ExecNodeKernel *> inputs,
             std::shared_ptr<Schema> output_schema, Expression filter)
      : ExecNodeKernel(plan, std::move(inputs), /*input_labels=*/{"target"},
                       std::move(output_schema),
                       /*num_outputs=*/1),
        filter_(std::move(filter)) {}

  static Result<ExecNodeKernel *> Make(ExecPlanKernel *plan,
                                       std::vector<ExecNodeKernel *> inputs,
                                       const ExecNodeOptions &options) {
    // RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "FilterNode"));
    auto schema = inputs[0]->output_schema();

    const auto &filter_options =
        checked_cast<const FilterNodeOptions &>(options);

    auto filter_expression = filter_options.filter_expression;
    if (!filter_expression.IsBound()) {
      ARROW_ASSIGN_OR_RAISE(filter_expression, filter_expression.Bind(*schema));
    }

    if (filter_expression.type()->id() != Type::BOOL) {
      return Status::TypeError("Filter expression must evaluate to bool, but ",
                               filter_expression.ToString(), " evaluates to ",
                               filter_expression.type()->ToString());
    }

    return plan->EmplaceNode<FilterNode>(plan, std::move(inputs),
                                         std::move(schema),
                                         std::move(filter_expression));
  }

  const char *kind_name() const override { return "FilterNode"; }

  Result<ExecBatch> DoFilter(const ExecBatch &target) {
    ARROW_ASSIGN_OR_RAISE(Expression simplified_filter,
                          SimplifyWithGuarantee(filter_, target.guarantee));

    ARROW_ASSIGN_OR_RAISE(Datum mask,
                          ExecuteScalarExpression(simplified_filter, target,
                                                  plan()->exec_context()));

    if (mask.is_scalar()) {
      const auto &mask_scalar = mask.scalar_as<BooleanScalar>();
      if (mask_scalar.is_valid && mask_scalar.value) {
        return target;
      }
      return target.Slice(0, 0);
    }

    // if the values are all scalar then the mask must also be
    DCHECK(!std::all_of(target.values.begin(), target.values.end(),
                        [](const Datum &value) { return value.is_scalar(); }));

    auto values = target.values;
    for (auto &value : values) {
      if (value.is_scalar())
        continue;
      ARROW_ASSIGN_OR_RAISE(value,
                            Filter(value, mask, FilterOptions::Defaults()));
    }
    return ExecBatch::Make(std::move(values));
  }

  void InputReceived(ExecNodeKernel *input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    auto maybe_filtered = DoFilter(std::move(batch));
    if (ErrorIfNotOk(maybe_filtered.status()))
      return;

    maybe_filtered->guarantee = batch.guarantee;
    outputs_[0]->InputReceived(this, maybe_filtered.MoveValueUnsafe());
  }

  void InputReceivedTask(ExecNodeKernel *input,
                         std::function<Result<ExecBatch>(ExecBatch)> prev_task,
                         ExecBatch batch) {

    auto task = [this](ExecBatch batch) -> Result<ExecBatch> {
      auto maybe_filtered = DoFilter(std::move(batch));
      if (ErrorIfNotOk(maybe_filtered.status())) {
        return maybe_filtered.status();
      }
      maybe_filtered->guarantee = batch.guarantee;
      return maybe_filtered.MoveValueUnsafe();
    };
    std::function<Result<ExecBatch>(ExecBatch)> func = [batch, prev_task,
                                                        task](ExecBatch input) {
      return task(prev_task(input).ValueOrDie());
    };
    outputs_[0]->InputReceivedTask(this, func, batch);
  }

  void ErrorReceived(ExecNodeKernel *input, Status error) override {
    DCHECK_EQ(input, inputs_[0]);
    outputs_[0]->ErrorReceived(this, std::move(error));
  }

  void InputFinished(ExecNodeKernel *input, int total_batches) override {
    DCHECK_EQ(input, inputs_[0]);
    finished_.MarkFinished();
    outputs_[0]->InputFinished(this, total_batches);
  }

  Status StartProducing() override { return Status::OK(); }

  void PauseProducing(ExecNodeKernel *output) override {}

  void ResumeProducing(ExecNodeKernel *output) override {}

  void StopProducing(ExecNodeKernel *output) override {
    DCHECK_EQ(output, outputs_[0]);
    StopProducing();
  }

  void StopProducing() override { inputs_[0]->StopProducing(this); }

  Future<> finished() override { return finished_; }

protected:
  std::string ToStringExtra() const override {
    return "filter=" + filter_.ToString();
  }

private:
  Expression filter_;
  Future<> finished_ = Future<>::Make();
};

class ProjectNode : public ExecNodeKernel {
public:
  ProjectNode(ExecPlanKernel *plan, std::vector<ExecNodeKernel *> inputs,
              std::shared_ptr<Schema> output_schema,
              std::vector<Expression> exprs)
      : ExecNodeKernel(plan, std::move(inputs), /*input_labels=*/{"target"},
                       std::move(output_schema),
                       /*num_outputs=*/1),
        exprs_(std::move(exprs)) {}

  static Result<ExecNodeKernel *> Make(ExecPlanKernel *plan,
                                       std::vector<ExecNodeKernel *> inputs,
                                       const ExecNodeOptions &options) {
    // RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "ProjectNode"));

    const auto &project_options =
        checked_cast<const ProjectNodeOptions &>(options);
    auto exprs = project_options.expressions;
    auto names = project_options.names;

    if (names.size() == 0) {
      names.resize(exprs.size());
      for (size_t i = 0; i < exprs.size(); ++i) {
        names[i] = exprs[i].ToString();
      }
    }

    FieldVector fields(exprs.size());
    int i = 0;
    for (auto &expr : exprs) {
      if (!expr.IsBound()) {
        ARROW_ASSIGN_OR_RAISE(expr, expr.Bind(*inputs[0]->output_schema()));
      }
      fields[i] = field(std::move(names[i]), expr.type());
      ++i;
    }

    return plan->EmplaceNode<ProjectNode>(
        plan, std::move(inputs), schema(std::move(fields)), std::move(exprs));
  }

  const char *kind_name() const override { return "ProjectNode"; }

  Result<ExecBatch> DoProject(const ExecBatch &target) {
    std::vector<Datum> values{exprs_.size()};
    for (size_t i = 0; i < exprs_.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(Expression simplified_expr,
                            SimplifyWithGuarantee(exprs_[i], target.guarantee));

      ARROW_ASSIGN_OR_RAISE(values[i],
                            ExecuteScalarExpression(simplified_expr, target,
                                                    plan()->exec_context()));
    }
    return ExecBatch{std::move(values), target.length};
  }

  void InputReceived(ExecNodeKernel *input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    auto maybe_projected = DoProject(std::move(batch));
    if (ErrorIfNotOk(maybe_projected.status()))
      return;

    maybe_projected->guarantee = batch.guarantee;
    outputs_[0]->InputReceived(this, maybe_projected.MoveValueUnsafe());
  }

  void InputReceivedTask(ExecNodeKernel *input,
                         std::function<Result<ExecBatch>(ExecBatch)> prev_task,
                         ExecBatch batch) {

    auto task = [this](ExecBatch batch) -> Result<ExecBatch> {
      auto maybe_projected = DoProject(std::move(batch));
      if (ErrorIfNotOk(maybe_projected.status())) {
        return maybe_projected.status();
      }
      maybe_projected->guarantee = batch.guarantee;
      return maybe_projected.MoveValueUnsafe();
    };
    std::function<Result<ExecBatch>(ExecBatch)> func = [batch, prev_task,
                                                        task](ExecBatch input) {
      return task(prev_task(input).ValueOrDie());
    };
    outputs_[0]->InputReceivedTask(this, func, batch);
  }

  void ErrorReceived(ExecNodeKernel *input, Status error) override {
    DCHECK_EQ(input, inputs_[0]);
    outputs_[0]->ErrorReceived(this, std::move(error));
  }

  void InputFinished(ExecNodeKernel *input, int total_batches) override {
    DCHECK_EQ(input, inputs_[0]);
    outputs_[0]->InputFinished(this, total_batches);
  }

  Status StartProducing() override { return Status::OK(); }

  void PauseProducing(ExecNodeKernel *output) override {}

  void ResumeProducing(ExecNodeKernel *output) override {}

  void StopProducing(ExecNodeKernel *output) override {
    DCHECK_EQ(output, outputs_[0]);
    StopProducing();
  }

  void StopProducing() override { inputs_[0]->StopProducing(this); }

  Future<> finished() override { return inputs_[0]->finished(); }

protected:
  std::string ToStringExtra() const override {
    std::stringstream ss;
    ss << "projection=[";
    for (int i = 0; static_cast<size_t>(i) < exprs_.size(); i++) {
      if (i > 0)
        ss << ", ";
      auto repr = exprs_[i].ToString();
      if (repr != output_schema_->field(i)->name()) {
        ss << '"' << output_schema_->field(i)->name() << "\": ";
      }
      ss << repr;
    }
    ss << ']';
    return ss.str();
  }

private:
  std::vector<Expression> exprs_;
};

} // namespace compute
} // namespace arrow

#include "arrow/testing/matchers.h"

using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;
using testing::Optional;
using testing::UnorderedElementsAreArray;

namespace arrow {
namespace compute {

namespace {

class SinkNode : public ExecNodeKernel {
public:
  SinkNode(ExecPlanKernel *plan, std::vector<ExecNodeKernel *> inputs,
           AsyncGenerator<util::optional<ExecBatch>> *generator,
           util::BackpressureOptions backpressure)
      : ExecNodeKernel(plan, std::move(inputs), {"collected"}, {},
                       /*num_outputs=*/0),
        producer_(MakeProducer(generator, std::move(backpressure))) {}

  static Result<ExecNodeKernel *> Make(ExecPlanKernel *plan,
                                       std::vector<ExecNodeKernel *> inputs,
                                       const ExecNodeOptions &options) {
    // RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "SinkNode"));

    const auto &sink_options = checked_cast<const SinkNodeOptions &>(options);
    return plan->EmplaceNode<SinkNode>(plan, std::move(inputs),
                                       sink_options.generator,
                                       sink_options.backpressure);
  }

  static MyPushGenerator<util::optional<ExecBatch>>::Producer
  MakeProducer(AsyncGenerator<util::optional<ExecBatch>> *out_gen,
               util::BackpressureOptions backpressure) {
    MyPushGenerator<util::optional<ExecBatch>> push_gen(
        std::move(backpressure));
    auto out = push_gen.producer();
    *out_gen = std::move(push_gen);
    return out;
  }
  const char *kind_name() const override { return "SinkNode"; }

  Status StartProducing() override {
    finished_ = Future<>::Make();
    return Status::OK();
  }

  // sink nodes have no outputs from which to feel backpressure
  [[noreturn]] static void NoOutputs() {
    Unreachable("no outputs; this should never be called");
  }
  [[noreturn]] void ResumeProducing(ExecNodeKernel *output) override {
    NoOutputs();
  }
  [[noreturn]] void PauseProducing(ExecNodeKernel *output) override {
    NoOutputs();
  }
  [[noreturn]] void StopProducing(ExecNodeKernel *output) override {
    NoOutputs();
  }

  void StopProducing() override {
    Finish();
    inputs_[0]->StopProducing(this);
  }

  Future<> finished() override { return finished_; }

  void InputReceived(ExecNodeKernel *input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    bool did_push = producer_.Push(std::move(batch));
    if (!did_push)
      return; // producer_ was Closed already

    if (input_counter_.Increment()) {
      Finish();
    }
  }

  void InputReceivedTask(ExecNodeKernel *input,
                         std::function<Result<ExecBatch>(ExecBatch)> prev_task,
                         ExecBatch batch) {
    auto task_output = prev_task(batch).ValueOrDie();
    InputReceived(input, task_output);
  }

  void ErrorReceived(ExecNodeKernel *input, Status error) override {
    DCHECK_EQ(input, inputs_[0]);

    producer_.Push(std::move(error));

    if (input_counter_.Cancel()) {
      Finish();
    }
    inputs_[0]->StopProducing(this);
  }

  void InputFinished(ExecNodeKernel *input, int total_batches) override {
    if (input_counter_.SetTotal(total_batches)) {
      Finish();
    }
  }

protected:
  virtual void Finish() {
    if (producer_.Close()) {
      finished_.MarkFinished();
    }
  }

  AtomicCounter input_counter_;
  Future<> finished_ = Future<>::MakeFinished();

  MyPushGenerator<util::optional<ExecBatch>>::Producer producer_;
};

} // namespace

ExecNodeKernel *MakeSourceNode(ExecPlanKernel *plan, std::string label,
                               SourceNodeOptions &options) {
  return SourceNode::Make(plan, {}, options).ValueOrDie();
}

ExecNodeKernel *MakeDataHolderNode(ExecPlanKernel *plan, std::string label,
                                   std::vector<ExecNodeKernel *> inputs,
                                   DataHolderNodeOptions &options) {
  return DataHolderNode::Make(plan, std::move(inputs), options).ValueOrDie();
}

ExecNodeKernel *MakeFilterNode(ExecPlanKernel *plan, std::string label,
                               std::vector<ExecNodeKernel *> inputs,
                               FilterNodeOptions &options) {
  return FilterNode::Make(plan, std::move(inputs), options).ValueOrDie();
}

ExecNodeKernel *MakeSinkNode(ExecPlanKernel *plan, std::string label,
                             std::vector<ExecNodeKernel *> inputs,
                             SinkNodeOptions &options) {
  return SinkNode::Make(plan, std::move(inputs), options).ValueOrDie();
}

constexpr auto kSeed = 0x0ff1ce;

void GenerateBatchesFromSchema(const std::shared_ptr<Schema> &schema,
                               size_t num_batches,
                               BatchesWithSchema *out_batches,
                               int multiplicity = 1, int64_t batch_size = 4) {
  ::arrow::random::RandomArrayGenerator rng_(kSeed);
  if (num_batches == 0) {
    auto empty_record_batch = ExecBatch(*rng_.BatchOf(schema->fields(), 0));
    out_batches->batches.push_back(empty_record_batch);
  } else {
    for (size_t j = 0; j < num_batches; j++) {
      out_batches->batches.push_back(
          ExecBatch(*rng_.BatchOf(schema->fields(), batch_size)));
    }
  }

  size_t batch_count = out_batches->batches.size();
  for (int repeat = 1; repeat < multiplicity; ++repeat) {
    for (size_t i = 0; i < batch_count; ++i) {
      out_batches->batches.push_back(out_batches->batches[i]);
    }
  }
  out_batches->schema = schema;
}

std::shared_ptr<Schema> GetSchema() {
  static std::shared_ptr<Schema> s =
      schema({field("a", int32()), field("b", boolean())});
  return s;
}

RecordBatchVector GenerateBatches(const std::shared_ptr<Schema> &schema,
                                  size_t num_batches, size_t batch_size) {
  BatchesWithSchema input_batches;

  RecordBatchVector batches;
  GenerateBatchesFromSchema(schema, num_batches, &input_batches, 1, batch_size);

  for (const auto &batch : input_batches.batches) {
    batches.push_back(batch.ToRecordBatch(schema).MoveValueUnsafe());
  }
  return batches;
}

BatchesWithSchema MakeBasicBatches(std::shared_ptr<Schema> schema,
                                   RecordBatchVector &batches) {
  BatchesWithSchema out;

  for (auto b : batches) {
    ExecBatch batch(*b);
    out.batches.push_back(batch);
  }
  out.schema = schema;
  return out;
}

void TestStartProducing() {
  std::cerr << "TestStartProducing\n";
  compute::ExecContext exec_context(default_memory_pool(),
                                    ::arrow::internal::GetCpuThreadPool());

  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlanKernel::Make(&exec_context));

  int num_batches = 100;
  int batch_size = 10;
  RecordBatchVector batches =
      ::arrow::compute::GenerateBatches(GetSchema(), num_batches, batch_size);

  auto basic_data = MakeBasicBatches(GetSchema(), batches);
  auto gen = basic_data.gen(/*parallel=*/true, /*slow=*/false);

  constexpr uint32_t kPauseIfAbove = 4;
  constexpr uint32_t kResumeIfBelow = 2;

  util::BackpressureOptions backpressure_options =
      util::BackpressureOptions::Make(kResumeIfBelow, kPauseIfAbove);

  if (backpressure_options.toggle) {
    printf("MakePauseable....\n");
    gen = MakePauseable(gen, backpressure_options.toggle);
  }

  SourceNodeOptions source_options{basic_data.schema, gen};
  auto source = MakeSourceNode(plan.get(), "source", source_options);

  DataHolderNodeOptions data_holder_options;
  auto data_holder = MakeDataHolderNode(plan.get(), "data_holder", {source},
                                        data_holder_options);

  compute::Expression b_is_true = field_ref("b");
  FilterNodeOptions filter_options = compute::FilterNodeOptions{b_is_true};
  auto filter =
      MakeFilterNode(plan.get(), "filter", {data_holder}, filter_options);

  AsyncGenerator<util::optional<ExecBatch>> sink_gen;
  SinkNodeOptions sink_options{&sink_gen, backpressure_options};
  MakeSinkNode(plan.get(), "sink", {filter}, sink_options);

  auto CustomStartAndCollect = [](ExecPlanKernel *plan,
                                  AsyncGenerator<util::optional<ExecBatch>> gen)
      -> Future<std::vector<ExecBatch>> {
    RETURN_NOT_OK(plan->Validate());
    RETURN_NOT_OK(plan->StartProducing());

    auto collected_fut = CollectAsyncGenerator(gen);

    return AllComplete({plan->finished(), Future<>(collected_fut)})
        .Then([collected_fut]() -> Result<std::vector<ExecBatch>> {
          ARROW_ASSIGN_OR_RAISE(auto collected, collected_fut.result());
          return ::arrow::internal::MapVector(
              [](util::optional<ExecBatch> batch) { return std::move(*batch); },
              std::move(collected));
        });
  };
  ASSERT_FINISHES_OK_AND_ASSIGN(auto exec_batches,
                                CustomStartAndCollect(plan.get(), sink_gen));
  plan->finished().Wait();

  //  auto res2 =
  //  Finishes(ResultWith(UnorderedElementsAreArray(basic_data.batches)));
  //  ASSERT_THAT(res1, res2);
  //  SleepABit();
  for (auto &&batch : exec_batches) {
    ::arrow::PrettyPrint(*batch.ToRecordBatch(basic_data.schema).ValueOrDie(),
                         {}, &std::cerr);
    PrintTo(batch, &std::cerr);
  }
}

} // namespace compute
} // namespace arrow

int main(int argc, char *argv[]) {

  //  arrow::internal::RunInSerialExecutor();

  //  arrow::internal::ThreadPoolSubmit();
  CPUMemoryResource::getInstance().initialize(/*host_memory_quota=*/0.75);

  arrow::compute::TestStartProducing();
  return 0;
}
