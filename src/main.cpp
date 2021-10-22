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
#include "cache_machine.h"
#include "exec_node.h"
#include "exec_plan.h"

namespace arrow {
using internal::checked_cast;

namespace compute {

/////////////////////////////////////////////////////////

class DataHolderNodeOptions : public ExecNodeOptions {
public:
  explicit DataHolderNodeOptions(bool async_mode = true)
      : async_mode(async_mode) {}

  bool async_mode;
};


class DataHolderPushGenerator {
public:
  using T = std::unique_ptr<DataHolder>;

  struct State {
    util::Mutex mutex;
    std::deque<Result<T>> result_q;
    util::optional<Future<T>> consumer_fut;
    bool finished = false;
  };
public:
  class Producer {
  public:

    explicit Producer(const std::shared_ptr<State>& state) : weak_state_(state) {}

    bool Push(Result<T> result) {
      auto state = weak_state_.lock();
      if(!state) {
        return false;
      }
      auto lock = state->mutex.Lock();
      if(state->finished) {
        return false;
      }
      if(state->consumer_fut.has_value()) {
        auto fut = std::move(state->consumer_fut.value());
        state->consumer_fut.reset();
        lock.Unlock();
        fut.MarkFinished(std::move(result));
      } else {
        state->result_q.push_back(std::move(result));
      }
      return true;
    }

    bool Close() {
      std::cerr << "close producer\n";
      auto state = weak_state_.lock();
      if(!state) {
        return false;
      }
      auto lock = state->mutex.Lock();
      if(state->finished) {
        return false;
      }
      state->finished = true;
      if(state->consumer_fut.has_value()) {
        auto fut = std::move(state->consumer_fut.value());
        state->consumer_fut.reset();
        lock.Unlock();
        fut.MarkFinished(IterationTraits<T>::End());
        std::cerr << " close producer> MarkFinished\n";
      }
      std::cerr << " close producer> TRUE\n";

      return true;
    }
    bool is_closed() const {
      auto state = weak_state_.lock();
      if (!state) {
        // Generator was destroyed
        return true;
      }
      auto lock = state->mutex.Lock();
      return state->finished;
    }
  private:
    const std::weak_ptr<State> weak_state_;
  };
  DataHolderPushGenerator() : state_(std::make_shared<State>()) {}

  Future<T> operator()() const {
    auto lock = state_->mutex.Lock();
    assert(!state_->consumer_fut.has_value());
    if (!state_->result_q.empty()) {
      auto front = std::move(state_->result_q.front());
      auto fut = Future<T>::MakeFinished(std::move(front));
      state_->result_q.pop_front();
      return fut;
    }
    if (state_->finished) {
      return AsyncGeneratorEnd<T>();
    }
    auto fut = Future<T>::Make();
    state_->consumer_fut = fut;
    return fut;
  }

  Producer producer() { return Producer{state_}; }

private:
  const std::shared_ptr<State> state_;
};


class CacheMachine {
public:
  CacheMachine(ExecContext *context) : context_(context), gen_(), producer_(gen_.producer()) {
    this->memory_resources_.push_back(&CPUMemoryResource::getInstance());
    this->memory_resources_.push_back(&DiskMemoryResource::getInstance());
  }

  void push(ExecBatch batch, std::shared_ptr<Schema> schema) {
    int cache_index = 0;
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
          this->producer_.Push(std::move(data_holder));
        } else if (cache_index == 1) {
          //          printf("push>DiskDataHolder\n");
          auto disk_data_holder =
              std::make_unique<DiskDataHolder>(batch, schema);
          this->producer_.Push(std::move(disk_data_holder));
        }
      }
      cache_index++;
    }
  }

  AsyncGenerator<DataHolderPushGenerator::T> generator() {
     return gen_;
  }

public:
  DataHolderPushGenerator gen_;
  DataHolderPushGenerator::Producer producer_;
  std::vector<MemoryResource *> memory_resources_;
  ExecContext *context_;
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
}

void DataHolderNode::Execute() {
//  while (!this->cache_machine_->is_finished()) {
//    std::unique_ptr<DataHolder> data_holder = this->cache_machine_->pull();
//    if (data_holder) {
//      printf("$$$$$$$$$$$$[0]->InputReceivedTask\n");
//      outputs_[0]->InputReceivedTask(
//          this, [](ExecBatch batch) { return batch; },
//          data_holder->Get().ValueOrDie());
//    }
//  }
//  printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2\n");
//  std::cerr << "pool->size() : " << cache_machine_->size() << std::endl;

  auto generator = this->cache_machine_->generator();
  using T =  DataHolderPushGenerator::T;
  struct LoopBody {
    Future<ControlFlow<bool>> operator()() {
      auto next = generator_();
       return next.Then([this](const T& result) -> Result<ControlFlow<bool>> {
        if (IsIterationEnd(result)) {
          printf("$$$$$$$$$$$$ node_->finished_.MarkFinished()\n");

          node_->finished_.MarkFinished();

          return Break(true);
        } else {
//          if (node_->finished_.is_finished()) {
//            printf("$$$$$$$$$$$ node_->finished_.is_finished() \n");
//            return Break(true);
//          }
          printf("$$$$$$$$$$$$[0]->InputReceivedTask\n");
          Result<ExecBatch>  batch = result->Get();
          node_->outputs_[0]->InputReceivedTask(node_, [](ExecBatch batch) { return batch; }, batch.ValueOrDie());
          return Continue();
        }
      });
    }
    AsyncGenerator<T> generator_;
    DataHolderNode* node_;
   };
  Loop(LoopBody{std::move(generator), this});
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
//    if (finish_st.ok()) {
      printf("###cache_machine_->finish\n");
//      this->cache_machine_->finish();
      this->cache_machine_->producer_.Close();

      if (this->run_thread_.joinable()) {
        this->run_thread_.join();
      }
//    }

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

  int num_batches = 10;
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
