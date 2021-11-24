#include <iostream>

#include "async.h"
#include "data_holder_manager.h"
#include "exec_node.h"
#include "exec_plan.h"

namespace arrow {
using internal::checked_cast;

namespace compute {

class DataHolderNodeOptions : public ExecNodeOptions {
 public:
  explicit DataHolderNodeOptions(bool async_mode = true)
      : async_mode(async_mode) {}

  bool async_mode;
};

// Status Schedule(std::function<Status(ExecBatch)> task, ExecBatch batch,
//                int priority) {}

class DataHolderNode : public ExecNodeKernel {
 public:
  DataHolderNode(ExecPlanKernel *plan, NodeVector inputs,
                 std::vector<std::string> input_labels,
                 std::shared_ptr<Schema> output_schema, int num_outputs);

  virtual ~DataHolderNode();

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
    auto schema = inputs[0]->output_schema();  // ???

    return plan->EmplaceNode<DataHolderNode>(
        plan, std::move(inputs), std::vector<std::string>{"target"},
        std::move(schema), /*num_outputs=*/1);
  }

  const char *kind_name() const override { return "DataHolderNode"; }

  void InputReceived(ExecNodeKernel *input, ExecBatch batch) override;

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

  std::unique_ptr<DataHolderManager> data_holder_manager_;
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
  data_holder_manager_ = std::make_unique<DataHolderManager>(
      (DataHolderExecContext *)plan->exec_context());

  auto status = task_group_.AddTask([this]() -> Result<Future<>> {
    return executor_->Submit(this->stop_source_.token(), [this] {
      auto generator = this->data_holder_manager_->generator();
      using T = std::unique_ptr<DataHolder>;
      struct LoopBody {
        Future<ControlFlow<bool>> operator()() {
          auto next = generator_();
          return next.Then(
              [this](const T &result) -> Result<ControlFlow<bool>> {
                if (IsIterationEnd(result)) {
                  node_->finished_.MarkFinished();
                  return Break(true);
                } else {
                  Result<ExecBatch> batch = result->Get();
                  node_->outputs_[0]->InputReceived(node_, batch.ValueOrDie());
                  return Continue();
                }
              });
        }
        AsyncGenerator<T> generator_;
        DataHolderNode *node_;
      };
      auto future = Loop(LoopBody{std::move(generator), this});
      auto ret = future.result();
      return Status::OK();
    });
  });
  if (!status.ok()) {
    if (input_counter_.Cancel()) {
      this->Finish(status);
    }
    inputs_[0]->StopProducing(this);
  }
}

DataHolderNode::~DataHolderNode() {}

void DataHolderNode::InputReceived(ExecNodeKernel *input, ExecBatch batch) {
  Status status;
  if (finished_.is_finished()) {
    return;
  }

  auto task = [this, batch]() {
    auto output_batch = batch;
    Status status;
    auto output_batch_unsafe = output_batch;
    data_holder_manager_->Push(output_batch_unsafe, this->output_schema_);
    status = Status::OK();
    return status;
  };
  status = task_group_.AddTask([this, task]() -> Result<Future<>> {
    return this->executor_->Submit(this->stop_source_.token(), [this, task]() {
      auto status = task();
      if (this->input_counter_.Increment()) {
        this->Finish(status);
      }
      return status;
    });
  });

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
    printf("###data_holder_manager_->finish\n");
    //      this->data_holder_manager_->finish();
    this->data_holder_manager_->producer_.Close();

    //    }

  } else {
    throw;
  }
}

////////////////////////////////////////////////////////////////////////////////
// static bool kEnableExecutablePipelines = true;

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
      // require any CPU_LEVEL work to decode. Otherwise, parsing etc should
      // have already been placed us on the desired Executor and no queues will
      // be pushed to.
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
                          outputs_[0]->InputReceived(this, std::move(batch));

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
      if (value.is_scalar()) continue;
      ARROW_ASSIGN_OR_RAISE(value,
                            Filter(value, mask, FilterOptions::Defaults()));
    }
    return ExecBatch::Make(std::move(values));
  }

  void InputReceived(ExecNodeKernel *input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);

    auto maybe_filtered = DoFilter(std::move(batch));
    if (ErrorIfNotOk(maybe_filtered.status())) return;

    maybe_filtered->guarantee = batch.guarantee;
    outputs_[0]->InputReceived(this, maybe_filtered.MoveValueUnsafe());
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
    if (ErrorIfNotOk(maybe_projected.status())) return;

    maybe_projected->guarantee = batch.guarantee;
    outputs_[0]->InputReceived(this, maybe_projected.MoveValueUnsafe());
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
      if (i > 0) ss << ", ";
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

}  // namespace compute
}  // namespace arrow

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

  static MyPushGenerator<util::optional<ExecBatch>>::Producer MakeProducer(
      AsyncGenerator<util::optional<ExecBatch>> *out_gen,
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
    if (!did_push) return;  // producer_ was Closed already

    if (input_counter_.Increment()) {
      Finish();
    }
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

}  // namespace

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
  DataHolderExecContext exec_context(default_memory_pool(),
                                     ::arrow::internal::GetCpuThreadPool());

  ASSERT_OK_AND_ASSIGN(auto plan, ExecPlanKernel::Make(&exec_context));

  int num_batches = 1000;
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
    //    ::arrow::PrettyPrint(*batch.ToRecordBatch(basic_data.schema).ValueOrDie(),
    //                         {}, &std::cerr);
    //    PrintTo(batch, &std::cerr);
  }
}

}  // namespace compute
}  // namespace arrow

/*
 * 1. Memory Resource API > SizeInBytes() [weston]
 * 2. DataHolder
 *  3. ExecBatch > move ownership (share_pointer ->)
 *
 *
 * 1. Fragments
 *            -> Person {ExecNode1, ExecNode2, ExecNode3 ... ExecNodeN} ->
 * DataHolder  ---
 *                                                                                        ---
 * JOIN -> [xecNode1, ExecNode2, ExecNode3 ... ExecNodeN] - SinkNode
 *            -> Country {ExecNode1, ExecNode2, ExecNode3 ... ExecNodeN} ->
 * DataHolder ---
 * 2. task processing >
 *     -  error handling! ->
 * */

namespace arrow {
namespace compute {

// Atomic value surrounded by padding bytes to avoid cache line invalidation
// whenever it is modified by a concurrent thread on a different CPU core.
//
template <typename T>
class AtomicWithPadding {
 private:
  static constexpr int kCacheLineSize = 64;
  uint8_t padding_before[kCacheLineSize];

 public:
  std::atomic<T> value;

 private:
  uint8_t padding_after[kCacheLineSize];
};

// Used for asynchronous execution of operations that can be broken into
// a fixed number of symmetric tasks that can be executed concurrently.
//
// Implements priorities between multiple such operations, called task groups.
//
// Allows to specify the maximum number of in-flight tasks at any moment.
//
// Also allows for executing next pending tasks immediately using a caller
// thread.
//
class TaskScheduler {
 public:
  using TaskImpl = std::function<Status(size_t, int64_t)>;
  using TaskGroupContinuationImpl = std::function<Status(size_t)>;
  using ScheduleImpl = std::function<Status(TaskGroupContinuationImpl)>;
  using AbortContinuationImpl = std::function<void()>;

  virtual ~TaskScheduler() = default;

  // Order in which task groups are registered represents priorities of their
  // tasks (the first group has the highest priority).
  //
  // Returns task group identifier that is used to request operations on the
  // task group.
  virtual int RegisterTaskGroup(TaskImpl task_impl,
                                TaskGroupContinuationImpl cont_impl) = 0;

  virtual void RegisterEnd() = 0;

  // total_num_tasks may be zero, in which case task group continuation will be
  // executed immediately
  virtual Status StartTaskGroup(size_t thread_id, int group_id,
                                int64_t total_num_tasks) = 0;

  // Execute given number of tasks immediately using caller thread
  virtual Status ExecuteMore(size_t thread_id, int num_tasks_to_execute,
                             bool execute_all) = 0;

  // Begin scheduling tasks using provided callback and
  // the limit on the number of in-flight tasks at any moment.
  //
  // Scheduling will continue as long as there are waiting tasks.
  //
  // It will automatically resume whenever new task group gets started.
  virtual Status StartScheduling(size_t thread_id, ScheduleImpl schedule_impl,
                                 int num_concurrent_tasks,
                                 bool use_sync_execution) = 0;

  // Abort scheduling and execution.
  // Used in case of being notified about unrecoverable error for the entire
  // query.
  virtual void Abort(AbortContinuationImpl impl) = 0;

  static std::unique_ptr<TaskScheduler> Make();
};

class TaskSchedulerImpl : public TaskScheduler {
 public:
  TaskSchedulerImpl();
  int RegisterTaskGroup(TaskImpl task_impl,
                        TaskGroupContinuationImpl cont_impl) override;
  void RegisterEnd() override;
  Status StartTaskGroup(size_t thread_id, int group_id,
                        int64_t total_num_tasks) override;
  Status ExecuteMore(size_t thread_id, int num_tasks_to_execute,
                     bool execute_all) override;
  Status StartScheduling(size_t thread_id, ScheduleImpl schedule_impl,
                         int num_concurrent_tasks,
                         bool use_sync_execution) override;
  void Abort(AbortContinuationImpl impl) override;

 private:
  // Task group state transitions progress one way.
  // Seeing an old version of the state by a thread is a valid situation.
  //
  enum class TaskGroupState : int {
    NOT_READY,
    READY,
    ALL_TASKS_STARTED,
    ALL_TASKS_FINISHED
  };

  struct TaskGroup {
    TaskGroup(TaskImpl task_impl, TaskGroupContinuationImpl cont_impl)
        : task_impl_(std::move(task_impl)),
          cont_impl_(std::move(cont_impl)),
          state_(TaskGroupState::NOT_READY),
          num_tasks_present_(0) {
      num_tasks_started_.value.store(0);
      num_tasks_finished_.value.store(0);
    }
    TaskGroup(const TaskGroup &src)
        : task_impl_(src.task_impl_),
          cont_impl_(src.cont_impl_),
          state_(TaskGroupState::NOT_READY),
          num_tasks_present_(0) {
      ARROW_DCHECK(src.state_ == TaskGroupState::NOT_READY);
      num_tasks_started_.value.store(0);
      num_tasks_finished_.value.store(0);
    }
    TaskImpl task_impl_;
    TaskGroupContinuationImpl cont_impl_;

    TaskGroupState state_;
    int64_t num_tasks_present_;

    AtomicWithPadding<int64_t> num_tasks_started_;
    AtomicWithPadding<int64_t> num_tasks_finished_;
  };

  std::vector<std::pair<int, int64_t>> PickTasks(int num_tasks,
                                                 int start_task_group = 0);
  Status ExecuteTask(size_t thread_id, int group_id, int64_t task_id,
                     bool *task_group_finished);
  bool PostExecuteTask(size_t thread_id, int group_id);
  Status OnTaskGroupFinished(size_t thread_id, int group_id,
                             bool *all_task_groups_finished);
  Status ScheduleMore(size_t thread_id, int num_tasks_finished = 0);

  bool use_sync_execution_;
  int num_concurrent_tasks_;
  ScheduleImpl schedule_impl_;
  AbortContinuationImpl abort_cont_impl_;

  std::vector<TaskGroup> task_groups_;
  bool aborted_;
  bool register_finished_;
  std::mutex
      mutex_;  // Mutex protecting task_groups_ (state_ and num_tasks_present_
  // fields), aborted_ flag and register_finished_ flag

  AtomicWithPadding<int> num_tasks_to_schedule_;
};

TaskSchedulerImpl::TaskSchedulerImpl()
    : use_sync_execution_(false),
      num_concurrent_tasks_(0),
      aborted_(false),
      register_finished_(false) {
  num_tasks_to_schedule_.value.store(0);
}

int TaskSchedulerImpl::RegisterTaskGroup(TaskImpl task_impl,
                                         TaskGroupContinuationImpl cont_impl) {
  int result = static_cast<int>(task_groups_.size());
  task_groups_.emplace_back(std::move(task_impl), std::move(cont_impl));
  return result;
}

void TaskSchedulerImpl::RegisterEnd() {
  std::lock_guard<std::mutex> lock(mutex_);

  register_finished_ = true;
}

Status TaskSchedulerImpl::StartTaskGroup(size_t thread_id, int group_id,
                                         int64_t total_num_tasks) {
  ARROW_DCHECK(group_id >= 0 &&
               group_id < static_cast<int>(task_groups_.size()));
  TaskGroup &task_group = task_groups_[group_id];

  bool aborted = false;
  bool all_tasks_finished = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    aborted = aborted_;

    if (task_group.state_ == TaskGroupState::NOT_READY) {
      task_group.num_tasks_present_ = total_num_tasks;
      if (total_num_tasks == 0) {
        task_group.state_ = TaskGroupState::ALL_TASKS_FINISHED;
        all_tasks_finished = true;
      }
      task_group.state_ = TaskGroupState::READY;
    }
  }

  if (!aborted && all_tasks_finished) {
    bool all_task_groups_finished = false;
    RETURN_NOT_OK(
        OnTaskGroupFinished(thread_id, group_id, &all_task_groups_finished));
    if (all_task_groups_finished) {
      return Status::OK();
    }
  }

  if (!aborted) {
    return ScheduleMore(thread_id);
  } else {
    return Status::Cancelled("Scheduler cancelled");
  }
}

std::vector<std::pair<int, int64_t>> TaskSchedulerImpl::PickTasks(
    int num_tasks, int start_task_group) {
  std::vector<std::pair<int, int64_t>> result;
  for (size_t i = 0; i < task_groups_.size(); ++i) {
    int task_group_id =
        static_cast<int>((start_task_group + i) % (task_groups_.size()));
    TaskGroup &task_group = task_groups_[task_group_id];

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (task_group.state_ != TaskGroupState::READY) {
        continue;
      }
    }

    int num_tasks_remaining = num_tasks - static_cast<int>(result.size());
    int64_t start_task =
        task_group.num_tasks_started_.value.fetch_add(num_tasks_remaining);
    if (start_task >= task_group.num_tasks_present_) {
      continue;
    }

    int num_tasks_current_group = num_tasks_remaining;
    if (start_task + num_tasks_current_group >= task_group.num_tasks_present_) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (task_group.state_ == TaskGroupState::READY) {
          task_group.state_ = TaskGroupState::ALL_TASKS_STARTED;
        }
      }
      num_tasks_current_group =
          static_cast<int>(task_group.num_tasks_present_ - start_task);
    }

    for (int64_t task_id = start_task;
         task_id < start_task + num_tasks_current_group; ++task_id) {
      result.push_back(std::make_pair(task_group_id, task_id));
    }

    if (static_cast<int>(result.size()) == num_tasks) {
      break;
    }
  }

  return result;
}

Status TaskSchedulerImpl::ExecuteTask(size_t thread_id, int group_id,
                                      int64_t task_id,
                                      bool *task_group_finished) {
  if (!aborted_) {
    RETURN_NOT_OK(task_groups_[group_id].task_impl_(thread_id, task_id));
  }
  *task_group_finished = PostExecuteTask(thread_id, group_id);
  return Status::OK();
}

bool TaskSchedulerImpl::PostExecuteTask(size_t thread_id, int group_id) {
  int64_t total = task_groups_[group_id].num_tasks_present_;
  int64_t prev_finished =
      task_groups_[group_id].num_tasks_finished_.value.fetch_add(1);
  bool all_tasks_finished = (prev_finished + 1 == total);
  return all_tasks_finished;
}

Status TaskSchedulerImpl::OnTaskGroupFinished(size_t thread_id, int group_id,
                                              bool *all_task_groups_finished) {
  bool aborted = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    aborted = aborted_;
    TaskGroup &task_group = task_groups_[group_id];
    task_group.state_ = TaskGroupState::ALL_TASKS_FINISHED;
    *all_task_groups_finished = true;
    for (size_t i = 0; i < task_groups_.size(); ++i) {
      if (task_groups_[i].state_ != TaskGroupState::ALL_TASKS_FINISHED) {
        *all_task_groups_finished = false;
        break;
      }
    }
  }

  if (aborted && *all_task_groups_finished) {
    abort_cont_impl_();
    return Status::Cancelled("Scheduler cancelled");
  }
  if (!aborted) {
    RETURN_NOT_OK(task_groups_[group_id].cont_impl_(thread_id));
  }
  return Status::OK();
}

Status TaskSchedulerImpl::ExecuteMore(size_t thread_id,
                                      int num_tasks_to_execute,
                                      bool execute_all) {
  num_tasks_to_execute = std::max(1, num_tasks_to_execute);

  int last_id = 0;
  for (;;) {
    if (aborted_) {
      return Status::Cancelled("Scheduler cancelled");
    }

    // Pick next bundle of tasks
    const auto &tasks = PickTasks(num_tasks_to_execute, last_id);
    if (tasks.empty()) {
      break;
    }
    last_id = tasks.back().first;

    // Execute picked tasks immediately
    for (size_t i = 0; i < tasks.size(); ++i) {
      int group_id = tasks[i].first;
      int64_t task_id = tasks[i].second;
      bool task_group_finished = false;
      Status status =
          ExecuteTask(thread_id, group_id, task_id, &task_group_finished);
      if (!status.ok()) {
        // Mark the remaining picked tasks as finished
        for (size_t j = i + 1; j < tasks.size(); ++j) {
          if (PostExecuteTask(thread_id, tasks[j].first)) {
            bool all_task_groups_finished = false;
            RETURN_NOT_OK(OnTaskGroupFinished(thread_id, group_id,
                                              &all_task_groups_finished));
            if (all_task_groups_finished) {
              return Status::OK();
            }
          }
        }
        return status;
      } else {
        if (task_group_finished) {
          bool all_task_groups_finished = false;
          RETURN_NOT_OK(OnTaskGroupFinished(thread_id, group_id,
                                            &all_task_groups_finished));
          if (all_task_groups_finished) {
            return Status::OK();
          }
        }
      }
    }

    if (!execute_all) {
      num_tasks_to_execute -= static_cast<int>(tasks.size());
      if (num_tasks_to_execute == 0) {
        break;
      }
    }
  }

  return Status::OK();
}

Status TaskSchedulerImpl::StartScheduling(size_t thread_id,
                                          ScheduleImpl schedule_impl,
                                          int num_concurrent_tasks,
                                          bool use_sync_execution) {
  schedule_impl_ = std::move(schedule_impl);
  use_sync_execution_ = use_sync_execution;
  num_concurrent_tasks_ = num_concurrent_tasks;
  num_tasks_to_schedule_.value += num_concurrent_tasks;
  return ScheduleMore(thread_id);
}

Status TaskSchedulerImpl::ScheduleMore(size_t thread_id,
                                       int num_tasks_finished) {
  if (aborted_) {
    return Status::Cancelled("Scheduler cancelled");
  }

  ARROW_DCHECK(register_finished_);

  if (use_sync_execution_) {
    return ExecuteMore(thread_id, 1, true);
  }

  int num_new_tasks = num_tasks_finished;
  for (;;) {
    int expected = num_tasks_to_schedule_.value.load();
    if (num_tasks_to_schedule_.value.compare_exchange_strong(expected, 0)) {
      num_new_tasks += expected;
      break;
    }
  }
  if (num_new_tasks == 0) {
    return Status::OK();
  }

  const auto &tasks = PickTasks(num_new_tasks);
  if (static_cast<int>(tasks.size()) < num_new_tasks) {
    num_tasks_to_schedule_.value +=
        num_new_tasks - static_cast<int>(tasks.size());
  }

  for (size_t i = 0; i < tasks.size(); ++i) {
    int group_id = tasks[i].first;
    int64_t task_id = tasks[i].second;
    RETURN_NOT_OK(
        schedule_impl_([this, group_id, task_id](size_t thread_id) -> Status {
          RETURN_NOT_OK(ScheduleMore(thread_id, 1));

          bool task_group_finished = false;
          RETURN_NOT_OK(
              ExecuteTask(thread_id, group_id, task_id, &task_group_finished));

          if (task_group_finished) {
            bool all_task_groups_finished = false;
            return OnTaskGroupFinished(thread_id, group_id,
                                       &all_task_groups_finished);
          }

          return Status::OK();
        }));
  }
  return Status::OK();
}

void TaskSchedulerImpl::Abort(AbortContinuationImpl impl) {
  bool all_finished = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    abort_cont_impl_ = std::move(impl);
    if (register_finished_) {
      for (size_t i = 0; i < task_groups_.size(); ++i) {
        TaskGroup &task_group = task_groups_[i];
        if (task_group.state_ == TaskGroupState::NOT_READY) {
          task_group.state_ = TaskGroupState::ALL_TASKS_FINISHED;
        } else if (task_group.state_ == TaskGroupState::READY) {
          int64_t expected = task_group.num_tasks_started_.value.load();
          for (;;) {
            if (task_group.num_tasks_started_.value.compare_exchange_strong(
                    expected, task_group.num_tasks_present_)) {
              break;
            }
          }
          int64_t before_add = task_group.num_tasks_finished_.value.fetch_add(
              task_group.num_tasks_present_ - expected);
          if (before_add >= expected) {
            task_group.state_ = TaskGroupState::ALL_TASKS_FINISHED;
          } else {
            all_finished = false;
            task_group.state_ = TaskGroupState::ALL_TASKS_STARTED;
          }
        }
      }
    }
  }
  if (all_finished) {
    abort_cont_impl_();
  }
}

std::unique_ptr<TaskScheduler> TaskScheduler::Make() {
  std::unique_ptr<TaskSchedulerImpl> impl{new TaskSchedulerImpl()};
  return std::move(impl);
}

struct SomeExecNode {
  using OutputBatchCallback = std::function<void(ExecBatch)>;
  using FinishedCallback = std::function<void(int64_t)>;
  int first_task_group;
  int second_task_group;
  int third_task_group;
  FinishedCallback finished_callback_;
  Future<> finished_ = Future<>::Make();

  SomeExecNode(::arrow::internal::Executor *executor) : executor_(executor) {
    scheduler_ = TaskScheduler::Make();

    bool use_sync_execution = !(executor_);
    size_t num_threads = use_sync_execution ? 1 : thread_indexer_.Capacity();

    this->finished_callback_ = [this](int64_t total) {
      std::cout << ">>> finished_callback_\n";
      finished_.MarkFinished();
    };

    this->first_task_group = scheduler_->RegisterTaskGroup(
        [this](size_t thread_index, int64_t task_id) -> Status {
          std::cout << "First Task Group\n";
          return Status::OK();
        },
        [this](size_t thread_index) -> Status {
          std::cout << "First Task Group Continuation\n";

          scheduler_->StartTaskGroup(thread_index, second_task_group, 1);
          return Status::OK();
        });

    this->second_task_group = scheduler_->RegisterTaskGroup(
        [this](size_t thread_index, int64_t task_id) -> Status {
          std::cout << "Second Task Group\n";
          return Status::OK();
        },
        [this](size_t thread_index) -> Status {
          std::cout << "Second Task Group Continuation\n";

          scheduler_->StartTaskGroup(thread_index, third_task_group, 1);

          return Status::OK();
        });

    this->third_task_group = scheduler_->RegisterTaskGroup(
        [this](size_t thread_index, int64_t task_id) -> Status {
          std::cout << "Third Task Group\n";
          return Status::OK();
        },
        [this](size_t thread_index) -> Status {
          std::cout << "Third Task Continuation\n";
          finished_callback_(3);

          return Status::OK();
        });

    scheduler_->RegisterEnd();

    auto schedule_task_callback =
        [this](std::function<Status(size_t)> func) -> Status {
      return this->ScheduleTaskCallback(std::move(func));
    };

    scheduler_->StartTaskGroup(0, first_task_group, 1);
    std::cerr << "> StartTaskGroup()\n";

    scheduler_->StartScheduling(0, std::move(schedule_task_callback),
                                2 * num_threads, use_sync_execution);

    std::cerr << "> StartScheduling()\n";
  }

  Status ScheduleTaskCallback(std::function<Status(size_t)> func) {
    auto status = executor_->Spawn([this, func] {
      size_t thread_index = thread_indexer_();
      Status status = func(thread_index);
      if (!status.ok()) {
        // StopProducing();
        // ErrorIfNotOk(status);
        return;
      }
    });
    return status;
  }

  ThreadIndexer thread_indexer_;
  ::arrow::internal::Executor *executor_;
  std::unique_ptr<TaskScheduler> scheduler_;
};

}  // namespace compute
}  // namespace arrow

int main(int argc, char *argv[]) {
  //  arrow::internal::RunInSerialExecutor();

  //  arrow::internal::ThreadPoolSubmit();

  // arrow::compute::TestStartProducing();

  arrow::compute::DataHolderExecContext exec_context(
      arrow::default_memory_pool(), ::arrow::internal::GetCpuThreadPool());
  auto executor = exec_context.executor();
  arrow::compute::SomeExecNode node(executor);
  std::cerr << "> node.finished_.Wait()\n";
  node.finished_.Wait();
  std::cerr << " >>>> node.finished_.Wait()\n";
  return 0;
}
