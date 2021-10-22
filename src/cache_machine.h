#pragma once

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
namespace compute {

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
    //    printf("DiskDataHolder:Read\n");

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

class CacheMachineOld {
public:
  CacheMachineOld(ExecContext *context) : context_(context) {
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


}
}
