#pragma once


#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>

//#include <parquet/file_reader.h>

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
//
//#include "parquet/arrow/reader.h"
//#include "parquet/arrow/writer.h"
//#include "parquet/platform.h"
//#include "parquet/properties.h"

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

#include "memory_resource.h"


#include <arrow/filesystem/filesystem.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include "arrow/io/file.h"

namespace arrow {
namespace compute {


struct DataHolderExecContext : ExecContext {
  explicit DataHolderExecContext(MemoryPool* pool = default_memory_pool(),
                                 ::arrow::internal::Executor* executor = NULLPTR,
                                 FunctionRegistry* func_registry = NULLPTR)
      : ExecContext (pool, executor, func_registry)
  {
    memory_resources_stats_.emplace_back(
        std::make_shared<::arrow::internal::CPUMemoryResourceStats>(pool));
    memory_resources_stats_.emplace_back(
        std::make_shared<::arrow::internal::DiskMemoryResourceStats>());
  }

  size_t memory_resources_size() {
    return memory_resources_stats_.size();
  }

  std::shared_ptr<::arrow::internal::MemoryResourceStats> memory_resource(size_t index) {
    return memory_resources_stats_[index];
  }
private:
  std::vector<std::shared_ptr<::arrow::internal::MemoryResourceStats>> memory_resources_stats_;
};

using namespace std::chrono_literals;

enum class DataHolderType { CPU_LEVEL, DISK_LEVEL };

struct DataHolder {
  explicit DataHolder(DataHolderType type)
      : type_(type) {}


  DataHolderType type() const { return type_; };

  virtual size_t SizeInBytes() const = 0;

  virtual Result<ExecBatch> Get() = 0;

private:
  DataHolderType type_;
};

class CPUDataHolder : public DataHolder {
public:
  explicit CPUDataHolder(const std::shared_ptr<RecordBatch>& record_batch)
      : DataHolder(DataHolderType::CPU_LEVEL), batch_(std::move(record_batch)) {}

  virtual size_t SizeInBytes() const override {
    auto record_batch = batch_;
    size_t size_in_bytes = 0;
    for (auto&& column : record_batch->columns()) {
      const auto& data = column->data();
      for (auto&& buffer : data->buffers) {
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

//using WriterProperties = parquet::WriterProperties;
//using ArrowWriterProperties = parquet::ArrowWriterProperties;
//
//using FileReader = parquet::arrow::FileReader;
//using ParquetFileReader = parquet::ParquetFileReader;

Status StoreRecordBatch(const std::shared_ptr<RecordBatch>& record_batch,
                        const std::shared_ptr<fs::FileSystem>& filesystem,
                        const std::string& file_path) {
  auto output = filesystem->OpenOutputStream(file_path).ValueOrDie();
  auto writer =
      arrow::ipc::MakeFileWriter(output.get(), record_batch->schema()).ValueOrDie();
  ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*record_batch));
  return writer->Close();
}
Result<std::shared_ptr<RecordBatch>> RecoverRecordBatch(const std::shared_ptr<fs::FileSystem>& filesystem, const std::string& file_path) {
  ARROW_ASSIGN_OR_RAISE(auto input, filesystem->OpenInputFile(file_path));
  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::feather::Reader::Open(input));
  std::shared_ptr<Table> table;
  ARROW_RETURN_NOT_OK(reader->Read(&table));
  TableBatchReader batch_iter(*table);
  ARROW_ASSIGN_OR_RAISE(auto batch, batch_iter.Next());
  return batch;
}
namespace ds = arrow::dataset;

class DiskDataHolder : public DataHolder {
public:
  explicit DiskDataHolder(const std::shared_ptr<RecordBatch>& record_batch)
      : DataHolder( DataHolderType::CPU_LEVEL) {
    std::string root_path;
    std::string file_name = "data-holder-temp-" + RandomString(64) + ".feather";

    filesystem_ =
        arrow::fs::FileSystemFromUri(cache_storage_root_path, &root_path).ValueOrDie();

    file_path_ = root_path + file_name;
    status_ = StoreRecordBatch(record_batch, filesystem_, file_path_);
   }
  virtual size_t SizeInBytes() const override {
    struct stat st;
    if (stat(this->file_path_.c_str(), &st) == 0)
      return (st.st_size);
    else
      throw;
  }

  Result<ExecBatch> Get() override {
    ARROW_ASSIGN_OR_RAISE(auto record_batch, RecoverRecordBatch(filesystem_, file_path_));
    return ExecBatch(*record_batch);
  }

private:
  std::string file_path_;
  std::shared_ptr<arrow::fs::FileSystem> filesystem_;
  Status status_;
  const std::string cache_storage_root_path = "file:///tmp/";

};

size_t SizeInBytes(ExecBatch batch, std::shared_ptr<Schema> schema) {
  auto record_batch = batch.ToRecordBatch(schema).ValueOrDie();
  size_t size_in_bytes = 0;
  for (auto&& column : record_batch->columns()) {
    const auto& data = column->data();
    for (auto&& buffer : data->buffers) {
      if (buffer) {
        size_in_bytes += buffer->size();
      }
    }
  }
  return size_in_bytes;
}

class DataHolderManager {
public:
  DataHolderManager(DataHolderExecContext* context)
      : context_(context), gen_(), producer_(gen_.producer()) {}

  Status Push(ExecBatch batch, std::shared_ptr<Schema> schema) {
    int index = 0;
    while (index < context_->memory_resources_size()) {
      auto&& resource = context_->memory_resource(index);
      auto memory_to_use = resource->memory_used() + SizeInBytes(batch, schema);
      if (memory_to_use < resource->memory_limit()) {
        if (index == static_cast<int>(DataHolderType::CPU_LEVEL)) {
          auto data_holder = std::make_unique<CPUDataHolder>(
              batch.ToRecordBatch(schema).ValueOrDie());
          this->producer_.Push(std::move(data_holder));
        } else if (index == static_cast<int>(DataHolderType::DISK_LEVEL)) {
          auto disk_data_holder = std::make_unique<DiskDataHolder>(batch.ToRecordBatch(schema).ValueOrDie());
          this->producer_.Push(std::move(disk_data_holder));
        } else {
          return Status::NotImplemented("There is not a default data holder registered");
        }
        break;
      }
      index++;
    }
    return Status::OK();
  }

  AsyncGenerator<std::unique_ptr<DataHolder>> generator() { return gen_; }

public:
  PushGenerator<std::unique_ptr<DataHolder>> gen_;
  PushGenerator<std::unique_ptr<DataHolder>>::Producer producer_;
  DataHolderExecContext* context_;
};

}  // namespace compute
}  // namespace arrow