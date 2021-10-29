// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/memory_pool.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#ifdef __linux__
#include <sys/statvfs.h>
#include <sys/sysinfo.h>
#endif

#include "arrow/util/windows_compatibility.h"

namespace arrow {

namespace internal {

struct MemoryResourceStats {
  virtual int64_t memory_limit() = 0;

  virtual int64_t memory_used() = 0;
};

struct CPUMemoryResourceStats : public MemoryResourceStats {
  CPUMemoryResourceStats(arrow::MemoryPool* pool, float memory_limit_threshold = 0.01)
      : pool_(pool) {
    total_memory_size_ = GetTotalMemorySize();
    memory_limit_ = memory_limit_threshold * total_memory_size_;
  }

  int64_t memory_used() override { return pool_->bytes_allocated(); }

  int64_t memory_limit() override { return memory_limit_; }

  size_t GetTotalMemorySize() {
#ifdef __APPLE__
    int mib[2];
    size_t physical_memory;
    size_t length;
    // Get the Physical memory size
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(size_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    return physical_memory;
#elif defined(_MSC_VER)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
#else  // Linux
    struct sysinfo si;
    return (size_t)si.freeram;
#endif
  }

private:
  arrow::MemoryPool* pool_;
  int64_t memory_limit_;
  int64_t total_memory_size_;
};

class DiskMemoryResourceStats : public MemoryResourceStats {
public:
  DiskMemoryResourceStats() {
    memory_used_ = 0;
    memory_limit_ = std::numeric_limits<int64_t>::max();
  }

  int64_t memory_limit() override { return memory_limit_; }

  int64_t memory_used() override { return memory_used_; }

private:
  int64_t memory_used_;
  int64_t memory_limit_;
};

}  // namespace internal
}  // namespace arrow