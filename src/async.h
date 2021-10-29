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

#pragma once

#include <atomic>
#include <cassert>
#include <cstring>
#include <deque>
#include <limits>
#include <queue>

#include "arrow/util/async_util.h"
#include "arrow/util/functional.h"
#include "arrow/util/future.h"
#include "arrow/util/io_util.h"
#include "arrow/util/iterator.h"
#include "arrow/util/mutex.h"
#include "arrow/util/optional.h"
#include "arrow/util/queue.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/async_generator.h"

namespace arrow {

/// \brief A generator where the producer pushes items on a queue.
///
/// No back-pressure is applied, so this generator is mostly useful when
/// producing the values is neither CPU_LEVEL- nor memory-expensive (e.g. fetching
/// filesystem metadata).
///
/// This generator is not async-reentrant.
template <typename T>
class MyPushGenerator {
  struct State {
    explicit State(util::BackpressureOptions backpressure)
        : backpressure(std::move(backpressure)) {}

    void OpenBackpressureIfFreeUnlocked(util::Mutex::Guard&& guard) {
      if (backpressure.toggle && result_q.size() < backpressure.resume_if_below) {
        // Open might trigger callbacks so release the lock first
        guard.Unlock();
        printf("backpressure.toggle->Open()\n");
        backpressure.toggle->Open();
      }
    }

    void CloseBackpressureIfFullUnlocked() {
      if (backpressure.toggle && result_q.size() > backpressure.pause_if_above) {
        backpressure.toggle->Close();
        printf("backpressure.toggle->Close()\n");
      }
    }

    util::BackpressureOptions backpressure;
    util::Mutex mutex;
    std::deque<Result<T>> result_q;
    util::optional<Future<T>> consumer_fut;
    bool finished = false;
  };

public:
  /// Producer API for MyPushGenerator
  class Producer {
  public:
    explicit Producer(const std::shared_ptr<State>& state) : weak_state_(state) {}

    /// \brief Push a value on the queue
    ///
    /// True is returned if the value was pushed, false if the generator is
    /// already closed or destroyed.  If the latter, it is recommended to stop
    /// producing any further values.
    bool Push(Result<T> result) {
      auto state = weak_state_.lock();
      if (!state) {
        // Generator was destroyed
        return false;
      }
      auto lock = state->mutex.Lock();
      if (state->finished) {
        // Closed early
        return false;
      }
      if (state->consumer_fut.has_value()) {
        auto fut = std::move(state->consumer_fut.value());
        state->consumer_fut.reset();
        lock.Unlock();  // unlock before potentially invoking a callback
        fut.MarkFinished(std::move(result));
      } else {
        state->result_q.push_back(std::move(result));
        state->CloseBackpressureIfFullUnlocked();
      }
      return true;
    }

    /// \brief Tell the consumer we have finished producing
    ///
    /// It is allowed to call this and later call Push() again ("early close").
    /// In this case, calls to Push() after the queue is closed are silently
    /// ignored.  This can help implementing non-trivial cancellation cases.
    ///
    /// True is returned on success, false if the generator is already closed
    /// or destroyed.
    bool Close() {
      auto state = weak_state_.lock();
      if (!state) {
        // Generator was destroyed
        return false;
      }
      auto lock = state->mutex.Lock();
      if (state->finished) {
        // Already closed
        return false;
      }
      state->finished = true;
      if (state->consumer_fut.has_value()) {
        auto fut = std::move(state->consumer_fut.value());
        state->consumer_fut.reset();
        lock.Unlock();  // unlock before potentially invoking a callback
        fut.MarkFinished(IterationTraits<T>::End());
      }
      return true;
    }

    /// Return whether the generator was closed or destroyed.
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

  explicit MyPushGenerator(util::BackpressureOptions backpressure = {})
      : state_(std::make_shared<State>(std::move(backpressure))) {}

  /// Read an item from the queue
  Future<T> operator()() const {
    auto lock = state_->mutex.Lock();
    assert(!state_->consumer_fut.has_value());  // Non-reentrant
    if (!state_->result_q.empty()) {
      auto fut = Future<T>::MakeFinished(std::move(state_->result_q.front()));
      state_->result_q.pop_front();
      state_->OpenBackpressureIfFreeUnlocked(std::move(lock));
      return fut;
    }
    if (state_->finished) {
      return AsyncGeneratorEnd<T>();
    }
    auto fut = Future<T>::Make();
    state_->consumer_fut = fut;
    return fut;
  }

  /// \brief Return producer-side interface
  ///
  /// The returned object must be used by the producer to Push values on the queue.
  /// Only a single Producer object should be instantiated.
  Producer producer() { return Producer{state_}; }

private:
  const std::shared_ptr<State> state_;
};
 
}  // namespace arrow
