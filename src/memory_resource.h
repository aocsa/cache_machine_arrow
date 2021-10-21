//
// Created by alexander on 2021-10-20.
//

#ifndef AUCAHUASI_ARROW_DATASET_SRC_MEMORY_RESOURCE_H_
#define AUCAHUASI_ARROW_DATASET_SRC_MEMORY_RESOURCE_H_


#include <cassert>
#include <atomic>

#include <sys/sysinfo.h>
#include <sys/statvfs.h>


class MemoryResource {
public:
  virtual size_t get_from_driver_available_memory() = 0 ; // driver.get_available_memory()
  virtual size_t get_memory_limit() = 0 ; // memory_limite = total_memory * threshold

  virtual size_t get_memory_used() = 0 ; // atomic
  virtual size_t get_total_memory() = 0 ; // total_memory
};



/**
        @brief This class represents a custom host memory resource used in the cache system.
*/
class internal_blazing_host_memory_resource{
public:
  // TODO: percy,cordova. Improve the design of get memory in real time
  internal_blazing_host_memory_resource(float custom_threshold)
  {
    struct sysinfo si;
    if (sysinfo(&si) < 0) {
      std::cerr << "@@ error sysinfo host "<< std::endl;
    }
    total_memory_size = (size_t)si.freeram;
    used_memory_size = 0;
    memory_limit = custom_threshold * total_memory_size;
  }

  virtual ~internal_blazing_host_memory_resource() = default;

  // TODO
  void allocate(std::size_t bytes)  {
    used_memory_size +=  bytes;
  }

  void deallocate(std::size_t bytes)  {
    used_memory_size -= bytes;
  }

  size_t get_from_driver_available_memory()  {
    struct sysinfo si;
    sysinfo (&si);
    // NOTE: sync point
    total_memory_size = (size_t)si.totalram;
    used_memory_size = total_memory_size - (size_t)si.freeram;;
    return used_memory_size;
  }

  size_t get_memory_used() {
    return used_memory_size;
  }

  size_t get_total_memory() {
    return total_memory_size;
  }

  size_t get_memory_limit() {
    return memory_limit;
  }

private:
  size_t memory_limit;
  size_t total_memory_size;
  std::atomic<std::size_t> used_memory_size;
};


/** -------------------------------------------------------------------------*
 * @brief CPUMemoryResource class maintains the host memory manager context.
 *
 * CPUMemoryResource is a singleton class, and should be accessed via getInstance().
 * A number of static convenience methods are provided that wrap getInstance()..
 * ------------------------------------------------------------------------**/
class CPUMemoryResource : public MemoryResource {
public:
  /** -----------------------------------------------------------------------*
     * @brief Get the CPUMemoryResource instance singleton object
     *
     * @return CPUMemoryResource& the CPUMemoryResource singleton
     * ----------------------------------------------------------------------**/
  static CPUMemoryResource& getInstance(){
    // Myers' singleton. Thread safe and unique. Note: C++11 required.
    static CPUMemoryResource instance;
    return instance;
  }

  size_t get_memory_used() override {
    // std::cout << "CPUMemoryResource: " << initialized_resource->get_memory_used() << std::endl;
    return initialized_resource->get_memory_used();
  }

  size_t get_total_memory() override {
    return initialized_resource->get_total_memory() ;
  }

  size_t get_from_driver_available_memory()  {
    return initialized_resource->get_from_driver_available_memory();
  }
  size_t get_memory_limit() {
    return initialized_resource->get_memory_limit() ;
  }

  void allocate(std::size_t bytes)  {
    initialized_resource->allocate(bytes);
  }

  void deallocate(std::size_t bytes)  {
    initialized_resource->deallocate(bytes);
  }

  /** -----------------------------------------------------------------------*
   * @brief Initialize
   *
   * Accepts an optional rmmOptions_t struct that describes the settings used
   * to initialize the memory manager. If no `options` is passed, default
   * options are used.
   *
   * @param[in] options Optional options to set
   * ----------------------------------------------------------------------**/
  void initialize(float host_mem_resouce_consumption_thresh) {

    std::lock_guard<std::mutex> guard(manager_mutex);

    // repeat initialization is a no-op
    if (isInitialized()) return;

    initialized_resource.reset(new internal_blazing_host_memory_resource(host_mem_resouce_consumption_thresh));

    is_initialized = true;
  }

  /** -----------------------------------------------------------------------*
     * @brief Shut down the blazing_device_memory_resource (clears the context)
     * ----------------------------------------------------------------------**/
  void finalize(){
    std::lock_guard<std::mutex> guard(manager_mutex);

    // finalization before initialization is a no-op
    if (isInitialized()) {
      initialized_resource.reset();
      is_initialized = false;
    }
  }

  /** -----------------------------------------------------------------------*
     * @brief Check whether the blazing_device_memory_resource has been initialized.
     *
     * @return true if blazing_device_memory_resource has been initialized.
     * @return false if blazing_device_memory_resource has not been initialized.
     * ----------------------------------------------------------------------**/
  bool isInitialized() {
    return getInstance().is_initialized;
  }

private:
  CPUMemoryResource() = default;
  ~CPUMemoryResource() = default;
  CPUMemoryResource(const CPUMemoryResource&) = delete;
  CPUMemoryResource& operator=(const CPUMemoryResource&) = delete;
  std::mutex manager_mutex;

  bool is_initialized{false};

  std::unique_ptr<internal_blazing_host_memory_resource> initialized_resource{};
};

/**
        @brief This class represents a custom disk memory resource used in the cache system.
*/
class DiskMemoryResource : public  MemoryResource {
public:
  static DiskMemoryResource& getInstance(){
    // Myers' singleton. Thread safe and unique. Note: C++11 required.
    static DiskMemoryResource instance;
    return instance;
  }

  // TODO: percy, cordova.Improve the design of get memory in real time
  DiskMemoryResource(float custom_threshold = 0.75) {
    struct statvfs stat_disk;
    int ret = statvfs("/", &stat_disk);

    total_memory_size = (size_t)(stat_disk.f_blocks * stat_disk.f_frsize);
    size_t available_disk_size = (size_t)(stat_disk.f_bfree * stat_disk.f_frsize);
    used_memory_size = total_memory_size - available_disk_size;

    memory_limit = custom_threshold *  total_memory_size;
  }

  virtual ~DiskMemoryResource() = default;

  virtual size_t get_from_driver_available_memory()  {
    struct sysinfo si;
    sysinfo (&si);
    // NOTE: sync point
    total_memory_size = (size_t)si.totalram;
    used_memory_size =  total_memory_size - (size_t)si.freeram;
    return used_memory_size;
  }
  size_t get_memory_limit()  {
    return memory_limit;
  }

  size_t get_memory_used() {
    return used_memory_size;
  }

  size_t get_total_memory() {
    return total_memory_size;
  }

private:
  size_t total_memory_size;
  size_t memory_limit;
  std::atomic<size_t> used_memory_size;
};

#endif // AUCAHUASI_ARROW_DATASET_SRC_MEMORY_RESOURCE_H_
