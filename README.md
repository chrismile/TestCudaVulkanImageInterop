# TestCudaVulkanImageInterop

Test app for CUDA-Vulkan image interop.

CUDA writes linearly increasing values to an RGBA 32-bit float image of size 1024 x 1024.
Vulkan then copies this to a device local buffer, and from there to a host visible buffer.

The test case works fine on Linux, but fails on Windows (as of 2026-02-01 using recent drivers) due to race conditions.
Synchronization can be either done by syncing on the CPU or using semaphores. Both cases fail.

Tested system configurations:
- Ubuntu 24.04, CUDA 12.6, driver 580.126.09, GCC 13.3

## How to build

```sh
git submodule update --init --recursive
mkdir build
cd build
cmake ..
cmake --build .

# Linux
./TestCudaVulkanImageInterop.exe

#Windows
./TestCudaVulkanImageInterop.exe
```
