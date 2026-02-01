/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2026, Christoph Neuhauser
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>

#include <Math/Math.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Shader/ShaderManager.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>

#include "CudaDeviceCode.hpp"

void initDevice(sgl::vk::Instance*& instance, sgl::vk::Device*& device) {
    instance = new sgl::vk::Instance;
    instance->createInstance({}, false);

    // Create a Vulkan device (for any NVIDIA GPU available in the system).
    device = new sgl::vk::Device;
    auto physicalDeviceCheckCallback = [&](
            VkPhysicalDevice physicalDevice,
            VkPhysicalDeviceProperties physicalDeviceProperties,
            std::vector<const char*>& requiredDeviceExtensions,
            std::vector<const char*>& optionalDeviceExtensions,
            sgl::vk::DeviceFeatures& requestedDeviceFeatures) {
        if (physicalDeviceProperties.apiVersion < VK_API_VERSION_1_1) {
            return false;
        }
        VkPhysicalDeviceDriverProperties physicalDeviceDriverProperties{};
        physicalDeviceDriverProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 deviceProperties2 = {};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &physicalDeviceDriverProperties;
        sgl::vk::getPhysicalDeviceProperties2(physicalDevice, deviceProperties2);
        if (physicalDeviceDriverProperties.driverID != VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
            return false;
        }
        return true;
    };
    device->setPhysicalDeviceCheckCallback(physicalDeviceCheckCallback);
    std::vector<const char*> optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
    std::vector<const char*> requiredDeviceExtensions = { VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME };
    sgl::vk::DeviceFeatures requestedDeviceFeatures{};
    device->createDeviceHeadless(
            instance, requiredDeviceExtensions, optionalDeviceExtensions, requestedDeviceFeatures);
    std::cout << "Running on " << device->getDeviceName() << std::endl;

    // Choose a CUDA device matching the Vulkan device using the CUDA driver API.
    if (!sgl::initializeCudaDeviceApiFunctionTable()) {
        throw std::runtime_error("Error in main: sgl::initializeCudaDeviceApiFunctionTable() returned false.");
    }
    CUresult cuResult = sgl::g_cudaDeviceApiFunctionTable.cuInit(0);
    if (cuResult == CUDA_ERROR_NO_DEVICE) {
        throw std::runtime_error("No CUDA-capable device was found. Disabling CUDA interop support.");
    }
    sgl::checkCUresult(cuResult, "Error in cuInit: ");
    CUdevice cuDevice = 0;
    if (!sgl::vk::getMatchingCudaDevice(device, &cuDevice)) {
        throw std::runtime_error("Error in main: sgl::vk::getMatchingCudaDevice could not find a matching device.");
    }

    // Set the selected CUDA driver API device in the runtime API.
    setCudaDevice(cuDevice);
}

// Checks whether the entries in the passed pointer are linearly increasing.
bool checkIsArrayLinear(
        const sgl::FormatInfo& formatInfo, size_t width, size_t height, void* ptr, std::string& errorMessage) {
    size_t numEntries = width * height * formatInfo.numChannels;
    auto* hostPtr = static_cast<float*>(ptr);
    for (size_t i = 0; i < numEntries; i++) {
        if (hostPtr[i] != float(i)) {
            size_t channelIdx = i % formatInfo.numChannels;
            size_t x = (i / formatInfo.numChannels) % width;
            size_t y = (i / formatInfo.numChannels) / width;
            errorMessage =
                    "Image content mismatch at x=" + std::to_string(x) + ", y=" + std::to_string(y)
                    + ", c=" + std::to_string(channelIdx);
            return false;
        }
    }
    return true;
}

/*
 * Creates an image, to which CUDA writes linearly increasing values in a kernel.
 * Vulkan then copies the image to a device buffer, and finally copies this to a host-visible buffer.
 */
void runTestCase(sgl::vk::Instance* instance, sgl::vk::Device* device) {
    auto format = VK_FORMAT_R32G32B32A32_SFLOAT;
    uint32_t width = 1024;
    uint32_t height = 1024;
    bool useSemaphore = false;

    CUstream cuStream{};
    CUresult cuResult = sgl::g_cudaDeviceApiFunctionTable.cuStreamCreate(&cuStream, CU_STREAM_DEFAULT);
    sgl::checkCUresult(cuResult, "Error in cuStreamCreate: ");

    auto* shaderManager = new sgl::vk::ShaderManagerVk(device);
    auto renderer = new sgl::vk::Renderer(device);

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = width;
    imageSettings.height = height;
    imageSettings.format = format;
    imageSettings.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    imageSettings.exportMemory = true;
    imageSettings.useDedicatedAllocationForExportedMemory = true;
    auto formatInfo = sgl::vk::getImageFormatInfo(format);
    size_t sizeInBytes = imageSettings.width * imageSettings.height * formatInfo.formatSizeInBytes;

    const char* SHADER_STRING_COPY_IMAGE_TO_BUFFER_COMPUTE_FMT = R"(
    #version 450 core
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
    layout(binding = 0, rgba32f) uniform restrict readonly image2D srcImage;
    layout(binding = 1, std430) writeonly buffer DestBuffer {
        vec4 destBuffer[];
    };
    void main() {
        ivec2 srcImageSize = imageSize(srcImage);
        ivec2 idx = ivec2(gl_GlobalInvocationID.xy);
        if (idx.x >= srcImageSize.x || idx.y >= srcImageSize.y) {
            return;
        }
        int linearIdx = idx.x + idx.y * srcImageSize.x;
        destBuffer[linearIdx] = imageLoad(srcImage, idx);
    }
    )";
    auto shaderStages = shaderManager->compileComputeShaderFromStringCached(
            "CopyImageToBufferShader.Compute", SHADER_STRING_COPY_IMAGE_TO_BUFFER_COMPUTE_FMT);

    std::string errorMessage;
    for (int it = 0; it < 1000; it++) {
        // Create semaphore (optional; used only if useSemaphore == true).
        uint64_t timelineValue = 0;
        sgl::vk::SemaphoreVkCudaDriverApiInteropPtr semaphoreVulkan = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
        auto fence = std::make_shared<sgl::vk::Fence>(device);

        // Create image and buffers.
        auto imageViewVulkan = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings));
        sgl::vk::SurfaceCudaExternalMemoryVkPtr imageInteropCuda =
                std::make_shared<sgl::vk::SurfaceCudaExternalMemoryVk>(imageViewVulkan);

        sgl::vk::BufferSettings bufferSettings{};
        bufferSettings.sizeInBytes = sizeInBytes;
        bufferSettings.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        auto bufferVulkan = std::make_shared<sgl::vk::Buffer>(device, bufferSettings); // for copy from image
        bufferSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        bufferSettings.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto stagingBufferVulkan = std::make_shared<sgl::vk::Buffer>(device, bufferSettings); // for copy back to CPU

        // Create command buffer.
        sgl::vk::CommandPoolType commandPoolType;
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        auto commandBuffer = std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType);

        sgl::vk::ComputePipelineInfo computePipelineInfo(shaderStages);
        sgl::vk::ComputePipelinePtr computePipeline = std::make_shared<sgl::vk::ComputePipeline>(
                device, computePipelineInfo);
        auto computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticImageView(imageViewVulkan, 0);
        computeData->setStaticBuffer(bufferVulkan, 1);

        // Write data with CUDA.
        CUsurfObject surfaceObject = imageInteropCuda->getCudaSurfaceObject();
        writeCudaSurfaceObjectIncreasingIndices(cuStream, surfaceObject, imageSettings.width, imageSettings.height);
        if (useSemaphore) {
            timelineValue++;
            semaphoreVulkan->signalSemaphoreCuda(cuStream, timelineValue);
        } else {
            cuResult = sgl::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(cuStream);
            sgl::checkCUresult(cuResult, "Error in cuStreamSynchronize: ");
        }

        // Copy image data to buffer with Vulkan.
        renderer->pushCommandBuffer(commandBuffer);
        commandBuffer->setFence(fence);
        if (useSemaphore) {
            semaphoreVulkan->setWaitSemaphoreValue(timelineValue);
            commandBuffer->pushWaitSemaphore(semaphoreVulkan);
        }
        renderer->beginCommandBuffer();
        renderer->insertImageMemoryBarrier(
                imageViewVulkan->getImage(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_EXTERNAL, renderer->getDevice()->getGraphicsQueueIndex());
        renderer->dispatch(computeData, sgl::uiceil(imageSettings.width, 16u), sgl::uiceil(imageSettings.height, 16u), 1);
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                bufferVulkan);
        bufferVulkan->copyDataTo(stagingBufferVulkan, commandBuffer->getVkCommandBuffer());
        renderer->endCommandBuffer();
        renderer->submitToQueue();
        fence->wait();

        // Check equality to expected values.
        void* hostPtr = stagingBufferVulkan->mapMemory();
        if (!checkIsArrayLinear(formatInfo, imageSettings.width, imageSettings.height, hostPtr, errorMessage)) {
            stagingBufferVulkan->unmapMemory();
            throw std::runtime_error("Memory content mismatched.");
        }
        stagingBufferVulkan->unmapMemory();
    }

    cuResult = sgl::g_cudaDeviceApiFunctionTable.cuStreamDestroy(cuStream);
    sgl::checkCUresult(cuResult, "Error in cuStreamDestroy: ");

    shaderStages = {};
    delete renderer;
    delete shaderManager;
}

int main() {
    sgl::Logfile::get()->createLogfile("LogfileCudaVulkanImageInterop.html", "TestCudaVulkanImageInterop");

    sgl::vk::Instance* instance = nullptr;
    sgl::vk::Device* device = nullptr;
    initDevice(instance, device);

    runTestCase(instance, device);
    std::cout << "All OK." << std::endl;

    delete device;
    delete instance;
    sgl::freeCudaDeviceApiFunctionTable();

    return 0;
}
