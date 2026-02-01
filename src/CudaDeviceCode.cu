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

#include <string>
#include <stdexcept>

#include <Math/Math.hpp>
#include "CudaDeviceCode.hpp"

static bool isCudaRuntimeApiInitialized = false;

static void errorCheckCuda(cudaError_t cudaError, const char* name) {
    if (cudaError != cudaSuccess) {
        throw std::runtime_error(
            std::string() + "CUDA error (" + std::to_string(int(cudaError)) + ") in " + name + ": "
            + cudaGetErrorString(cudaError));
    }
}

bool getIsCudaRuntimeApiInitialized() {
    return isCudaRuntimeApiInitialized;
}

void setCudaDevice(CUdevice cuDevice) {
    cudaError_t cudaError = cudaSetDevice(cuDevice);
    errorCheckCuda(cudaError, "cudaSetDevice");
    cudaError = cudaFree(nullptr);
    errorCheckCuda(cudaError, "cudaSetDevice");
    isCudaRuntimeApiInitialized = true;
}


template<typename T, int C> struct MakeVec;
template<> struct MakeVec<float, 4> {
    typedef union { float4 data; float arr[4]; } type;
};

template<typename T, int C>
__global__ void writeCudaSurfaceObjectIncreasingIndicesKernel(
        cudaSurfaceObject_t surfaceObject, int width, int height) {
    int idX = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    int idY = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    int linearIdx = (idX + idY * width) * C;
    if (idX >= width || idY >= height) {
        return;
    }
    typedef typename MakeVec<T, C>::type vect;
    vect element;
    for (int c = 0; c < C; c++) {
        element.arr[c] = T(linearIdx + c);
    }
    surf2Dwrite(element.data, surfaceObject, idX * sizeof(vect), idY);
}

void writeCudaSurfaceObjectIncreasingIndices(
        CUstream cuStream, CUsurfObject surfaceObject, size_t width, size_t height) {
    auto iwidth = static_cast<int>(width);
    auto iheight = static_cast<int>(height);
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(sgl::iceil(iwidth, static_cast<int>(blockDim.x)), sgl::iceil(iheight, static_cast<int>(blockDim.y)), 1);
    writeCudaSurfaceObjectIncreasingIndicesKernel<float, 4><<<gridDim, blockDim, 0, cuStream>>>(surfaceObject, iwidth, iheight);
}
