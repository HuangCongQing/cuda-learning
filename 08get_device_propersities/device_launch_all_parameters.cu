#include "device_launch_parameters.h"
#include <iostream>


// all params path : /usr/local/cuda-11.4/targets/x86_64-linux/include/driver_types.h
int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++) // 遍历所有的显卡
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量totalGlobalMem： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量multiProcessorCount：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小sharedMemPerBlock：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数maxThreadsPerBlock：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量regsPerBlock： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数maxThreadsPerMultiProcessor：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数(maxThreadsPerMultiProcessor/32)：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量multiProcessorCount： " << devProp.multiProcessorCount << std::endl;
        std::cout << "luidDeviceNodeMask:  " << devProp.luidDeviceNodeMask << std::endl;
        std::cout << "totalGlobalMem:  " << devProp.totalGlobalMem << std::endl;
        std::cout << "sharedMemPerBlock:  " << devProp.sharedMemPerBlock << std::endl;
        std::cout << "regsPerBlock:  " << devProp.regsPerBlock << std::endl;
        std::cout << "warpSize:  " << devProp.warpSize << std::endl;
        std::cout << "memPitch:  " << devProp.memPitch << std::endl;
        std::cout << "maxThreadsPerBlock:  " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "maxThreadsDim:  " << devProp.maxThreadsDim << std::endl;
        std::cout << "maxGridSize:  " << devProp.maxGridSize << std::endl;
        std::cout << "clockRate:  " << devProp.clockRate << std::endl;
        std::cout << "totalConstMem:  " << devProp.totalConstMem << std::endl;
        std::cout << "major:  " << devProp.major << std::endl;
        std::cout << "minor:  " << devProp.minor << std::endl;
        std::cout << "textureAlignment:  " << devProp.textureAlignment << std::endl;
        std::cout << "texturePitchAlignment:  " << devProp.texturePitchAlignment << std::endl;
        std::cout << "deviceOverlap:  " << devProp.deviceOverlap << std::endl;
        std::cout << "multiProcessorCount:  " << devProp.multiProcessorCount << std::endl;
        std::cout << "kernelExecTimeoutEnabled:  " << devProp.kernelExecTimeoutEnabled << std::endl;
        std::cout << "integrated:  " << devProp.integrated << std::endl;
        std::cout << "computeMode:  " << devProp.computeMode << std::endl;
        std::cout << "maxTexture1D:  " << devProp.maxTexture1D << std::endl;
        std::cout << "maxTexture1DMipmap:  " << devProp.maxTexture1DMipmap << std::endl;
        std::cout << "maxTexture1DLinear:  " << devProp.maxTexture1DLinear << std::endl;
        std::cout << "maxTexture2D:  " << devProp.maxTexture2D << std::endl;
        std::cout << "maxTexture2DMipmap:  " << devProp.maxTexture2DMipmap << std::endl;
        std::cout << "maxTexture2DLinear:  " << devProp.maxTexture2DLinear << std::endl;
        std::cout << "maxTexture2DGather:  " << devProp.maxTexture2DGather << std::endl;
        std::cout << "maxTexture3D:  " << devProp.maxTexture3D << std::endl;
        std::cout << "maxTexture3DAlt:  " << devProp.maxTexture3DAlt << std::endl;
        std::cout << "maxTextureCubemap:  " << devProp.maxTextureCubemap << std::endl;
        std::cout << "maxTexture1DLayered:  " << devProp.maxTexture1DLayered << std::endl;
        std::cout << "maxTexture2DLayered:  " << devProp.maxTexture2DLayered << std::endl;
        std::cout << "maxTextureCubemapLayered:  " << devProp.maxTextureCubemapLayered << std::endl;
        std::cout << "maxSurface1D:  " << devProp.maxSurface1D << std::endl;
        std::cout << "maxSurface2D:  " << devProp.maxSurface2D << std::endl;
        std::cout << "maxSurface3D:  " << devProp.maxSurface3D << std::endl;
        std::cout << "maxSurface1DLayered:  " << devProp.maxSurface1DLayered << std::endl;
        std::cout << "maxSurface2DLayered:  " << devProp.maxSurface2DLayered << std::endl;
        std::cout << "maxSurfaceCubemap:  " << devProp.maxSurfaceCubemap << std::endl;
        std::cout << "maxSurfaceCubemapLayered:  " << devProp.maxSurfaceCubemapLayered << std::endl;
        std::cout << "surfaceAlignment:  " << devProp.surfaceAlignment << std::endl;
        std::cout << "concurrentKernels:  " << devProp.concurrentKernels << std::endl;
        std::cout << "ECCEnabled:  " << devProp.ECCEnabled << std::endl;
        std::cout << "pciBusID:  " << devProp.pciBusID << std::endl;
        std::cout << "pciDeviceID:  " << devProp.pciDeviceID << std::endl;
        std::cout << "pciDomainID:  " << devProp.pciDomainID << std::endl;
        std::cout << "tccDriver:  " << devProp.tccDriver << std::endl;
        std::cout << "asyncEngineCount:  " << devProp.asyncEngineCount << std::endl;
        std::cout << "unifiedAddressing:  " << devProp.unifiedAddressing << std::endl;
        std::cout << "memoryClockRate:  " << devProp.memoryClockRate << std::endl;
        std::cout << "memoryBusWidth:  " << devProp.memoryBusWidth << std::endl;
        std::cout << "l2CacheSize:  " << devProp.l2CacheSize << std::endl;
        std::cout << "persistingL2CacheMaxSize:  " << devProp.persistingL2CacheMaxSize << std::endl;
        std::cout << "maxThreadsPerMultiProcessor:  " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "streamPrioritiesSupported:  " << devProp.streamPrioritiesSupported << std::endl;
        std::cout << "globalL1CacheSupported:  " << devProp.globalL1CacheSupported << std::endl;
        std::cout << "localL1CacheSupported:  " << devProp.localL1CacheSupported << std::endl;
        std::cout << "sharedMemPerMultiprocessor:  " << devProp.sharedMemPerMultiprocessor << std::endl;
        std::cout << "regsPerMultiprocessor:  " << devProp.regsPerMultiprocessor << std::endl;
        std::cout << "managedMemory:  " << devProp.managedMemory << std::endl;
        std::cout << "isMultiGpuBoard:  " << devProp.isMultiGpuBoard << std::endl;
        std::cout << "multiGpuBoardGroupID:  " << devProp.multiGpuBoardGroupID << std::endl;
        std::cout << "hostNativeAtomicSupported:  " << devProp.hostNativeAtomicSupported << std::endl;
        std::cout << "singleToDoublePrecisionPerfRatio:  " << devProp.singleToDoublePrecisionPerfRatio << std::endl;
        std::cout << "pageableMemoryAccess:  " << devProp.pageableMemoryAccess << std::endl;
        std::cout << "concurrentManagedAccess:  " << devProp.concurrentManagedAccess << std::endl;
        std::cout << "computePreemptionSupported:  " << devProp.computePreemptionSupported << std::endl;
        std::cout << "canUseHostPointerForRegisteredMem:  " << devProp.canUseHostPointerForRegisteredMem << std::endl;
        std::cout << "cooperativeLaunch:  " << devProp.cooperativeLaunch << std::endl;
        std::cout << "cooperativeMultiDeviceLaunch:  " << devProp.cooperativeMultiDeviceLaunch << std::endl;
        std::cout << "sharedMemPerBlockOptin:  " << devProp.sharedMemPerBlockOptin << std::endl;
        std::cout << "pageableMemoryAccessUsesHostPageTables:  " << devProp.pageableMemoryAccessUsesHostPageTables << std::endl;
        std::cout << "directManagedMemAccessFromHost:  " << devProp.directManagedMemAccessFromHost << std::endl;
        std::cout << "maxBlocksPerMultiProcessor:  " << devProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "accessPolicyMaxWindowSize:  " << devProp.accessPolicyMaxWindowSize << std::endl;
        std::cout << "reservedSharedMemPerBlock:  " << devProp.reservedSharedMemPerBlock << std::endl;
        std::cout << "======================================================" << std::endl;     
        
    }
    return 0;
}
/*  
使用GPU device 0: NVIDIA GeForce RTX 3090
设备全局内存总量： 24106MB
SM的数量：82
每个线程块的共享内存大小：48 KB
每个线程块的最大线程数：1024
设备上一个线程块（Block）种可用的32位寄存器数量： 65536
每个EM的最大线程数：1536
每个EM的最大线程束数：48
设备上多处理器的数量： 82

*/