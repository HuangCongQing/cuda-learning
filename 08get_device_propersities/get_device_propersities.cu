#include <stdio.h>

int main(){
    int id;
    cudaGetDevice(&id);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, id); //根据这个id找到对应的props属性

    printf("device id: %d \n sms流处理器个数: %d \n capability major: %d \n capability minor: %d  \n warp size: %d \n",
                 id, props.multiProcessorCount, props.major, props.minor, props.warpSize);

    return 0;
}
/* 
device id: 0 
 sms: 16 
 capability major: 7 
 capability minor: 5  
 warp size: 32 
 */