#include "utils.h"
#include "stdio.h"

#define tbp 512

// kernel for calculating minimum value
__global__ void cmin(float *d_in, float *min, int len)
{
  extern __shared__ float smin[]; 

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

  smin[tid] = d_in[i]<d_in[i+len] ? d_in[i] : d_in[i+len];

  __syncthreads();
  if(blockDim.x > 256 && tid<256) {if(smin[tid] > smin[tid+256]) smin[tid] = smin[tid+256];}  __syncthreads();
  if(blockDim.x > 128 && tid<128) {if(smin[tid] > smin[tid+128]) smin[tid] = smin[tid+128];}  __syncthreads();
  if(blockDim.x > 64 && tid<64) {if(smin[tid] > smin[tid+64]) smin[tid] = smin[tid+64];}  __syncthreads();
  if(tid<32) {
    if(blockDim.x > 32 && smin[tid] > smin[tid+32]) smin[tid] = smin[tid+32];
    if(blockDim.x > 16 && smin[tid] > smin[tid+16]) smin[tid] = smin[tid+16];
    if(blockDim.x > 8 && smin[tid] > smin[tid+8]) smin[tid] = smin[tid+8];
    if(blockDim.x > 4 && smin[tid] > smin[tid+4]) smin[tid] = smin[tid+4];
    if(blockDim.x > 2 && smin[tid] > smin[tid+2]) smin[tid] = smin[tid+2];
    if(smin[tid] > smin[tid+1]) smin[tid] = smin[tid+1];
    __syncthreads();
    }  

  if(tid == 0 ) 
  {
    min[blockIdx.x] = smin[0];
  }
}
__global__ void cmax(float *d_in, float *max, int len)
{
  extern __shared__ float smax[]; 

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

  smax[tid] = d_in[i]>d_in[i+len] ? d_in[i] : d_in[i+len];

  __syncthreads();

  if(blockDim.x > 256 && tid<256) {if(smax[tid] < smax[tid+256]) smax[tid] = smax[tid+256];}  __syncthreads();
  if(blockDim.x > 128 && tid<128) {if(smax[tid] < smax[tid+128]) smax[tid] = smax[tid+128];}  __syncthreads();
  if(blockDim.x > 64 && tid<64) {if(smax[tid] < smax[tid+64]) smax[tid] = smax[tid+64];}  __syncthreads();
  if(tid<32) {
    if(blockDim.x > 32 && smax[tid] < smax[tid+32]) smax[tid] = smax[tid+32];
    if(blockDim.x > 16 && smax[tid] < smax[tid+16]) smax[tid] = smax[tid+16];
    if(blockDim.x > 8 && smax[tid] < smax[tid+8]) smax[tid] = smax[tid+8];
    if(blockDim.x > 4 && smax[tid] < smax[tid+4]) smax[tid] = smax[tid+4];
    if(blockDim.x > 2 && smax[tid] < smax[tid+2]) smax[tid] = smax[tid+2];
    if(smax[tid] < smax[tid+1]) smax[tid] = smax[tid+1];
    __syncthreads();
    } 
  if(tid == 0 ) 
  {
    max[blockIdx.x] = smax[0];
  }
}

float get_min_max(const float* const d_in, const size_t size, int flag){
  float result;

  float *dev_in, *dev_out;
  cudaMalloc((void **) &dev_in, size * sizeof(float));
  cudaMemcpy(dev_in, d_in, size * sizeof(float), cudaMemcpyHostToDevice);

  int nblocks = ( (size + tbp - 1) / tbp) /2;
  int nblocks_pre = nblocks;

  while(nblocks>1){
    cudaMalloc((void **) &dev_out, nblocks * sizeof(float));
    if (flag){
      cmin <<<nblocks,tbp,tbp * sizeof(float)>>>(dev_in, dev_out, tbp); 
    }else {
      cmax <<<nblocks,tbp,tbp * sizeof(float)>>>(dev_in, dev_out, tbp); 
    }
    
    cudaFree(dev_in);
    dev_in = dev_out;
    nblocks_pre = nblocks;
    nblocks =( (nblocks + tbp - 1) / tbp) /2;

  }
  cudaMalloc((void **) &dev_out, sizeof(float));
  if(flag){
    cmin <<<1,nblocks_pre/2,nblocks_pre * sizeof(float)/2>>>(dev_in, dev_out, nblocks_pre/2); 
  }else{
    cmax <<<1,nblocks_pre/2,nblocks_pre * sizeof(float)/2>>>(dev_in, dev_out, nblocks_pre/2); 
  }
  
  
  cudaMemcpy(&result, dev_out, sizeof(float),cudaMemcpyDeviceToHost);
  
  cudaFree(dev_out);
  cudaFree(dev_in);
  return result;
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    const size_t size = numRows*numCols;
    min_logLum = get_min_max(d_logLuminance, size, 1);
    max_logLum = get_min_max(d_logLuminance, size, 0);
    
    printf("got min of %f\n", min_logLum);
    printf("got max of %f\n", max_logLum);
    printf("numBins %lu\n", numBins);


}
