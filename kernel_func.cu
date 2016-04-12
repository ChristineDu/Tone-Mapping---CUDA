#include "utils.h"
#include "stdio.h"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#define tbp 1024
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// #ifdef ZERO_BANK_CONFLICTS
// #define CONFLICT_FREE_OFFSET(n) ((n)>>NUM_BANKS + (n)>>(2*LOG_NUM_BANKS))
// #else
#define CONFLICT_FREE_OFFSET(n) (n)>> LOG_NUM_BANKS
// #endif

//kernel for generating histogram
__global__ void generate_histogram(unsigned int* bins, const float* dIn, const int binNumber, const float lumMin, const float lumMax, const int size) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > size)
    return;

  float range = lumMax - lumMin;
  int bin = ((dIn[i] - lumMin) / range) * binNumber;

  atomicAdd(&bins[bin], 1);
}

 __global__ void generate_histogram_smem(unsigned int* bins, const float* dIn, const int binNumber, const float lumMin, const float lumMax, const int size) {
 
  __shared__ unsigned int temp[1024];
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  int nt = blockDim.x * blockDim.y;

  if (i > size)
    return;

  if(tid < binNumber)
    temp[tid] = 0;
  __syncthreads();

  float range = lumMax - lumMin;
  unsigned int bin = ((dIn[i] - lumMin) / range) * binNumber;
  if(bin<1024)
  	atomicAdd(&temp[bin], 1);
  __syncthreads();

  atomicAdd(&bins[tid],temp[tid]);
 
}

__global__ void generate_binID(const float* dIn, int* out, const int binNumber, const float lumMin, const float lumMax, const int size) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i>size)
  {
    return;
  }

  float range = lumMax - lumMin;
  int bin = ((dIn[i] - lumMin) / range) * binNumber;

  out[i] = bin; 
}

__global__ void update_bins(unsigned int* bins, int* in_binID, int binNumber, const int size){
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x;
   int nt = blockDim.x * blockDim.y;  

   __shared__ unsigned int temp[1024];
   temp[tid] = 0;
   __syncthreads();

  for(int x=tid; x<size; x+=nt){
  if(in_binID[x] == i){;
    temp[tid]++;
  }
  if(in_binID[x] > i){
    break;
  }
  } 
  __syncthreads();

  if(tid == 0){
    for(int x = 0; x<binNumber;x++){
    bins[i] += temp[x];
  }
}


}

//Scan Kernel
__global__ 
void scan_kernel(unsigned int* d_bins, int size) {
    extern __shared__ unsigned int tmp[];
    int mid = threadIdx.x;
    tmp[mid] = d_bins[mid];
    int basein = size, baseout = 0;

    for(int s = 1; s <= size; s *= 2) {
      basein = size - basein;
      baseout = size - basein;
      __syncthreads();    
      tmp[baseout + mid] = tmp[basein + mid];
      if(mid >= s)
        tmp[baseout + mid] += tmp[basein + mid - s];
    
    }
    __syncthreads();   
    d_bins[mid] = tmp[mid];

}
__global__ void  blelloch_no_padding(unsigned int* d_in_array, const size_t numBins)
/*

  \Params:
    * d_in_array - input array of histogram values in each bin. Gets converted
      to cdf by the end of the function.
    * numBins - number of bins in the histogram (Must be < 2*MAX_THREADS_PER_BLOCK)
*/
{

  int thid = threadIdx.x;

  extern __shared__ float temp_array[];

  temp_array[thid] = d_in_array[thid];
  temp_array[thid + numBins/2] = d_in_array[thid + numBins/2];

  __syncthreads();

  // Part 1: Up Sweep, reduction
  int stride = 1;
  for (int d = numBins>>1; d > 0; d>>=1) {

    if (thid < d) {
      int neighbor = stride*(2*thid+1) - 1;
      int index = stride*(2*thid+2) - 1;

      temp_array[index] += temp_array[neighbor];
    }
    stride *=2;
    __syncthreads();
  }
  // Now set last element to identity:
  if (thid == 0)  temp_array[numBins-1] = 0;

  // Part 2: Down sweep
  for (int d=1; d<numBins; d *= 2) {
    stride >>= 1;
    __syncthreads();

    if(thid < d) {
      int neighbor = stride*(2*thid+1) - 1;
      int index = stride*(2*thid+2) - 1;

      float t = temp_array[neighbor];
      temp_array[neighbor] = temp_array[index];
      temp_array[index] += t;
    }
  }

  __syncthreads();

  d_in_array[thid] = temp_array[thid];
  d_in_array[thid + numBins/2] = temp_array[thid + numBins/2];

}
__global__ void  blelloch_padding(unsigned int* d_in_array, const size_t numBins)
/*
  \Params:
    * d_in_array - input array of histogram values in each bin. Gets converted
      to cdf by the end of the function.
    * numBins - number of bins in the histogram (Must be < 2*MAX_THREADS_PER_BLOCK)
*/
{

  int thid = threadIdx.x;
  int ai = thid;
  int bi = thid + (int(numBins)/2);

  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  extern __shared__ float temp_array[];

  temp_array[ai + bankOffsetA] = d_in_array[ai];
  temp_array[bi + bankOffsetB] = d_in_array[bi];

  __syncthreads();

  // Part 1: Up Sweep, reduction
  int stride = 1;
  for (int d = numBins>>1; d > 0; d>>=1) {

    if (thid < d) {

      int neighbor = stride*(2*thid+1) - 1;
      int index = stride*(2*thid+2) - 1;
      neighbor += int(CONFLICT_FREE_OFFSET(neighbor));
      index += int(CONFLICT_FREE_OFFSET(index));

      temp_array[index] += temp_array[neighbor];
    }
    stride *=2;
    __syncthreads();
  }
  // Now set last element to identity:
  if (thid == 0)  temp_array[int(numBins-1) + int(CONFLICT_FREE_OFFSET(numBins-1))] = 0;

  // Part 2: Down sweep
  for (int d=1; d<numBins; d *= 2) {
    stride >>= 1;
    __syncthreads();

    if(thid < d) {
      int neighbor = stride*(2*thid+1) - 1;
      int index = stride*(2*thid+2) - 1;
      neighbor += int(CONFLICT_FREE_OFFSET(neighbor));
      index += int(CONFLICT_FREE_OFFSET(index));

      float t = temp_array[neighbor];
      temp_array[neighbor] = temp_array[index];
      temp_array[index] += t;
    }
  }

  __syncthreads();

  d_in_array[ai] = temp_array[ai + bankOffsetA];
  d_in_array[bi] = temp_array[bi + bankOffsetB];

}

// kernel for calculating minimum value
__global__ void cmin(float *d_in, float *min, int len)
{
  extern __shared__ float smin[]; 

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

  smin[tid] = d_in[i]<d_in[i+len] ? d_in[i] : d_in[i+len];

  __syncthreads();
  if(blockDim.x > 512 && tid<512) {if(smin[tid] > smin[tid+512]) smin[tid] = smin[tid+512];}  __syncthreads();
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
  if(blockDim.x > 512 && tid<512) {if(smax[tid] < smax[tid+512]) smax[tid] = smax[tid+512];}  __syncthreads();
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

int get_max_size(int n, int d) {
    return n%d == 0? n/d : n/d +1;
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


    unsigned int* bins;
    size_t histogramSize = sizeof(unsigned int)*numBins;

    cudaMalloc(&bins, histogramSize);
    cudaMemset(bins, 0, histogramSize);

    int nblocks = ((size + tbp) - 1) / tbp;
    dim3 thread_dim(1024);
//Generate Histogram
    
    generate_histogram_smem<<<nblocks, tbp>>>(bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
    cudaDeviceSynchronize();

//Sort And Search Generate Histogram
/*    int* bin_ID;
    cudaMalloc(&bin_ID, sizeof(int)*size);
    generate_binID<<<nblocks, tbp>>>(d_logLuminance, bin_ID, numBins, min_logLum, max_logLum, size);
    cudaDeviceSynchronize();
    thrust::device_ptr<int> id(bin_ID);
    thrust::sort(id, id + size);
    update_bins<<<1024,1024>>>(bins, bin_ID, numBins, size);
    cudaDeviceSynchronize();
*/
 //    int h_out[100];
 //    cudaMemcpy(&h_out, bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
 //    for(int i = 0; i < 100; i++)
 //        printf("hist out %d\n", h_out[i]);

//Histogram stored in bins

   dim3 scan_block_dim(get_max_size(numBins, thread_dim.x));

   scan_kernel<<<scan_block_dim, thread_dim, sizeof(unsigned int)* 1024 * 2>>>(bins, numBins);
   //blelloch_no_padding<<<scan_block_dim,512,numBins*sizeof(int)>>>(bins, numBins);
   //blelloch_padding<<<scan_block_dim,512,(numBins+64)*sizeof(int)>>>(bins, numBins);

   cudaDeviceSynchronize(); 
    
  //  cudaMemcpy(&h_out, bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
  //  for(int i = 0; i < 100; i++)
  //      printf("cdf out %d\n", h_out[i]);

    
   cudaMemcpy(d_cdf, bins, histogramSize, cudaMemcpyDeviceToDevice);

    
   checkCudaErrors(cudaFree(bins));

}
