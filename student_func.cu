#include "utils.h"
#include "stdio.h"

#define tbp 1024

//kernel for generating histogram
__global__ void generate_histogram(unsigned int* bins, const float* dIn, const int binNumber, const float lumMin, const float lumMax, const int size) {
  extern __shared__ float sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > size)
    return;

  float range = lumMax - lumMin;
  int bin = ((dIn[i] - lumMin) / range) * binNumber;

  atomicAdd(&bins[bin], 1);
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
__global__ void  blelloch_scan_single_block(unsigned int* d_in_array, const size_t numBins)
/*
  Computes the blelloch exclusive scan for a cumulative distribution function of a
  histogram, as long as the number of Bins of the histogram is not more than twice
  the number of threads per block.

  Also, if numBins < 2*num_threads, then it will full the end of the
  input array with zeros.

  \Params:
    * d_in_array - input array of histogram values in each bin. Gets converted
      to cdf by the end of the function.
    * numBins - number of bins in the histogram (Must be < 2*MAX_THREADS_PER_BLOCK)
*/
{

  int thid = threadIdx.x;

  extern __shared__ float temp_array[];

  // Make sure that we do not read from undefined part of array if it
  // is smaller than the number of threads that we gave defined. If
  // that is the case, the final values of the input array are
  // extended to zero.

  if (thid < numBins) temp_array[thid] = d_in_array[thid];
  else temp_array[thid] = 0;
  if( (thid + numBins/2) < numBins)
    temp_array[thid + numBins/2] = d_in_array[thid + numBins/2];
  else temp_array[thid + numBins/2] = 0;

  __syncthreads();

  // Part 1: Up Sweep, reduction
  // Iterate log_2(numBins) times, and each element adds value 'stride'
  // elements away to its own value.
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
  // Iterate log(n) times. Each thread adds value stride elements away to
  // its own value, and sets the value stride elements away to its own
  // previous value.
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

  if (thid < numBins) d_in_array[thid] = temp_array[thid];
  if ((thid + numBins/2) < numBins)
    d_in_array[thid + numBins/2] = temp_array[thid + numBins/2];

  __syncthreads();
 if(global_id == 0){
	for (int i = 0; i != numBins - 1; i++) {
  	 	*(d_in_array + i) = *(d_in_array + i + 1);
	}
	d_in_array[numBins - 1] += last;
 }

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

//Generate Histogram
    unsigned int* bins;
    size_t histogramSize = sizeof(unsigned int)*numBins;

    cudaMalloc(&bins, histogramSize);
    cudaMemset(bins, 0, histogramSize);

    int nblocks = ((size + tbp) - 1) / tbp;

    dim3 thread_dim(1024);
    dim3 hist_block_dim(get_max_size(size, thread_dim.x));
    generate_histogram<<<nblocks, tbp>>>(bins, d_logLuminance, numBins, min_logLum, max_logLum, size);
    cudaDeviceSynchronize();

    unsigned int h_out[100];
    cudaMemcpy(&h_out, bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 100; i++)
        //printf("hist out %d\n", h_out[i]);

//Histogram stored in bins

   dim3 scan_block_dim(get_max_size(numBins, thread_dim.x));
   printf("scan_block_dim is %d",get_max_size(numBins, thread_dim.x));

   scan_kernel<<<scan_block_dim, thread_dim, sizeof(unsigned int)* 1024 * 2>>>(bins, numBins);
   //nblocks = (numBins/2 - 1) / 512 + 1;
   //blelloch_scan_single_block<<<scan_block_dim,512,numBins*sizeof(int)>>>(bins, numBins);


   cudaDeviceSynchronize(); 
    
   cudaMemcpy(&h_out, bins, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost);
   //for(int i = 0; i < 100; i++)
       //printf("cdf out %d\n", h_out[i]);
    
   cudaMemcpy(d_cdf, bins, histogramSize, cudaMemcpyDeviceToDevice);

    
   checkCudaErrors(cudaFree(bins));

}
