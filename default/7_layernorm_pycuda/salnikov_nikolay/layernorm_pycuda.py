import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

LAYERNORM_KERNEL = """
#define FINAL_MASK 0xffffffff

__inline__ __device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(FINAL_MASK, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val)
{
    __shared__ float shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads(); 

    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__global__ void layernorm_kernel(float* input, float* output, float* gamma, float* beta, int row_size, float eps)
{
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    
    float* row_ptr = input + row_idx * row_size;
    float* out_ptr = output + row_idx * row_size;
    
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    double sum = 0.0;
    for (int i = tid; i < row_size; i += block_dim)
    {
        sum += (double)row_ptr[i];
    }
    
    float block_sum = blockReduceSum((float)sum);
    
    if (tid == 0)
    {
        s_mean = block_sum / row_size;
    }
    __syncthreads(); 
    
    double var_sum = 0.0;
    float local_mean = s_mean; 
    
    for (int i = tid; i < row_size; i += block_dim)
    {
        float diff = row_ptr[i] - local_mean;
        var_sum += (double)(diff * diff);
    }
    
    float block_var_sum = blockReduceSum((float)var_sum);
    
    if (tid == 0)
    {
        float var = block_var_sum / row_size;
        s_inv_std = rsqrtf(var + eps); 
    }
    __syncthreads(); 
    
    float local_inv_std = s_inv_std; 
    
    for (int i = tid; i < row_size; i += block_dim)
    {
        out_ptr[i] = ((row_ptr[i] - local_mean) * local_inv_std) * gamma[i] + beta[i];
    }
}
"""

module = SourceModule(LAYERNORM_KERNEL)
layernorm_gpu = module.get_function("layernorm_kernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
        
    num_elements = input.size
    num_rows = num_elements // row_size
    
    output = gpuarray.empty_like(input)
    
    threads_per_block = min(row_size, 1024)
    threads_per_block = int(np.ceil(threads_per_block / 32.0) * 32)
    if threads_per_block == 0: threads_per_block = 32

    grid_size = (num_rows, 1, 1)
    
    layernorm_gpu(
        input, output, gamma, beta,
        np.int32(row_size), np.float32(eps),
        block=(threads_per_block, 1, 1),
        grid=grid_size
    )
    
    return output