#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<assert.h>


using namespace::std;

__global__ 
void g_mat_mul( int *a, int *b, int *c, int m){
    //Kernel for matrix multiplication on GPU

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  for(int i=0; i < m; i++){
    //printf("\nValue of a is %d at index %d", a[(index/m)*m + i], (index/m)*m + i);
    //printf("\nValue of b is %d at index %d", b[(index%3) + i], (index%m) + i);
    c[index] += a[(index/m)*m + i] * b[(index % m) + i*m];
  }
  //printf("\nValue for C at thread %d is %d\n", index, c[index]);
}

__global__
void gpu_transpose(int *mat, int *res, int m){
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int row = index/m;
  int col = index%m;

  res[col*m + row] = mat[index];
}

void cpu_matrix_mul(int *a, int *b, int *c, int m){
  for( int i = 0; i < m; i++){
    for( int j=0; j < m; j++){
      for( int l = 0; l < m; l++){
        c[i + m*j] += a[i + m*l]*b[m*l + j];
      }
    }
  }
}


int main(void){
  int m = 1024;
  int size;


  size = m*m;

  //Allocated Memory on device for matrices
  // a = (m*n) , b = (n*k), c = (m*k)
  int *device_a, *device_b, *device_c; //cudaOperation
  int *a, *b, *c;

  a = (int *) malloc(sizeof(int)*size);
  b = (int *) malloc(sizeof(int)*size);
  c = (int *) malloc(sizeof(int)*size);

  cudaMallocManaged(&device_a, sizeof(int)*size);
  cudaMallocManaged(&device_b , sizeof(int)*size);
  cudaMallocManaged(&device_c , sizeof(int)*size);

  //Randomly initialize Matrices

  //Randomly initialize A
  std::cout<<"Initialized A";
  for(int i=0; i < size; i++){
    if(i%(m+1) == 0){
      device_a[i] =1 ;
      a[i] = 1;
    }else{
      device_a[i] = 0;
      a[i] = 0;
    }
  }
  std::cout<<"Initialized B";
  for(int i=0; i < size; i++){
    device_b[i] = i+1;
    b[i] = i + 1;
  }
    
  float gpu_elapsed_time, cpu_elapsed_time;
  cudaEvent_t start, stop;  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);

  g_mat_mul<<<m,m>>>(
        device_a, device_b, device_c, m
  );
  //cpu_matrix_mul(a,b,c,m,n,k);
  cudaDeviceSynchronize();

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  
  cudaEventElapsedTime(&gpu_elapsed_time, start, stop);
  printf("\nTime elapsed on matrix multiplication on GPU: %f ms.\n\n", gpu_elapsed_time);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
    
  int *res;
  cudaMallocManaged(&res, sizeof(int)*size);
  cudaEventRecord(start,0);
  //cpu_matrix_mul(a,b,c,m);
  gpu_transpose<<<m,m>>>(
        device_a, res, m  
  );
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_elapsed_time, start, stop);
  printf("\nTime elapsed on matrix transpose on GPU: %f ms.\n\n", cpu_elapsed_time);
}
