#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

//-----------------------------------------------------------------------------
__global__ void kernelFunc(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M, int j) 
{
	int k = bx * 512 + tx;
	
	int m, n;
	float z_r, z_i, w_r, w_i, temp_r, temp_i;
	
	if(j == N/2) {
		int i, p, q;
		
		p = 2*k;
		q = 0;
		
		for(i=0; i<M; i++) {
			q = q + ((p & (1 << i)) >> i) * (1 << (M-i-1));
		}
		
		X_r_d[p] = x_r_d[q];
		X_i_d[p] = x_i_d[q];
		X_r_d[p+1] = x_r_d[q+N/2];
		X_i_d[p+1] = x_i_d[q+N/2];
	}
		
		m = (k*j)%(N/2);
		n = (k/(N/(2*j)))*(N/j) + (k%(N/(2*j)));
		
		w_r =  cos((2*PI*m)/N);
		w_i = -sin((2*PI*m)/N);
		
		z_r = w_r*X_r_d[n+N/(2*j)] - w_i*X_i_d[n+N/(2*j)];
		z_i = w_r*X_i_d[n+N/(2*j)] + w_i*X_r_d[n+N/(2*j)];
		
		temp_r = X_r_d[n];
		temp_i = X_i_d[n];
		
		X_r_d[n] = temp_r + z_r;
		X_i_d[n] = temp_i + z_i;
		X_r_d[n+N/(2*j)] = temp_r - z_r;
		X_i_d[n+N/(2*j)] = temp_i - z_i;
	
}
//-----------------------------------------------------------------------------
__global__ void kernelFunc2(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M, int j) 
{
	int k = bx * 512 + tx;
	
	if(j == N/2) {
		
		int i, p, q;
			
		p = 2*k;
		q = (p & 1) * N/2;
		
		for(i=1; i<M; i+=2) {
			q = q + ((p & (3 << i)) >> i) * (1 << M-i-2); 
		}
		
		X_r_d[q] = x_r_d[p] + x_r_d[p+1];
		X_i_d[q] = x_i_d[p] + x_i_d[p+1];
		X_r_d[q+N/2] = x_r_d[p] - x_r_d[p+1];
		X_i_d[q+N/2] = x_i_d[p] - x_i_d[p+1];
	
	} else {
		int i, m, n;
		float z_r[4], z_i[4], w_r[4], w_i[4], temp_r[4], temp_i[4];

		n = (k/(N/(4*j)))*(N/j) + (k%(N/(4*j)));
		
		for(i=0; i<4; i++){
			
			temp_r[i] = x_r_d[n+i*N/(4*j)];
			temp_i[i] = x_i_d[n+i*N/(4*j)];
			
			m = (j==N/4)?0:(k%(N/(4*j)))*i*j;
			
			w_r[i] =  cos((2*PI*m)/N);
			w_i[i] = -sin((2*PI*m)/N);
			
		}

		z_r[0] = temp_r[0] + temp_r[1] + temp_r[2] + temp_r[3];
		z_i[0] = temp_i[0] + temp_i[1] + temp_i[2] + temp_i[3];

		z_r[1] = temp_r[0] + temp_i[1] - temp_r[2] - temp_i[3]; 
		z_i[1] = temp_i[0] - temp_r[1] - temp_i[2] + temp_r[3];
		
		z_r[2] = temp_r[0] - temp_r[1] + temp_r[2] - temp_r[3]; 
		z_i[2] = temp_i[0] - temp_i[1] + temp_i[2] - temp_i[3];
		
		z_r[3] = temp_r[0] - temp_i[1] - temp_r[2] + temp_i[3]; 
		z_i[3] = temp_i[0] + temp_r[1] - temp_i[2] - temp_r[3];
		
		for(i=0; i<4; i++){
			
			x_r_d[n+i*N/(4*j)] = w_r[i]*z_r[i] - w_i[i]*z_i[i];
			x_i_d[n+i*N/(4*j)] = w_r[i]*z_i[i] + w_i[i]*z_r[i];
			
		}
		
		if(j == N/4) {
			int p, q;
			
			p = 4*k;
			q = 0;
			
			for(i=0; i<M; i+=2) {
				q = q + ((p & (3 << i)) >> i) * (1 << M-i-2);
			}
			
			X_r_d[q] = x_r_d[p];
			X_i_d[q] = x_i_d[p];
			X_r_d[q+N/4] = x_r_d[p+1];
			X_i_d[q+N/4] = x_i_d[p+1];
			X_r_d[q+N/2] = x_r_d[p+2];
			X_i_d[q+N/2] = x_i_d[p+2];
			X_r_d[q+(3*N)/4] = x_r_d[p+3];
			X_i_d[q+(3*N)/4] = x_i_d[p+3];
		}
	}
}
//-----------------------------------------------------------------------------
void gpuKernel_simple(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{	
	int j;
	
	if(N <= 1024) {
		dim3 dimGrid(1,1);
		dim3 dimBlock(N/2,1);
		
		for(j=N/2; j>=1; j=j/2) {
			kernelFunc <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, j);
		}
	} else {
		dim3 dimGrid(N/1024,1);
		dim3 dimBlock(512,1);
		
		for(j=N/2; j>=1; j=j/2) {
			kernelFunc <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, j);
		}
	}	
}
//-----------------------------------------------------------------------------
void gpuKernel_efficient(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{
	int j;

	dim3 dimGrid(N/2048,1);
	dim3 dimBlock(512,1);

	for(j=1; j<N/2; j*=4) {
		kernelFunc2 <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, j);
	}
	
	if(M % 2 == 1) {
		dim3 dimGrid2(N/1024,1);
		dim3 dimBlock2(512,1);
	
		kernelFunc2 <<< dimGrid2, dimBlock2 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M, N/2);
	}
	
}
