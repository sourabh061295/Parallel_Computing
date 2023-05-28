#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_ALPHABETS 26
#define MAX_DG        10
#define PWD_LEN       4

__device__ const unsigned int s_table[] = 
{
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22 ,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20 ,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23 ,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 
};

__device__ const unsigned int k_table[] = 
{
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee ,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501 ,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be ,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821 ,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa ,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8 ,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed ,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a ,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c ,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70 ,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05 ,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665 ,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039 ,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1 ,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1 ,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 
};

__device__ const int digests_4letters[] = 
{
    0x03c82e91, 0xe449ceb2, 0x8d0641a5, 0x70b55a49, 
    0xe26f4639, 0x342a062b, 0x3c9fe0cf, 0x6848c2c8, 
    0xd8235545, 0xb71a8a6a, 0x0832d3c7, 0xe71902fe, 
    0x8fec9be7, 0xa1f82f9e, 0xf7382938, 0xcc585413, 
    0x283b084b, 0xa1858b5a, 0x455602e6, 0xe39cd3da, 
    0xe8ffa0f2, 0x4fd4c83e, 0x24b6e42b, 0xde7df4b0, 
    0xc9418d85, 0xfab897e3, 0x6d04bb34, 0x76f25580, 
    0x4a6bcc95, 0xaddeabc5, 0xa8734bc7, 0xd90072ba, 
    0xa4ae137c, 0xde6d6e7d, 0xc0d262fd, 0xa4b25306, 
    0xf567183e, 0x4530e8ae, 0x35be5f77, 0xe13c6a5e 
};

__device__ void md5(char* message, int length, unsigned int* digest) 
{
	// Encryption init values (key)
   	unsigned int a0 = 0x67452301;
	unsigned int b0 = 0xefcdab89; 
   	unsigned int c0 = 0x98badcfe; 
   	unsigned int d0 = 0x10325476; 

	// Init values
	unsigned int A = a0;
	unsigned int B = b0;
	unsigned int C = c0;
	unsigned int D = d0;

	// Encrypted message
	unsigned int M[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

	// Move the input password to the temp buffer message
	memcpy(M,message,length);
	// Attach a termination character at the end of the input password inside the temp buffer
	((char*)M)[length]=0x80;
	// Add the length * 8 value in last but one position of the array
	M[14]=length*8;
	// Loop 64 times
	for (int i = 0;i < 64;i++) 
	{
		// Encryption logic
		unsigned int F = (B & C) | ((~B) & D);
		unsigned int G = (D & B) | ((~D) & C);
		unsigned int H = B ^ C ^ D;
		unsigned int I = C ^ (B | (~D));
		unsigned int tempD = D;
		D = C;
		C = B;
		unsigned int X = I;
		unsigned int g = (7*i) & 15;
		if (i < 48) { X = H; g = (3*i+5) & 15; }
		if (i < 32) { X = G; g = (5*i+1) & 15; }
		if (i < 16) { X = F; g = i; }

		unsigned int tmp = A + X + k_table[i] + M[g];
		B = B + ((tmp << s_table[i]) | ((tmp & 0xffffffff) >> (32-s_table[i])));
		A = tempD;
	}
	// Store the encrypted password
	digest[0] = a0 + A;
	digest[1] = b0 + B;
	digest[2] = c0 + C;
	digest[3] = d0 + D;
}

__device__ int pwd_num = 0;

__global__ void passwordCrackKernel(char* match_d)
{
    unsigned int dg[4];
    char pwd[PWD_LEN + 1];
    // Get the thread and block ids for iteration
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    int l = threadIdx.y;

    // Index validity check
    if (i + j + k + l <= PWD_LEN * (NUM_ALPHABETS - 1))
    {
        // Initialize the password with a character sequence for comparison
        pwd[0] = 'a' + i;
        pwd[1] = 'a' + j;
        pwd[2] = 'a' + k;
        pwd[3] = 'a' + l;
        pwd[4] = 0;

        // Get the encrypted version of the input password sequence
        md5(pwd, PWD_LEN, dg);
        // Loop through all the digest combination for a match
        for (int ii = 0;ii < MAX_DG ;ii++)
        {
            // Check for a encryption hit
            if (( dg[0] == digests_4letters[ii*4] ) && ( dg[1] == digests_4letters[ii*4+1] ) && ( dg[2] == digests_4letters[ii*4+2] ) && ( dg[3] == digests_4letters[ii*4+3] )) 
            {
                // Use a temporary pointer to point to next available empty location
                char* temp = &match_d[pwd_num];
                // Copy the matching password to the location specified
                memcpy(temp,pwd,PWD_LEN);
                // Increement the password indicator to the next location
                pwd_num += PWD_LEN;
            }
        }
    }
}

// Main function
int main()
{
    // Local variables for host and device
    char match_h[PWD_LEN*MAX_DG];
    char* match_d;

    // Allocate the memory for the pointers
    cudaMalloc((void**) &match_d, PWD_LEN*MAX_DG);

    // CUDA kernel block and grid dimensions
    dim3 threadsPerBlock(NUM_ALPHABETS, NUM_ALPHABETS);
    dim3 blocksPerGrid(NUM_ALPHABETS, NUM_ALPHABETS);

	clock_t begin = clock();

    // Invoke cuda kernel
    passwordCrackKernel<<<blocksPerGrid, threadsPerBlock>>>(match_d);

    // Synchronize all the threads
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(match_h, match_d, PWD_LEN*MAX_DG, cudaMemcpyDeviceToHost);

    // Print the password
    for (int i = 0; i < PWD_LEN*MAX_DG; i = i+PWD_LEN)
    {
        printf("%c", match_h[i]);
        printf("%c", match_h[i+1]);
        printf("%c", match_h[i+2]);
        printf("%c\n", match_h[i+3]);
    }

	printf("\nElapsed time: %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

    // Free the pointers
    cudaFree(match_d);
}