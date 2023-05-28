
// This is a copy of the sequential code, please try to optimise this version
// as much as possible. You can change the code, algorthim, use any openmp directives, number of threads etc ..
void compute_lud(float *a, int size) 
{
    int i,j,k;
    float sum;

    int threads = 26;
    printf("%d", threads);
    omp_set_num_threads(threads);
    

    #pragma omp parallel for private(i,j,k,sum)
    for (i=0; i<size; i++) 
    {
        #pragma omp parallel for private(j,k,sum)
        for (j=i; j<size; j++) 
        {
            sum=a[i*size+j];
            #pragma omp parallel for private(k) reduction(-:sum)
            for (k=0; k<i; k++) 
            {
                sum -= a[i * size + k] * a[k * size + j];
            }
            a[i*size+j]=sum;
        }

        #pragma omp parallel for private(j,k,sum)
        for (j=i+1;j<size; j++)
        {
            sum=a[j*size+i];
            #pragma omp parallel for private(k) reduction(-:sum)
            for (k=0; k<i; k++) 
            {
                sum -= a[j * size + k] * a[k * size + i];
            }
            a[j*size+i]=sum/a[i*size+i];
        }
    }
}