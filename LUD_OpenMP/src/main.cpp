/**
 * Please do not change this file
 */

#include <iostream>
#include <omp.h>

#if LUD_PAR
  #include "lud_par.cpp"
#elif LUD_OPT
  #include "lud_opt.cpp"
#else
  #include "lud_seq.cpp"
#endif

// Matrix input
int matrix_dim = 1024;
int num_elements = matrix_dim*matrix_dim;
float *matrix;

/**
 * Calculate time difference
 */
static int64_t diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    int64_t d;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    d = temp.tv_sec*1000000000+temp.tv_nsec;
    return d;
}

int main (int argc, char *argv[] )  {

    // Checking args
    if ( argc < 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        exit(1);
    }

    // Reading matrix input from file
    const char *input_file = argv[1];
    FILE *fp = NULL;
    fp = fopen(input_file, "rb");
    if ( fp == NULL) {
        fprintf(stderr,"Could not open file: %s \n", input_file);
        exit(1);
    }
    printf("Reading input matrix ...\n");
    fscanf(fp, "%d\n", &matrix_dim);
    num_elements = matrix_dim * matrix_dim;

    // allocate memory
    matrix = (float*) malloc(sizeof(float)*num_elements);
    if ( matrix == NULL) {
        fclose(fp);
        fprintf(stderr,"Could not allocate memory\n");
        exit(1);
    }
    for (int i=0; i < matrix_dim; i++) {
        for (int j=0; j < matrix_dim; j++) {
            fscanf(fp, "%f ", matrix+i*matrix_dim+j);
        }
    }
    printf("Matrix size = %d x %d \n", matrix_dim, matrix_dim);
    printf("Matrix data size = %d KB \n\n", sizeof(float)*num_elements/1024);


    // Execute
    #if LUD_PAR
        printf("Executing lud_par: compute_lud parallel ...\n");
    #elif LUD_OPT
        printf("Executing  lud_opt: compute_lud optimised ...\n");
    #else
        printf("Executing  lud_seq: compute_lud sequential ...\n");
    #endif

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME,&start);

    compute_lud(matrix, matrix_dim);

    clock_gettime(CLOCK_REALTIME,&end);


    double time = (double) diff(start,end)/1000000;
    printf("\nExecution time: \t %.3f ms\n", time);


    printf("\nSaving output matrix ...\n");
    FILE *fo = fopen("output.dat", "w");
    for (int i=0; i<matrix_dim;i++) {
        for (int j=0; j<matrix_dim;j++)
            fprintf(fo, "%f ", matrix[i*matrix_dim+j]);
        fprintf(fo, "\n");
    }

    fclose(fp);
    fclose(fo);

    return 0;
}