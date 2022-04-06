/*! \file cooling_wrapper.cu
 *  \brief Wrapper file for to load CUDA cooling tables. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//Texture and cudaArray declaration.
texture<float, 3, cudaReadModeElementType> coolTexObj;
cudaArray *cuCoolArray = 0;


// Load the cooling_table into the CPU Memory.
void Load_Cooling_Tables(float *cooling_table)
{
    double *ne_arr;
    double *te_arr;
    double *r_arr;
    double *cool_arr;

    double radius;
    double ne;
    double te;
    double cool;

    int i = 0;
    int nx = 70; // Number of Te data.
    int ny = 70; // Number of ne data.
    int nz = 70; // Number of R data.

    FILE *infile;

    // Allocate arrays for temperature, electronic density and radius data.
    ne_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    te_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    cool_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    r_arr = (double *)malloc(nx * ny * nz * sizeof(double));

    // Reading the cooling table
    infile = fopen("cooling_table_log.txt", "r"); // this command is to ignore the first line.

    if (infile == NULL)
    {
        printf("Unable to open cooling file.\n");
        exit(1);
    }

    fscanf(infile, "%*[^\n]\n");
    while (fscanf(infile, "%lf, %lf, %lf, %lf", &radius, &ne, &te, &cool) == 4)
    {
        r_arr[i] = radius;
        ne_arr[i] = ne;
        te_arr[i] = te;
        cool_arr[i] = cool;

        i++;
    }

    fclose(infile);
    // copy data from cooling array into the table
    for (i = 0; i < nx * ny * nz; i++)
    {
        cooling_table[i] = float(cool_arr[i]);
    }

    // Free arrays used to read in table data
    free(ne_arr);
    free(te_arr);
    free(r_arr);
    free(cool_arr);
}

/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures()
{

    float *cooling_table;

    // number of elements in each variable
    const int nx = 70; //te
    const int ny = 70; //ne
    const int nz = 70; //r


    // allocate host arrays to be copied to textures
    cooling_table = (float *)malloc(nx * ny * nz * sizeof(float));

    // Load cooling tables into the host arrays
    Load_Cooling_Tables(cooling_table);

    // Allocate CUDA arrays in device memory
    // The value of 64 in the CUDA channel must be checked, otherwise use 32 for float.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent volumeSize = make_cudaExtent(nx, ny, nz);
    cudaMalloc3DArray(&cuCoolArray, &channelDesc, volumeSize);

    // Copy to device memory the cooling and heating arrays
    // in host memory
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void *)cooling_table, nx * sizeof(float), nx, ny); 
    copyParams.dstArray = cuCoolArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Specify texture reference parameters (same for both tables)
    coolTexObj.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values
    coolTexObj.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values
    coolTexObj.addressMode[2] = cudaAddressModeClamp; // out-of-bounds fetches return border values
    coolTexObj.filterMode = cudaFilterModeLinear;     // bi-linear interpolation
    coolTexObj.normalized = true;                     // Normalization of logarithm scale going from 0 to 1

    // Command to bind the array into the texture
    cudaBindTextureToArray(coolTexObj, cuCoolArray);
    // Free the memory associated with the cooling tables on the host
    free(cooling_table);
}

void Free_Cuda_Textures()
{
    // unbind the cuda textures
    cudaUnbindTexture(coolTexObj);
    // Free the device memory associated with the cuda arrays
    cudaFreeArray(cuCoolArray);
}

//Function used to interpolate the values of the cooling table.
__global__ void cooling_function()
{
    float v1, v2, v3, lambda;

    //Values for testing;
    v1 = 9.22; //r parameter
    v2 = 12.58; //ne parameter
    v3 = 7.57; //te parameter
    printf("a = %f, b = %f, c = %f\n", v1, v2, v3);

    /*For the non normalized version only.
     The remapping formula goes (variable - initial_value) * (N - 1)/(max_value - init_value)*/
    // const int nx = 70; //Number of te used to generate table
    // const int ny = 70; //Number of ne used to generate table
    // const int nz = 70; //Number of r used to generate table
    //v1 = round((v1 - 6) * (nz - 1)/6);
    //v2 = round((v2 - 12) * (ny - 1)/8);
    //v3 = round((v3 - 6) * (nx - 1)/4);
    //printf("a = %f, b = %f, c = %f\n", v1, v2, v3);

    // // For the normalized version only.
    const int nx = 70; //Number of te used to generate table
    const int ny = 70; //Number of ne used to generate table
    const int nz = 70; //Number of r used to generate table
     v1 = (round((v1 - 6) * (nz - 1)/6) + 0.5)/nz;
     v2 = (round((v2 - 12) * (ny - 1)/8) + 0.5 )/ny;
     v3 = (round((v3 - 6)* (nx - 1)/4) + 0.5)/nx; 
    printf("a = %f, b = %f, c = %f\n", v1, v2, v3);

    //For the non normalized version only.
    //lambda = tex3D<float>(coolTexObj, v3 + 0.5f, v2 + 0.5f, v1 + 0.5f); 

    // //For the normalized version only.
     lambda = tex3D<float>(coolTexObj, v3, v2, v1); 

    printf("Lambda = %lf\n", lambda);
    return;
}

int main()
{
    Load_Cuda_Textures();
    cooling_function<<<1, 1>>>();

    Free_Cuda_Textures();

    return 0;
}
