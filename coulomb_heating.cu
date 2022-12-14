/*! \file cooling_wrapper.cu
 *  \brief Wrapper file for to load CUDA cooling tables. */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

// Load the cooling_table into the CPU Memory.
void Load_Coulomb_Tables(float *coulomb_table)
{
    double *ne_arr;
    double *te_arr;
    double *ti_arr;
    double *coulomb_arr;

    double ne;
    double te;
    double ti;
    double coulomb;

    int i = 0;
    int nx = 100; // Number of Te data.
    int ny = 100; // Number of ne data.
    int nz = 100; // Number of Bmag data.

    FILE *infile;

    // Allocate arrays for temperature, electronic density and radius data.
    ne_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    te_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    coulomb_arr = (double *)malloc(nx * ny *  nz * sizeof(double));
    ti_arr = (double *)malloc(nx * ny * nz * sizeof(double));

    // Reading the cooling table
    infile = fopen("source_coulomb.txt", "r"); // this command is to ignore the first line.

    if (infile == NULL)
    {
        printf("Unable to open cooling file.\n");
        exit(1);
    }

    fscanf(infile, "%*[^\n]\n");
    while (fscanf(infile, "%lf, %lf, %lf, %lf", &ne, &ti, &te, &coulomb) == 4)
    {
        ne_arr[i] = ne;
        te_arr[i] = te;
        ti_arr[i] = ti;
        coulomb_arr[i] = coulomb;

        i++;
    }

    fclose(infile);
    // copy data from cooling array into the table
    for (i = 0; i < nx * ny * nz; i++)
    {
        coulomb_table[i] = float(coulomb_arr[i]);
    }

    // Free arrays used to read in table data
    free(ne_arr);
    free(te_arr);    
    free(ti_arr);
    free(coulomb_arr);
}

void CreateTexture()
{

    float *coulomb_table; //Device Array with cooling floats
    // number of elements in each variable
    const int nx = 100; //te
    const int ny = 100; //ti
    const int nz = 100; //ne
    coulomb_table = (float *)malloc(nx* ny * nz * sizeof(float));
    Load_Coulomb_Tables(coulomb_table); //Loading Cooling Values into pointer
    //cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cuda Array
    cudaArray *cuCoulombArray;
    //checkCudaErrors(cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx*sizeof(float),ny,nz), 0));
    cudaMalloc3DArray(&cuCoulombArray, &channelDesc, make_cudaExtent(nx,ny,nz), 0);
    cudaMemcpy3DParms copyParams = {0};

    //Array creation
    copyParams.srcPtr   = make_cudaPitchedPtr((void *) coulomb_table, nx*sizeof(float), nx, ny);
    copyParams.dstArray = cuCoulombArray;
    copyParams.extent   = make_cudaExtent(nx,ny,nz);
    copyParams.kind     = cudaMemcpyHostToDevice;
    //checkCudaErrors(cudaMemcpy3D(&copyParams));
    cudaMemcpy3D(&copyParams);
    //Array creation End

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(texRes));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = cuCoulombArray;
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    //checkCudaErrors(cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL));}
    cudaCreateTextureObject(&coulombTexObj, &texRes, &texDescr, NULL);
}
__global__ void coulomb_function(cudaTextureObject_t my_tex, float a1, float a2, float a3)
{
    float v1, v2, v3, lambda;

    //Values for testing;
    v1 = a1; //ne parameter
    v2 = a2; //ti parameter
    v3 = a3; //te parameter
    printf("Values you chose:\n");
    printf("ne = %f, ti= %f, Te = %f\n", v1, v2, v3);

    //For the non normalized version only.
    //The remapping formula goes (variable - initial_value) * (N - 1)/(max_value - init_value)
    // const int nx = 70; //Number of te used to generate table
    // const int ny = 70; //Number of ne used to generate table
    // const int nz = 70; //Number of r used to generate table
    //v1 = round((v1 - 6) * (nz - 1)/6);
    //v2 = round((v2 - 12) * (ny - 1)/8);
    //v3 = round((v3 - 6) * (nx - 1)/4);
    //printf("a = %f, b = %f, c = %f\n", v1, v2, v3);

    // For the normalized version only.
    const int nx = 100; //Number of te used to generate table
    const int ny = 100; //Number of ti used to generate table
    const int nz = 100; //Number of ne used to generate table
     v1 = (round((v1 - 12) * (nz - 1)/10) + 0.5 )/nz;
     v2 = (round((v2 - 4) * (ny - 1)/11) + 0.5 )/ny;
     v3 = (round((v3 - 4) * (nx - 1)/11) + 0.5 )/nx;

    printf("Coordinates in texture grid:\n");
    printf("ne = %f, ti = %f, Te = %f\n", v1, v2, v3);

    //For the non normalized version only.
    //lambda = tex3D<float>(coolTexObj, v3 + 0.5f, v2 + 0.5f, v1 + 0.5f); 

    // //For the normalized version only.
    lambda = tex3D<float>(my_tex, v3, v2, v1); 
    printf("Cooling value = %lf\n", lambda);
    return;
}

int main()
{
    float read1, read2, read3;
    float loop = 100;
    char str[1];
    CreateTexture();
    while (loop > 1)
    {
	    printf("ne value:\n");
	    scanf("%f", &read1);
	    printf("ti value:\n");
	    scanf("%f", &read2);
	    printf("Te value:\n");
	    scanf("%f", &read3);
	    coulomb_function<<<1, 1>>>(coulombTexObj, read1, read2, read3);
        sleep(1);
	    printf("Do you want to read other values? y/n\n");
	    scanf("%s", str);
	    if (strcmp(str, "n") == 0)
	    {
	    	loop = 0;
	    }
	}
    cudaDestroyTextureObject(coulombTexObj);
    return 0;
}
//DEPRECATED Texture Reference in CUDA 11.0
/*
//Texture and cudaArray declaration.
 texture<float, 3, cudaReadModeElementType> coolTexObj;
cudaArray *cuCoolArray = 0;


// Load the cooling_table into the CPU Memory.
void Load_Cooling_Tables(float *cooling_table)
{
    double *ne_arr;
    double *te_arr;
    double *bmag_arr;
    double *cool_arr;

    double ne;
    double te;
    double bmag;
    double cool;

    int i = 0;
    int nx = 100; // Number of Te data.
    int ny = 100; // Number of ne data.
    int nz = 100; // Number of Bmag data.

    FILE *infile;

    // Allocate arrays for temperature, electronic density and radius data.
    ne_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    te_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    cool_arr = (double *)malloc(nx * ny *  nz * sizeof(double));
    bmag_arr = (double *)malloc(nx * ny * nz * sizeof(double));

    // Reading the cooling table
    infile = fopen("cooling_table_log_mag.txt", "r"); // this command is to ignore the first line.

    if (infile == NULL)
    {
        printf("Unable to open cooling file.\n");
        exit(1);
    }

    fscanf(infile, "%*[^\n]\n");
    while (fscanf(infile, "%lf, %lf, %lf, %lf", &bmag, &ne, &te, &cool) == 4)
    {
        ne_arr[i] = ne;
        te_arr[i] = te;
        bmag_arr[i] = bmag;
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
    free(bmag_arr);
    free(cool_arr);
}

 // \brief Load the Cloudy cooling tables into texture memory on the GPU. 
void Load_Cuda_Textures()
{

    float *cooling_table;

    // number of elements in each variable
    const int nx = 100; //te
    const int ny = 100; //ne
    const int nz = 100; //bmag


    // allocate host arrays to be copied to textures
    cooling_table = (float *)malloc(nx* ny * nz * sizeof(float));

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
__global__ void cooling_function(float a1, float a2, float a3)
{
    float v1, v2, v3, lambda;

    //Values for testing;
    v1 = a1; //Bmag parameter
    v2 = a2; //ne parameter
    v3 = a3; //te parameter
    printf("Values you chose:\n");
    printf("Bmag = %f, ne = %f, Te = %f\n", v1, v2, v3);

    //For the non normalized version only.
    //The remapping formula goes (variable - initial_value) * (N - 1)/(max_value - init_value)
    // const int nx = 70; //Number of te used to generate table
    // const int ny = 70; //Number of ne used to generate table
    // const int nz = 70; //Number of r used to generate table
    //v1 = round((v1 - 6) * (nz - 1)/6);
    //v2 = round((v2 - 12) * (ny - 1)/8);
    //v3 = round((v3 - 6) * (nx - 1)/4);
    //printf("a = %f, b = %f, c = %f\n", v1, v2, v3);

    // For the normalized version only.
    const int nx = 100; //Number of te used to generate table
    const int ny = 100; //Number of ne used to generate table
    const int nz = 100; //Number of Bmag used to generate table
     v1 = (round((v1 - 0.1) * (nz - 1)/9.99) + 0.5)/nz;
     v2 = (round((v2 - 12) * (ny - 1)/10) + 0.5 )/ny;
     v3 = (round((v3 - 4) * (nx - 1)/11) + 0.5 )/nx;

    printf("Coordinates in texture grid:\n");
    printf("Bmag = %f, ne = %f, Te = %f\n", v1, v2, v3);

    //For the non normalized version only.
    //lambda = tex3D<float>(coolTexObj, v3 + 0.5f, v2 + 0.5f, v1 + 0.5f); 

    // //For the normalized version only.
    lambda = tex3D<float>(coolTexObj, v3, v2, v1); 
    printf("Cooling value = %lf\n", lambda);
    return;
}

int main()
{
    float read1, read2, read3;
    float loop = 100;
    char str[1];
    Load_Cuda_Textures();
    while (loop > 1)
    {
	    printf("Bmag value:\n");
	    scanf("%f", &read1);
	    printf("ne value:\n");
	    scanf("%f", &read2);
	    printf("Te value:\n");
	    scanf("%f", &read3);
	    cooling_function<<<1, 1>>>(read1, read2, read3);
        sleep(1);
	    printf("Do you want to read other values? y/n\n");
	    scanf("%s", str);
	    if (strcmp(str, "n") == 0)
	    {
	    	loop = 0;
	    }
	}
    Free_Cuda_Textures();

    return 0;
}*/