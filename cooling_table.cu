#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZEOF_H 32
#define SIZEOF_B 32
#define SIZEOF_TE 32
#define SIZEOF_NE 32
#define N_RESOLUTION 33792
#define SINGLE_TEST (0)
#define RESOLUTION_TEST (0)
cudaTextureObject_t coolTexObj;
cudaArray *cuCoolArray = 0;

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

// Load the cooling_table into the CPU Memory.
void Load_Cooling_Tables(float *cooling_table)
{
    double *scale_height_arr;
    double *ne_arr;
    double *te_arr;
    double *bmag_arr;
    double *cool_arr;

    double scale_height;
    double ne;
    double te;
    double bmag;
    double cool;

    int i = 0;
    int nw = SIZEOF_H; 
    int nx = SIZEOF_TE; // Number of Te data.
    int ny = SIZEOF_NE; // Number of ne data.
    int nz = SIZEOF_B; // Number of Bmag data.

    FILE *infile;

    // Allocate arrays for temperature, electronic density and scale_height data.
    scale_height_arr = (double *)malloc(nw * nx * ny * nz * sizeof(double));
    ne_arr = (double *)malloc(nw * nx * ny * nz * sizeof(double));
    te_arr = (double *)malloc(nw * nx * ny * nz * sizeof(double));
    cool_arr = (double *)malloc(nw * nx * ny *  nz * sizeof(double));
    bmag_arr = (double *)malloc(nw * nx * ny * nz * sizeof(double));

    // Reading the cooling table
    infile = fopen("cooling_table.txt", "r");

    if (infile == NULL)
    {
        printf("Unable to open cooling file.\n");
        exit(1);
    }

    fscanf(infile, "%*[^\n]\n"); // this command is to ignore the first line.
    while (fscanf(infile, "%lf, %lf, %lf, %lf, %lf", &scale_height, &bmag, &ne, &te, &cool) == 5)
    {
        scale_height_arr[i] = scale_height;
        ne_arr[i] = ne;
        te_arr[i] = te;
        bmag_arr[i] = bmag;
        cool_arr[i] = cool;

        i++;
    }

    fclose(infile);
    // copy data from cooling array into the table
    for (i = 0; i < nw * nx * ny * nz; i++)
    {
        cooling_table[i] = float(cool_arr[i]);
    }

    // Free arrays used to read in table data
    free(scale_height_arr);
    free(ne_arr);
    free(te_arr);    
    free(bmag_arr);
    free(cool_arr);
    return;
}

void CreateTexture(void)
{

    float *cooling_table; //Device Array with cooling floats
    // number of elements in each variable
    const int nw = SIZEOF_H; //H
    const int nx = SIZEOF_TE; //te
    const int ny = SIZEOF_NE; //ne
    const int nz = SIZEOF_B; //bmag
    cooling_table = (float *)malloc(nw * nx * ny * nz * sizeof(float));
    Load_Cooling_Tables(cooling_table); //Loading Cooling Values into pointer
    //cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cuda Array
    cudaArray *cuCoolArray;
    //checkCudaErrors(cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx*sizeof(float),ny,nz), 0));
    cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx*ny,nz, nw), 0);
    cudaMemcpy3DParms copyParams = {0};

    //Array creation
    copyParams.srcPtr   = make_cudaPitchedPtr((void *) cooling_table, nx * ny * sizeof(float), nx * ny, nz);
    copyParams.dstArray = cuCoolArray;
    copyParams.extent   = make_cudaExtent(nx * ny, nz, nw);
    copyParams.kind     = cudaMemcpyHostToDevice;
    //checkCudaErrors(cudaMemcpy3D(&copyParams));
    cudaMemcpy3D(&copyParams);
    //Array creation End

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(texRes));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = cuCoolArray;
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    //checkCudaErrors(cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL));}
    cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL);
    return;
}
__global__ void cooling_function(cudaTextureObject_t my_tex, float a0, float a1, float a2, float a3
    #if(RESOLUTION_TEST)
    , double * value
    #endif
    )
{
    float v0, v1, v2, v3, v4;
    double lambda;

    //Values for testing;
    v0 = a0; //H parameter
    v1 = a1; //Bmag parameter
    v2 = a2; //ne parameter
    v3 = a3; //te parameter
    #if(SINGLE_TEST)
    printf("Values you chose:\n");
    printf("scale_height = %f, Bmag = %f, ne = %f, Te = %f\n", v0, v1, v2, v3);
    #endif

    // For the normalized version only.
    const int nw = SIZEOF_H; //Number of H used to generate table
    const int nx = SIZEOF_TE; //Number of te used to generate table
    const int ny = SIZEOF_NE; //Number of ne used to generate table
    const int nz = SIZEOF_B; //Number of Bmag used to generate table
     v0 = (round((v0 - 3) * (nz - 1)/5) + 0.5)/nw; //scale_height
     v1 = (round((v1 - 0) * (nz - 1)/10) + 0.5)/nz; // Bmag
     v4 = ((round((v3 -2) * (nx - 1)/13) + 0.5) + round((v2 - 10) * (ny - 1)/15) * nx)/(nx * ny); //Te + ne

    //For the non normalized version only.
    //lambda = tex3D<float>(coolTexObj, v3 + 0.5f, v2 + 0.5f, v1 + 0.5f); 

    // //For the normalized version only.
    lambda = tex3D<float>(my_tex, v4, v1, v0); 
    #if(SINGLE_TEST)
    printf("Coordinates in texture grid:\n");
    printf("Scale_height = %f, Bmag = %f, ne = %f, te = %f, ne+te = = %f\n", v0, v1, v2, v3, v4);
    printf("Cooling value = %lf\n", lambda);
    #endif

    #if(RESOLUTION_TEST)
        *value = lambda;
    #endif
    return;
}

__global__ void cooling_function_test(cudaTextureObject_t my_tex, double * a0, double * a1, double * a2, double * a3, double * value)
{
    double v0, v1, v2, v3, v4;
    double lambda;
    int i;
    // For the normalized version only.
    const int nw = SIZEOF_H; //Number of H used to generate table
    const int nx = SIZEOF_TE; //Number of te used to generate table
    const int ny = SIZEOF_NE; //Number of ne used to generate table
    const int nz = SIZEOF_B; //Number of Bmag used to generate table
    for (i = 0; i < N_RESOLUTION; i++) {
        v0 = a0[i]; //H parameter
        v1 = a1[i]; //Bmag parameter
        v2 = a2[i]; //ne parameter
        v3 = a3[i]; //te parameter

        //printf("scale_height = %f, Bmag = %f, ne = %f, Te = %f\n", v0, v1, v2, v3);

        v0 = (round((v0 - 3) * (nz - 1)/5) + 0.5)/nw; //scale_height
        v1 = (round((v1 - 0) * (nz - 1)/10) + 0.5)/nz; // Bmag
        v4 = ((round((v3 -2) * (nx - 1)/13) + 0.5) + round((v2 - 10) * (ny - 1)/15) * nx)/(nx * ny); //Te + ne

        // //For the normalized version only.
        lambda = tex3D<float>(my_tex, v4, v1, v0); 
        //printf("lambda = %le\n", lambda);

        value[i] = lambda;
    }
    return;
}

int main()
{
    #if(SINGLE_TEST)
        float read0, read1, read2, read3;
        float loop = 100;
        char str[1];
        CreateTexture();
        while (loop > 1)
        {
            printf("scale_height value:\n");
            scanf("%f", &read0);
            printf("Bmag value:\n");
            scanf("%f", &read1);
            printf("ne value:\n");
            scanf("%f", &read2);
            printf("Te value:\n");
            scanf("%f", &read3);
            cooling_function<<<1, 1>>>(coolTexObj, read0, read1, read2, read3);
            printf("Do you want to read other values? y/n\n");
            scanf("%s", str);
            if (strcmp(str, "n") == 0)
            {
                loop = 0;
            }
        }
        cudaDestroyTextureObject(coolTexObj);
    #elif(RESOLUTION_TEST)
        double *H_test, *B_test, *ne_test, *Te_test, *cool_test;
        H_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        B_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        ne_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        Te_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        cool_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        int i;

        //Allocating memory in device memory.
        double* d_H_test;
        cudaMalloc(&d_H_test, N_RESOLUTION * sizeof(double));
        double * d_B_test;
        cudaMalloc(&d_B_test, N_RESOLUTION * sizeof(double));
        double * d_ne_test;
        cudaMalloc(&d_ne_test, N_RESOLUTION * sizeof(double));
        double * d_Te_test;
        cudaMalloc(&d_Te_test, N_RESOLUTION * sizeof(double));
        double * d_cool_test;
        cudaMalloc(&d_cool_test, N_RESOLUTION * sizeof(double));


        printf("Initializing resolution test reading\n");
        FILE *file_result;
        file_result = fopen("cooling_table_test_cuda.txt", "w");
        FILE *file_height_test;
        file_height_test = fopen("scaleheight_sim.txt", "r");
        FILE *file_e_density_test;
        file_e_density_test = fopen("electronic_density_sim.txt", "r");
        FILE *file_temperature_test;
        file_temperature_test = fopen("electronic_temperature_sim.txt", "r");
        FILE *file_mag_field_test;
        file_mag_field_test = fopen("magnetic_field_sim.txt", "r");
        CreateTexture();
        for (i = 0; fscanf(file_height_test, "%lf", H_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_mag_field_test, "%lf", B_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_e_density_test, "%lf", ne_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_temperature_test, "%lf", Te_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        cudaMemcpy(d_H_test, H_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_test, B_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ne_test, ne_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Te_test, Te_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);

        printf("Reading and getting values from texture memory...\n");

        cooling_function_test<<<1, 1>>>(coolTexObj, d_H_test, d_B_test, d_ne_test, d_Te_test, d_cool_test);
        cudaMemcpy(cool_test, d_cool_test, N_RESOLUTION * sizeof(double), cudaMemcpyDeviceToHost);

        for (i = 0; i < N_RESOLUTION; i++) {
            fprintf(file_result, "%.8e\n", cool_test[i]);
        } 

        free(H_test);
        free(B_test);
        free(ne_test);
        free(Te_test);
        free(cool_test);
        cudaDestroyTextureObject(coolTexObj);

        cudaFree(d_H_test);
        cudaFree(d_B_test);
        cudaFree(d_ne_test);
        cudaFree(d_Te_test);
        cudaFree(d_cool_test);

        fclose(file_height_test);
        fclose(file_e_density_test);
        fclose(file_temperature_test);
        fclose(file_mag_field_test);
        printf("Test Sucessfull\n");
    #endif

    return 0;
}
