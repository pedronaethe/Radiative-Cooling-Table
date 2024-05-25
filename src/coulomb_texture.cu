#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZEOF_TE 201
#define SIZEOF_NE 201
#define SIZEOF_TI 201


#define SINGLE_VALUE (1)
#define RESOLUTION_TEST (0)
#define N_RESOLUTION (72192)

cudaTextureObject_t coulTexObj;
cudaArray *cucoulArray = 0;

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

// Load the coulomb_table into the CPU Memory.
void Load_coulomb_Tables(float *coulomb_table)
{

    int i = 0;
    int nx = SIZEOF_NE; // Number of ne data.
    int ny = SIZEOF_TI; // Number of Ti data.
    int nz = SIZEOF_TE; // Number of Te data.

    FILE *infile;
    double value;

    fprintf(stderr, "Loading Table...\n");

    // Reading the coulomb table
    infile = fopen("../tables/coulomb_table_200.bin", "rb");

    if (infile == NULL)
    {
        printf("Unable to open coulomb file.\n");
        exit(1);
    }
    fprintf(stderr, "Reading Data...\n");
    // copy data from coulomb array into the table
    for (i = 0; i < nx * ny * nz; i++)
    {
        fread(&value, sizeof(double), 1, infile);
        coulomb_table[i] = float(value);
    }
    fprintf(stderr, "Finished transfering .binary data to memory!\n");
    fclose(infile);
    printf("Table Loaded!\n");

    return;
}

void CreateTexture(void)
{

    float *coulomb_table; //Device Array with coulomb floats
    // number of elements in each variable
    const int nx = SIZEOF_NE; //ne
    const int ny = SIZEOF_TI; //Ti
    const int nz = SIZEOF_TE; //Te
    coulomb_table = (float *)malloc(nx * ny * nz * sizeof(float));
    Load_coulomb_Tables(coulomb_table); //Loading coulomb Values into pointer
    //cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cuda Array
    cudaArray *cucoulArray;
    cudaMalloc3DArray(&cucoulArray, &channelDesc, make_cudaExtent(nz,ny,nx), 0);
    cudaMemcpy3DParms copyParams = {0};

    //Array creation
    copyParams.srcPtr   = make_cudaPitchedPtr((void *) coulomb_table, nx * sizeof(float), nx, ny);
    copyParams.dstArray = cucoulArray;
    copyParams.extent   = make_cudaExtent(nx, ny, nz);
    copyParams.kind     = cudaMemcpyHostToDevice;
    //checkCudaErrors(cudaMemcpy3D(&copyParams));
    cudaMemcpy3D(&copyParams);
    //Array creation End

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(texRes));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = cucoulArray;
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    cudaCreateTextureObject(&coulTexObj, &texRes, &texDescr, NULL);
    printf("Texture Created!\n");
    return;
}
__global__ void coulomb_function(cudaTextureObject_t my_tex, float a1, float a2, float a3, float* result)
{
    float v1, v2, v3, lambda;

    //Values for testing;
    v1 = a1; //ne parameter
    v2 = a2; //ti parameter
    v3 = a3; //te parameter
    printf("Values you chose:\n");
    printf("ne = %f, Ti = %f, Te = %f\n", v1, v2, v3);

    // For the normalized version only.
    const int nx = SIZEOF_TE; //Number of te used to generate table
    const int ny = SIZEOF_TI; //Number of ne used to generate table
    const int nz = SIZEOF_NE; //Number of Bmag used to generate table
    // if (a2 - a1 < 1e-7){
    //     a2 = 2;
    //     a1 = 2;
    // }
    v1 = ((((a1 - 2.) > 0? a1 - 2. : 0) * (nz - 1.)/23.) + 0.5);
    v2 = ((((a2 - 2.) > 0? a2 - 2. : 0) * (ny - 1.)/13.) + 0.5);
    v3 = ((((a3 - 2.) > 0? a3 - 2. : 0) * (nx - 1.)/13.) + 0.5);

    printf("Coordinates in texture grid:\n");
    printf("ne= %f, ti = %f, te = = %f\n", v1, v2, v3);

    // //For the normalized version only.
    lambda = tex3D<float>(my_tex, v3, v2, v1); 
    #if(!SINGLE_VALUE)
    *result = lambda;
    #endif
    printf("Coulomb value = %le\n", lambda);
    return;
}

__global__ void coulomb_function_test(cudaTextureObject_t my_tex, double * a0, double * a1, double * a2, double * value)
{
    float v1, v2, v3;
    float lambda;
    int i;
    // For the normalized version only.
    const int nx = SIZEOF_TE; //Number of te used to generate table
    const int ny = SIZEOF_TI; //Number of ne used to generate table
    const int nz = SIZEOF_NE; //Number of Bmag used to generate table
    for (i = 0; i < N_RESOLUTION; i++) {
        v1 = a0[i]; //ne parameter
        v2 = a1[i]; //ti parameter
        v3 = a2[i]; //te parameter

        //Melhor sem o roundoff
        v1 = ((((v1 - 2.) > 0? v1 - 2. : 0) * (nz - 1.)/23.) + 0.5);
        v2 = ((((v2 - 2.) > 0? v2 - 2. : 0) * (ny - 1.)/13.) + 0.5);
        v3 = ((((v3 - 2.) > 0? v3 - 2. : 0) * (nx - 1.)/13.) + 0.5);

        // //For the normalized version only.
        lambda = tex3D<float>(my_tex, v3, v2, v1); 
        value[i] = lambda;
    }
    return;
}


int main()
{
    #if(SINGLE_VALUE)
    float read1, read2, read3;
    float loop = 100;
    char str[1];
    float * value;
    value = (float*)malloc(sizeof(float));
    float *deviceValue;
    // Allocate device memory for value
    cudaMalloc((void**)&deviceValue, sizeof(float));
    CreateTexture();
    while (loop > 1)
    {
	    printf("Ne value:\n");
	    scanf("%f", &read1);
	    printf("Ti value:\n");
	    scanf("%f", &read2);
	    printf("Te value:\n");
	    scanf("%f", &read3);
        cudaMemcpy(deviceValue, value, sizeof(float), cudaMemcpyHostToDevice);
	    coulomb_function<<<1, 1>>>(coulTexObj, read1, read2, read3, deviceValue);
        cudaMemcpy(value, deviceValue, sizeof(float), cudaMemcpyDeviceToHost);
	    printf("Do you want to read other values? y/n\n");
	    scanf("%s", str);
	    if (strcmp(str, "n") == 0)
	    {
	    	loop = 0;
	    }
	}
        #elif(RESOLUTION_TEST)
        double *ne_test, *Ti_test, *Te_test, *coul_test;
        ne_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        Te_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        Ti_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        coul_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        int i;

        //Allocating memory in device memory.
        double * d_ne_test;
        cudaMalloc(&d_ne_test, N_RESOLUTION * sizeof(double));
        double * d_Te_test;
        cudaMalloc(&d_Te_test, N_RESOLUTION * sizeof(double));
        double * d_Ti_test;
        cudaMalloc(&d_Ti_test, N_RESOLUTION * sizeof(double));
        double * d_coul_test;
        cudaMalloc(&d_coul_test, N_RESOLUTION * sizeof(double));
        


        printf("Initializing resolution test reading\n");
        FILE *file_result;
        file_result = fopen("coulomb_table_test_cuda.txt", "w");
        FILE *file_e_density_test;
        file_e_density_test = fopen("electronic_density_sim.txt", "r");
        FILE *file_etemperature_test;
        file_etemperature_test = fopen("electronic_temperature_sim.txt", "r");
        FILE *file_itemperature_test;
        file_itemperature_test = fopen("ion_temperature_sim.txt", "r");
        if (file_e_density_test == NULL || file_etemperature_test == NULL || file_itemperature_test == NULL) {
            printf("Error Reading File\n");
        }
        CreateTexture();
        for (i = 0; fscanf(file_e_density_test, "%lf", ne_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_etemperature_test, "%lf", Te_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_itemperature_test, "%lf", Ti_test + i) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        cudaMemcpy(d_ne_test, ne_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ti_test, Ti_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Te_test, Te_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        printf("Reading and getting values from texture memory...\n");

        coulomb_function_test<<<1, 1>>>(coulTexObj, d_ne_test, d_Ti_test, d_Te_test, d_coul_test);
        cudaMemcpy(coul_test, d_coul_test, N_RESOLUTION * sizeof(double), cudaMemcpyDeviceToHost);

        for (i = 0; i < N_RESOLUTION; i++) {
            fprintf(file_result, "%.8e\n", coul_test[i]);
        } 

        free(ne_test);
        free(Ti_test);
        free(Te_test);
        free(coul_test);

        cudaFree(d_ne_test);
        cudaFree(d_Te_test);
        cudaFree(d_Ti_test);
        cudaFree(d_coul_test);

        fclose(file_e_density_test);
        fclose(file_etemperature_test);
        fclose(file_itemperature_test);
        printf("Test Sucessfull\n");
    #endif
    cudaDestroyTextureObject(coulTexObj);
    return 0;
}
