#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZEOF_H 100
#define SIZEOF_B 100
#define SIZEOF_TE 100
#define SIZEOF_NE 100
#define N_RESOLUTION 72192
#define SINGLE_TEST (0)
#define RESOLUTION_TEST (1)
#define COMPARISON_MARCEL (0)
#define DT 7.336005915070878e-07
#define THOMSON_CGS (6.652e-25) 
#define BOLTZ_CGS (1.3806504e-16)
cudaTextureObject_t coolTexObj;
cudaArray *cuCoolArray = 0;

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

// Load the cooling_table into the CPU Memory.
void Load_Cooling_Tables(float *cooling_table)
{
    fprintf(stderr, "Loading Table...\n");

    int i = 0;
    int nw = SIZEOF_H; 
    int nx = SIZEOF_TE; // Number of Te data.
    int ny = SIZEOF_NE; // Number of ne data.
    int nz = SIZEOF_B; // Number of Bmag data.

    FILE *infile;
    double value;


    // Reading the cooling table
    infile = fopen("cooling_table_100.bin", "rb");
    
    if (infile == NULL)
    {
        fprintf(stderr, "Unable to open cooling file.\n");
        exit(1);
    }
    fprintf(stderr, "Reading Data...\n");
    //fscanf(infile, "%*[^\n]\n"); // this command is to ignore the first line.
    // while (fscanf(infile, "%lf, %lf, %lf, %lf, %lf", &scale_height, &bmag, &ne, &te, &cool) == 5)
    // {
    //     cool_arr[i] = cool;

    //     i++;
    // },

    // copy data from cooling array into the table
    for (i = 0; i < nw * nx * ny * nz; i++)
    {
        fread(&value, sizeof(double), 1, infile);
        cooling_table[i] = float(value);
    }

    fprintf(stderr, "Finished transfering .binary data to memory!\n");
    fclose(infile);

    printf("Table Loaded!\n");

    return;
}

__device__ float roundTo6Decimals(float num) {
    float multiplier = powf(10, 6);
    printf("Number rounded = %f", roundf(num * multiplier) / multiplier);
    return roundf(num * multiplier) / multiplier;
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
    cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx,ny * nz, nw), 0);
    cudaMemcpy3DParms copyParams = {0};

    //Array creation
    copyParams.srcPtr   = make_cudaPitchedPtr((void *) cooling_table, nx * sizeof(float), nx, ny * nz);
    copyParams.dstArray = cuCoolArray;
    copyParams.extent   = make_cudaExtent(nx, ny * nz, nw);
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
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    //checkCudaErrors(cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL));}
    cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL);
    printf("Texture Created!\n");
    return;
}
__global__ void cooling_function(cudaTextureObject_t my_tex, float a0, float a1, float a2, float a3
    ,float* result
){
    float v0, v1, v4;
    double lambda;
    // For the normalized version only.
    const int nw = SIZEOF_H; //Number of H used to generate table
    const int nx = SIZEOF_TE; //Number of te used to generate table
    const int ny = SIZEOF_NE; //Number of ne used to generate table
    const int nz = SIZEOF_B; //Number of Bmag used to generate table
    //  v0 = (round((v0 - 3) * (nz - 1)/5) + 0.5)/nw; //scale_height
    //  v1 = (round((v1 - 0) * (nz - 1)/10) + 0.5)/nz; // Bmag
    //  v4 = ((round((v3 -2) * (nx - 1)/13) + 0.5) + round((v2 - 10) * (ny - 1)/15) * nx)/(nx * ny); //Te + ne

    // v0 = (round((a0 - 3.) * (nw - 1.)/5)) + 0.5;
    // v1 = (round((a2 - 2.) * (ny - 1.)/23.) + round((a1 - 0.) * (nz - 1.)/10.) * (ny)) + 0.5;
    // v4 = (round((a3 - 2.) * (nx - 1.)/13.)) + 0.5;
    v0 = (floor(((a0 - 5.) > 0? a0 - 5. : 0) * (nw - 1.)/3.) + 0.5);
    v1 = (floor(((a2 - 2.) > 0? a2 - 2. : 0) * (ny - 1.)/23.) + floor((a1 - 0.) * (nz - 1.)/10.) * (ny) + 0.5);
    v4 = (floor(((a3 - 2.) > 0? a3 - 2. : 0) * (nx - 1.)/13.) + 0.5);

    //For the non normalized version only.
    //lambda = tex3D<float>(coolTexObj, v3 + 0.5f, v2 + 0.5f, v1 + 0.5f); 

    // //For the normalized version only.
    lambda = tex3D<float>(my_tex, v4, v1, v0);
    #if(!SINGLE_TEST)
    *result = lambda;
    #endif
    printf("Coordinates in texture grid:\n");
    printf("Scale_height = %f, Bmag + ne = %f, te = = %f\n", v0, v1, v4);
    printf("Cooling value = %lf\n", lambda);
    return;
}

__global__ void cooling_function_marcel(cudaTextureObject_t my_tex, float a0, double * a1, double * a2
    , double * value
    )
{ 
    float v0, v1, v4;
    double ne_test, B_test, mu = 0.1 ;
    // For the normalized version only.
    const int nw = SIZEOF_H; //Number of H used to generate table
    const int nx = SIZEOF_TE; //Number of te used to generate table
    const int ny = SIZEOF_NE; //Number of ne used to generate table
    const int nz = SIZEOF_B; //Number of Bmag used to generate table
    //  v0 = (round((v0 - 3) * (nz - 1)/5) + 0.5)/nw; //scale_height
    //  v1 = (round((v1 - 0) * (nz - 1)/10) + 0.5)/nz; // Bmag
    //  v4 = ((round((v3 -2) * (nx - 1)/13) + 0.5) + round((v2 - 10) * (ny - 1)/15) * nx)/(nx * ny); //Te + ne
    for (int i = 0; i < 20; i++) {
        ne_test = a1[i]/(a0 * THOMSON_CGS);
        for(int k = 0; k < 20; k++){
                B_test = sqrt(2 * mu *BOLTZ_CGS* ne_test * a2[k]);
                // v0 = (round((log10(a0) - 3.) * (nw - 1.)/5.) + 0.5)/nw;
                // v1 = (round((log10(ne_test) - 10.) * (ny - 1.)/15.) + 0.5 + round((log10(B_test) - 0.) * (nz - 1.)/10.) * ny)/(nz * ny);
                // v4 = (round((log10(a2[k]) - 2.) * (nx - 1.)/13.) + 0.5)/nx;
                v0 = (round((log10(a0) - 5.) * (nw - 1.)/3.) + 0.5)/nw;
                v1 = (round((log10(ne_test) - 8.) * (ny - 1.)/15.) + 0.5 + round((log10(B_test) - 0.) * (nz - 1.)/10.) * ny)/(nz * ny);
                v4 = (round((log10(a2[k]) - 2.) * (nx - 1.)/13.) + 0.5)/nx;
                value[20*i + k] = tex3D<float>(my_tex, v4, v1, v0); 
        }
    }
    return;
}

__global__ void cooling_function_test(cudaTextureObject_t my_tex, double * a0, double * a1, double * a2, double * a3, double * ucov, double * ug, double * value)
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

        // v0 = (floor((v0 - 3.) * (nw - 1.)/5) + 0.5);
        // v1 = (floor((v2 - 2.) * (ny - 1.)/23.) + floor((v1 - 0.) * (nz - 1.)/10.) * (ny) + 0.5);
        // v4 = (floor((v3 - 2.) * (nx - 1.)/13.)+ 0.5);

        //Esse aqui deu certo para 32^4 finalmente pqp
        // v0 = (floor(((v0 - 3.) > 0? v0 - 3. : 0) * (nw - 1.)/5.) + 0.5);
        // v1 = (floor(((v2 - 10.) > 0? v2 - 10. : 0) * (ny - 1.)/15.) + floor((v1 - 0.) * (nz - 1.)/10.) * (ny) + 0.5);
        // v4 = (floor(((v3 - 2.) > 0? v3 - 2. : 0) * (nx - 1.)/13.) + 0.5);

        //Teste para o 100^4: FUNCIONOUUUUUUUU
        v0 = (floor(((v0 - 5.) > 0? v0 - 5. : 0) * (nw - 1.)/3.) + 0.5);
        v1 = (floor(((v2 - 2.) > 0? v2 - 2. : 0) * (ny - 1.)/23.) + floor((v1 - 0.) * (nz - 1.)/10.) * (ny) + 0.5);
        v4 = (floor(((v3 - 2.) > 0? v3 - 2. : 0) * (nx - 1.)/13.) + 0.5);

        // //For the normalized version only.
        lambda = tex3D<float>(my_tex, v4, v1, v0); 
        //printf("lambda = %le\n", lambda);
        //if (fabsf(ucov[i] * DT * lambda)> 0.3 * fabsf(ug[i])){
        //    lambda *= 0.3 * fabsf(ug[i])/(DT * fabsf(ucov[i] * lambda));
        //}
        value[i] = lambda;
    }
    return;
}
void logspace(double start, double end, int num, double* result) {
    double log_start = log10(start); //Initial value
    double log_end = log10(end); //End value
    double step = (log_end - log_start) / (num - 1); // number of steps
    int i;
    for (i = 0; i < num; ++i) {
        result[i] = pow(10.0, log_start + i * step);
    }
}


int main()
{
    #if(SINGLE_TEST)
        float read0, read1, read2, read3;
        float loop = 100;
        float * value;
        value = (float*)malloc(sizeof(float));
        float *deviceValue;
        // Allocate device memory for value
        cudaMalloc((void**)&deviceValue, sizeof(float));
        

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
            cudaMemcpy(deviceValue, value, sizeof(float), cudaMemcpyHostToDevice);
            cooling_function<<<1, 1>>>(coolTexObj, read0, read1, read2, read3, value);
            cudaMemcpy(value, deviceValue, sizeof(float), cudaMemcpyDeviceToHost);
            //cooling_function<<<1, 1>>>(coolTexObj, read0, read1, read2, read3);
            printf("Do you want to read other values? y/n\n");
            scanf("%s", str);
            if (strcmp(str, "n") == 0)
            {
                loop = 0;
            }
        }
        cudaDestroyTextureObject(coolTexObj);
    #elif(RESOLUTION_TEST)
        double *H_test, *B_test, *ne_test, *Te_test, *cool_test, *ug_test, *ucov_test;
        H_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        B_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        ne_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        Te_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        cool_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        ug_test = (double*)malloc(N_RESOLUTION * sizeof(double));
        ucov_test = (double*)malloc(N_RESOLUTION * sizeof(double));
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
        double * d_ug_test;
        cudaMalloc(&d_ug_test, N_RESOLUTION * sizeof(double));
        double * d_ucov_test;
        cudaMalloc(&d_ucov_test, N_RESOLUTION * sizeof(double));
        


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
        //FILE *file_ucov_test;
        //file_ucov_test = fopen("ucov_sim.txt", "r");
        //FILE *file_ug_test;
        //file_ug_test = fopen("ug_sim.txt", "r");
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
        // for (i = 0; fscanf(file_ug_test, "%lf", ucov_test + i) == 1; i++) {
        //     // Do nothing inside the loop body, everything is done in the for loop header
        // }
        // for (i = 0; fscanf(file_ucov_test, "%lf", ug_test + i) == 1; i++) {
        //     // Do nothing inside the loop body, everything is done in the for loop header
        // }
        cudaMemcpy(d_H_test, H_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_test, B_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ne_test, ne_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Te_test, Te_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_ucov_test, ucov_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_ug_test, ug_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
        printf("Reading and getting values from texture memory...\n");

        cooling_function_test<<<1, 1>>>(coolTexObj, d_H_test, d_B_test, d_ne_test, d_Te_test, d_ucov_test, d_ug_test, d_cool_test);
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
    #if (COMPARISON_MARCEL)
        CreateTexture();
        double *te_test, H = 0.1 * 1.483366675977058e6 * 30, *tau_test, *result;
        double *tau_test_d, *te_test_d, *result_d;
        cudaMalloc(&tau_test_d, 20 * sizeof(double));
        cudaMalloc(&te_test_d, 20 * sizeof(double));
        cudaMalloc(&result_d, 400 * sizeof(double));

        tau_test = (double*)malloc(20 * sizeof(double)); // Allocate memory for tau_test on the host
        te_test = (double*)malloc(20 * sizeof(double)); 
        result = (double*)malloc(400 * sizeof(double)); 



        double tau_start = 1.e-6, tau_end = 5.e2;
        double te_start = 5.e4, te_end = 2.e11;
        FILE *file_result;
        file_result = fopen("marcel_comp.txt", "w");
        logspace(tau_start, tau_end, 20, tau_test);
        logspace(te_start, te_end, 20, te_test);
        
        cudaMemcpy(tau_test_d, tau_test, 20 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(te_test_d, te_test, 20 * sizeof(double), cudaMemcpyHostToDevice);

        cooling_function_marcel<<<1, 1>>>(coolTexObj, H, tau_test_d, te_test_d, result_d);
        cudaMemcpy(result, result_d, 400 * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 400; i++) {
            fprintf(file_result, "%.8e,", pow(10.,result[i]));
        }
        fclose(file_result);
        //free(te_test);
        //free(tau_test);
    #endif

    return 0;
}
