#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZEOF_H 101
#define SIZEOF_B 101
#define SIZEOF_TE 101
#define SIZEOF_NE 101
#define N_RESOLUTION 72192
#define DT 7.336005915070878e-07
#define THOMSON_CGS (6.652e-25)
#define BOLTZ_CGS (1.3806504e-16)
#define TABLE_SIZE (SIZEOF_H * SIZEOF_B * SIZEOF_TE * SIZEOF_NE)
#define SIZEOF_TEST 130
#include <time.h>

#define SINGLE_TEST (0)
#define RESOLUTION_TEST (1)
#define COMPARISON_MARCEL (0)
#define GLOBAL_MEMORY_TEST (0)
#define INDEX(i, j, k, l) (l + SIZEOF_TE * (k + SIZEOF_NE * (j + SIZEOF_B * i)))

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
    int nz = SIZEOF_B;  // Number of Bmag data.

    FILE *infile;
    double value;

    // Reading the cooling table
    infile = fopen("cooling_table_100_001.bin", "rb");

    if (infile == NULL)
    {
        fprintf(stderr, "Unable to open cooling file.\n");
        exit(1);
    }
    fprintf(stderr, "Reading Data...\n");
    // fscanf(infile, "%*[^\n]\n"); // this command is to ignore the first line.
    //  while (fscanf(infile, "%lf, %lf, %lf, %lf, %lf", &scale_height, &bmag, &ne, &te, &cool) == 5)
    //  {
    //      cool_arr[i] = cool;

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

void CreateTexture(void)
{
    float *cooling_table; // Device Array with cooling floats
    // number of elements in each variable
    const int nw = SIZEOF_H;  // H
    const int nx = SIZEOF_TE; // te
    const int ny = SIZEOF_NE; // ne
    const int nz = SIZEOF_B;  // bmag
    cooling_table = (float *)malloc(nw * nx * ny * nz * sizeof(float));
    Load_Cooling_Tables(cooling_table); // Loading Cooling Values into pointer
    // cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    // cuda Array
    cudaArray *cuCoolArray;
    // checkCudaErrors(cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx*sizeof(float),ny,nz), 0));
    cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx, ny * nz, nw), 0);
    cudaMemcpy3DParms copyParams = {0};

    // Array creation
    copyParams.srcPtr = make_cudaPitchedPtr((void *)cooling_table, nx * sizeof(float), nx, ny * nz);
    copyParams.dstArray = cuCoolArray;
    copyParams.extent = make_cudaExtent(nx, ny * nz, nw);
    copyParams.kind = cudaMemcpyHostToDevice;
    // checkCudaErrors(cudaMemcpy3D(&copyParams));
    cudaMemcpy3D(&copyParams);
    // Array creation End

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(texRes));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuCoolArray;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp; // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    // checkCudaErrors(cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL));}
    cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL);
    printf("Texture Created!\n");
    return;
}

__device__ void linspace(float start, float end, int numPoints, float *result)
{
    float stepSize = (end - start) / (float)(numPoints - 1);
    for (int i = 0; i < numPoints; i++)
    {
        result[i] = start + i * stepSize;
    }
}

__global__ void cooling_function(cudaTextureObject_t my_tex, float a0, float a1, float a2, float a3)
{
    float v0, v1, v4;
    double lambda;

    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table

    // Because we are going to interpolate manually, we need to define the lists that will hold the value for the parameters (same as table)
    float listofa1[SIZEOF_B];
    float listofa2[SIZEOF_NE];
    float a1_index, a2_index;

    // Generate the values used in the table by both parameters B and ne
    linspace(0, 10, SIZEOF_B, listofa1);
    linspace(2, 25, SIZEOF_NE, listofa2);

    // Calculate both dimensions that are not flattened
    v0 = ((((a0 - 3.) > 0 ? a0 - 3. : 0) * (nw - 1.) / 5.) + 0.5);

    if (a3 < 7.7730466 && a3 > 7.72){
        printf("Primeiro caso \n");
        v4 = (floor(((a3 - 2.) > 0 ? a3 - 2. : 0) * (nx - 1.) / 13.) + 0.5);
    }else if (a3 >7.7730466 && a3 < 7.85){
        printf("Segundo caso \n");
        v4 =  (floor(((a3 - 2.) > 0 ? a3 - 2. : 0) * (nx - 1.) / 13.) + 1 + 0.5);
    }else{
        printf("Terceiro caso \n");
        v4 = ((((a3 - 2.) > 0 ? a3 - 2. : 0) * (nx - 1.) / 13.) + 0.5);
    }
    printf("v4 = %lf\n", v4);
    // Select maximum values separetly
    if (a1 > 10)
    {
        a1 = 10;
    }
    else if (a2 > 25)
    {
        a2 = 25;
    }

    // These will give us the indexing of B and ne from the table, we gotta see if they are integers or not.
    a1_index = (((a1 - 0.) > 0 ? a1 : 0) * (nz - 1.) / 10.);
    a2_index = (((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.);

    if (a1_index == (int)a1_index && a2_index == (int)a2_index)
    {
        printf("Entrance 1 \n");
        v1 = ((((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + ((a1 - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
        printf("v0 = %lf, v1 = %lf, v4 = %lf\n", v0, v1, v4);
        lambda = tex3D<float>(my_tex, v4, v1, v0);
    }
    else if (a1_index != (int)a1_index && a2_index != (int)a2_index)
    {
        printf("Entrance 2 \n");
        float alpha, beta, v1_ij, v1_i1j, v1_ij1, v1_i1j1;
        alpha = a1_index - floor(a1_index);
        beta = a2_index - floor(a2_index);

        v1_ij = (floor(((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + floor((a1 - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
        v1_i1j = ((floor(((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + 1) + floor((a1 - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
        v1_ij1 = ((floor(((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.)) + (floor((a1 - 0.) * (nz - 1.) / 10.) + 1) * (ny) + 0.5);
        v1_i1j1 = ((floor(((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + 1) + (floor((a1 - 0.) * (nz - 1.) / 10.) + 1) * (ny) + 0.5);
        printf("B_before = %lf, B_after = %lf, ne_before = %lf, ne_after = %lf\n", floor(a1_index), floor(a1_index) + 1, floor(a2_index), floor(a2_index) + 1);
        printf("alpha = %lf, beta = %lf, texij = %lf, texi1j = %lf, texij1 = %lf, texi1j1 = %lf \n", alpha, beta, tex3D<float>(my_tex, v4, v1_ij, v0),
               tex3D<float>(my_tex, v4, v1_i1j, v0), tex3D<float>(my_tex, v4, v1_ij1, v0), tex3D<float>(my_tex, v4, v1_i1j1, v0));

        lambda = (1 - alpha) * (1 - beta) * tex3D<float>(my_tex, v4, v1_ij, v0) + alpha * (1 - beta) * tex3D<float>(my_tex, v4, v1_i1j, v0) +
                 (1 - alpha) * beta * tex3D<float>(my_tex, v4, v1_ij1, v0) + alpha * beta * tex3D<float>(my_tex, v4, v1_i1j1, v0);
    }
    else if (a1_index != (int)a1_index)
    {
        printf("Entrance 3 \n");
        float alpha, v1_i, v1_i1;
        printf("a1_index = %lf\n", a1_index);
        alpha = a1_index - floor(a1_index);
        v1_i = ((((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + floor((a1 - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
        v1_i1 = ((((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + (floor((a1 - 0.) * (nz - 1.) / 10.) + 1) * (ny) + 0.5);
        printf("v1_i = %lf, v1_i1 = %lf \n", v1_i, v1_i1);
        printf("alpha = %lf, tex before = %lf, tex after = %lf \n", alpha, tex3D<float>(my_tex, v4, v1_i, v0), tex3D<float>(my_tex, v4, v1_i1, v0));
        lambda = (1 - alpha) * tex3D<float>(my_tex, v4, v1_i, v0) + alpha * tex3D<float>(my_tex, v4, v1_i1, v0);
    }
    else
    {
        printf("Entrance 4 \n");
        float alpha, v1_i, v1_i1;
        alpha = a2_index - floor(a2_index);
        v1_i = (floor(((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + ((a1 - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
        v1_i1 = ((floor(((a2 - 2.) > 0 ? a2 - 2. : 0) * (ny - 1.) / 23.) + 1) + ((a1 - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
        printf("v1_i = %lf, v1_i1 = %lf \n", v1_i, v1_i1);
        printf("alpha = %lf, tex before = %lf, tex after = %lf \n", alpha, tex3D<float>(my_tex, v4, v1_i, v0), tex3D<float>(my_tex, v4, v1_i1, v0));
        lambda = (1 - alpha) * tex3D<float>(my_tex, v4, v1_i, v0) + alpha * tex3D<float>(my_tex, v4, v1_i1, v0);
    }

    printf("Coordinates in texture grid:\n");
    printf("Cooling value = %lf\n", lambda);
    return;
}

__global__ void cooling_function_marcel(cudaTextureObject_t my_tex, float a0, double *a1, double *a2, double *value)
{
    float v0, v1, v4;
    double ne_test, B_test, mu = 0.1;
    // For the normalized version only.
    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table
    //  v0 = (round((v0 - 3) * (nz - 1)/5) + 0.5)/nw; //scale_height
    //  v1 = (round((v1 - 0) * (nz - 1)/10) + 0.5)/nz; // Bmag
    //  v4 = ((round((v3 -2) * (nx - 1)/13) + 0.5) + round((v2 - 10) * (ny - 1)/15) * nx)/(nx * ny); //Te + ne
    for (int i = 0; i < 20; i++)
    {
        ne_test = a1[i] / (a0 * THOMSON_CGS);
        for (int k = 0; k < 20; k++)
        {
            B_test = sqrt(2 * mu * BOLTZ_CGS * ne_test * a2[k]);
            // v0 = (round((log10(a0) - 3.) * (nw - 1.)/5.) + 0.5)/nw;
            // v1 = (round((log10(ne_test) - 10.) * (ny - 1.)/15.) + 0.5 + round((log10(B_test) - 0.) * (nz - 1.)/10.) * ny)/(nz * ny);
            // v4 = (round((log10(a2[k]) - 2.) * (nx - 1.)/13.) + 0.5)/nx;
            v0 = (round((log10(a0) - 5.) * (nw - 1.) / 3.) + 0.5) / nw;
            v1 = (round((log10(ne_test) - 8.) * (ny - 1.) / 15.) + 0.5 + round((log10(B_test) - 0.) * (nz - 1.) / 10.) * ny) / (nz * ny);
            v4 = (round((log10(a2[k]) - 2.) * (nx - 1.) / 13.) + 0.5) / nx;
            value[20 * i + k] = tex3D<float>(my_tex, v4, v1, v0);
        }
    }
    return;
}

__global__ void cooling_function_test(cudaTextureObject_t my_tex, double *a0, double *a1, double *a2, double *a3, double *ucov, double *ug, double *value)
{
    double v0, v1, v2, v3, v4;
    double lambda;
    int i;

    // For the normalized version only.
    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table
    for (i = 0; i < N_RESOLUTION; i++)
    {


        // Because we are going to interpolate manually, we need to define the lists that will hold the value for the parameters (same as table)
        float listofa1[SIZEOF_B];
        float listofa2[SIZEOF_NE];
        float a1_index, a2_index;

        // Generate the values used in the table by both parameters B and ne
        linspace(0, 10, SIZEOF_B, listofa1);
        linspace(2, 25, SIZEOF_NE, listofa2);

        // Calculate both dimensions that are not flattened
        v0 = ((((a0[i] - 3.) > 0 ? a0[i] - 3. : 0) * (nw - 1.) / 5.) + 0.5);

        if (a3[i] < 7.7730466 && a3[i] > 7.72){
            v4 = (floor(((a3[i] - 2.) > 0 ? a3[i] - 2. : 0) * (nx - 1.) / 13.) + 0.5);
        }else if (a3[i] >7.7730466 && a3[i] < 7.85){
            v4 =  (floor(((a3[i] - 2.) > 0 ? a3[i] - 2. : 0) * (nx - 1.) / 13.) + 1 + 0.5);
        }else{
            v4 = ((((a3[i] - 2.) > 0 ? a3[i] - 2. : 0) * (nx - 1.) / 13.) + 0.5);
        }
        //v4 = ((((a3[i] - 2.) > 0 ? a3[i] - 2. : 0) * (nx - 1.) / 13.) + 0.5);
        // Select maximum values separetly
        if (a1[i] > 10)
        {
            a1[i] = 10;
        }
        else if (a2[i] > 25)
        {
            a2[i] = 25;
        }

        //v1 = (floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + floor((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);

        // These will give us the indexing of B and ne from the table, we gotta see if they are integers or not.
        a1_index = (((a1[i] - 0.) > 0 ? a1[i] : 0) * (nz - 1.) / 10.);
        a2_index = (((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.);

        if (a1_index == (int)a1_index && a2_index == (int)a2_index)
        {
            v1 = ((((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + ((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
            lambda = tex3D<float>(my_tex, v4, v1, v0);
        }
        else if (a1_index != (int)a1_index && a2_index != (int)a2_index)
        {
            float alpha, beta, v1_ij, v1_i1j, v1_ij1, v1_i1j1;
            beta = a1_index - floor(a1_index);
            alpha = a2_index - floor(a2_index);

            v1_ij = (floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + floor((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
            v1_i1j = ((floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + 1) + floor((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
            v1_ij1 = ((floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.)) + (floor((a1[i] - 0.) * (nz - 1.) / 10.) + 1) * (ny) + 0.5);
            v1_i1j1 = ((floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + 1) + (floor((a1[i] - 0.) * (nz - 1.) / 10.) + 1) * (ny) + 0.5);

            lambda = (1 - alpha) * (1 - beta) * tex3D<float>(my_tex, v4, v1_ij, v0) + alpha * (1 - beta) * tex3D<float>(my_tex, v4, v1_i1j, v0) +
                     (1 - alpha) * beta * tex3D<float>(my_tex, v4, v1_ij1, v0) + alpha * beta * tex3D<float>(my_tex, v4, v1_i1j1, v0);
        }
        else if (a1_index != (int)a1_index)
        {
            float alpha, v1_i, v1_i1;
            alpha = a1_index - floor(a1_index);
            v1_i = ((((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + floor((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
            v1_i1 = ((((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + (floor((a1[i] - 0.) * (nz - 1.) / 10.) + 1) * (ny) + 0.5);
            lambda = (1 - alpha) * tex3D<float>(my_tex, v4, v1_i, v0) + alpha * tex3D<float>(my_tex, v4, v1_i1, v0);
        }
        else
        {
            float alpha, v1_i, v1_i1;
            alpha = a2_index - floor(a2_index);
            v1_i = (floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + ((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
            v1_i1 = ((floor(((a2[i] - 2.) > 0 ? a2[i] - 2. : 0) * (ny - 1.) / 23.) + 1) + ((a1[i] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
            lambda = (1 - alpha) * tex3D<float>(my_tex, v4, v1_i, v0) + alpha * tex3D<float>(my_tex, v4, v1_i1, v0);
        }
        //lambda =tex3D<float>(my_tex,v4,v1,v0);
        value[i] = lambda;
    }
    return;
}

__device__ int binarySearchClosest(double *tablevalue, int size, double target)
{
    int left = 0;
    int right = size - 1;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;

        // Check if the target is present at the middle
        if (tablevalue[mid] == target)
            return mid;

        // If target greater, ignore left half
        if (tablevalue[mid] < target)
            left = mid + 1;

        // If target is smaller, ignore right half
        else
            right = mid - 1;
    }

    // Find the index of the closest element
    if (abs(tablevalue[left] - target) < abs(tablevalue[right] - target))
        return left;
    else
        return right;
}

__global__ void global_memory_reading(double *parameterH, double *parameterB, double *parameterNe, double *parameterTe, double *H_list, double *B_list, double *ne_list, double *Te_list, double *cooling, double *value)
{
    for (int i = 0; i < SIZEOF_TEST; i++)
    {
        int indexH = binarySearchClosest(H_list, SIZEOF_H, parameterH[i]);
        for (int j = 0; j < SIZEOF_TEST; j++)
        {
            int indexB = binarySearchClosest(B_list, SIZEOF_B, parameterB[j]);
            for (int k = 0; k < SIZEOF_TEST; k++)
            {
                int indexNe = binarySearchClosest(ne_list, SIZEOF_NE, parameterNe[k]);
                for (int l = 0; l < SIZEOF_TEST; l++)
                {
                    int indexTe = binarySearchClosest(Te_list, SIZEOF_TE, parameterTe[l]);
                    value[INDEX(indexH, indexB, indexNe, indexTe)] = cooling[INDEX(indexH, indexB, indexNe, indexTe)];
                }
            }
        }
    }
    // printf("Hello!");
}

__global__ void test()
{
    printf("Hello!\n");
    return;
}

void logspace(double start, double end, int num, double *result)
{
    double log_start = log10(start);                 // Initial value
    double log_end = log10(end);                     // End value
    double step = (log_end - log_start) / (num - 1); // number of steps
    int i;
    for (i = 0; i < num; ++i)
    {
        result[i] = log_start + i * step;
    }
}

__global__ void cooling_function_comparison_global(cudaTextureObject_t my_tex, double *a0, double *a1, double *a2, double *a3, double *value)
{
    double v0, v1, v2, v3, v4;
    double lambda;
    // For the normalized version only.
    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table

    for (int i = 0; i < SIZEOF_TEST; i++)
    {
        for (int j = 0; j < SIZEOF_TEST; j++)
        {
            for (int k = 0; k < SIZEOF_TEST; k++)
            {
                for (int l = 0; l < SIZEOF_TEST; l++)
                {
                    v0 = (floor(((a0[i] - 5.) > 0 ? a0[i] - 5. : 0) * (nw - 1.) / 3.) + 0.5);
                    v1 = (floor(((a2[k] - 2.) > 0 ? a2[k] - 2. : 0) * (ny - 1.) / 23.) + floor((a1[j] - 0.) * (nz - 1.) / 10.) * (ny) + 0.5);
                    v4 = (floor(((a3[l] - 2.) > 0 ? a3[l] - 2. : 0) * (nx - 1.) / 13.) + 0.5);
                    lambda = tex3D<float>(my_tex, v4, v1, v0);
                    value[INDEX(i, j, k, l)] = lambda;
                }
            }
        }
    }
    return;
}
int main()
{
#if (SINGLE_TEST)
    float read0, read1, read2, read3;
    float loop = 100;
    float *value;

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
        cudaDeviceSynchronize();
        printf("Do you want to read other values? y/n\n");
        scanf("%s", str);
        if (strcmp(str, "n") == 0)
        {
            loop = 0;
        }
    }
    cudaDestroyTextureObject(coolTexObj);
#elif (RESOLUTION_TEST)
    double *H_test, *B_test, *ne_test, *Te_test, *cool_test, *ug_test, *ucov_test;
    H_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    B_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    ne_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    Te_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    cool_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    ug_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    ucov_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    int i;

    // Allocating memory in device memory.
    double *d_H_test;
    cudaMalloc(&d_H_test, N_RESOLUTION * sizeof(double));
    double *d_B_test;
    cudaMalloc(&d_B_test, N_RESOLUTION * sizeof(double));
    double *d_ne_test;
    cudaMalloc(&d_ne_test, N_RESOLUTION * sizeof(double));
    double *d_Te_test;
    cudaMalloc(&d_Te_test, N_RESOLUTION * sizeof(double));
    double *d_cool_test;
    cudaMalloc(&d_cool_test, N_RESOLUTION * sizeof(double));
    double *d_ug_test;
    cudaMalloc(&d_ug_test, N_RESOLUTION * sizeof(double));
    double *d_ucov_test;
    cudaMalloc(&d_ucov_test, N_RESOLUTION * sizeof(double));

    printf("Initializing resolution test reading\n");
    FILE *file_result;
    file_result = fopen("cuda_test_nofloor_temp.txt", "w");
    FILE *file_height_test;
    file_height_test = fopen("scaleheight_sim.txt", "r");
    FILE *file_e_density_test;
    file_e_density_test = fopen("electronic_density_sim.txt", "r");
    FILE *file_temperature_test;
    file_temperature_test = fopen("electronic_temperature_sim.txt", "r");
    FILE *file_mag_field_test;
    file_mag_field_test = fopen("magnetic_field_sim.txt", "r");
    // FILE *file_ucov_test;
    // file_ucov_test = fopen("ucov_sim.txt", "r");
    // FILE *file_ug_test;
    // file_ug_test = fopen("ug_sim.txt", "r");
    CreateTexture();
    for (i = 0; fscanf(file_height_test, "%lf", H_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_mag_field_test, "%lf", B_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_e_density_test, "%lf", ne_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_temperature_test, "%lf", Te_test + i) == 1; i++)
    {
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
    // cudaMemcpy(d_ucov_test, ucov_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ug_test, ug_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
    printf("Reading and getting values from texture memory...\n");

    cooling_function_test<<<1, 1>>>(coolTexObj, d_H_test, d_B_test, d_ne_test, d_Te_test, d_ucov_test, d_ug_test, d_cool_test);
    cudaMemcpy(cool_test, d_cool_test, N_RESOLUTION * sizeof(double), cudaMemcpyDeviceToHost);

    for (i = 0; i < N_RESOLUTION; i++)
    {
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
#elif (COMPARISON_MARCEL)
    CreateTexture();
    double *te_test, H = 0.1 * 1.483366675977058e6 * 30, *tau_test, *result;
    double *tau_test_d, *te_test_d, *result_d;
    cudaMalloc(&tau_test_d, 20 * sizeof(double));
    cudaMalloc(&te_test_d, 20 * sizeof(double));
    cudaMalloc(&result_d, 400 * sizeof(double));

    tau_test = (double *)malloc(20 * sizeof(double)); // Allocate memory for tau_test on the host
    te_test = (double *)malloc(20 * sizeof(double));
    result = (double *)malloc(400 * sizeof(double));

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

    for (int i = 0; i < 400; i++)
    {
        fprintf(file_result, "%.8e,", pow(10., result[i]));
    }
    fclose(file_result);
    // free(te_test);
    // free(tau_test);
#elif (GLOBAL_MEMORY_TEST)
    double *H_test, *B_test, *ne_test, *Te_test, *cool_test;
    double *H_random, *B_random, *ne_random, *Te_random;
    double *results = 0;
    clock_t start_time, end_time;
    double duration;
    cudaError_t error;

    H_test = (double *)malloc(SIZEOF_H * sizeof(double));
    B_test = (double *)malloc(SIZEOF_B * sizeof(double));
    ne_test = (double *)malloc(SIZEOF_NE * sizeof(double));
    Te_test = (double *)malloc(SIZEOF_TE * sizeof(double));
    cool_test = (double *)malloc(TABLE_SIZE * sizeof(double));

    H_random = (double *)malloc(SIZEOF_TEST * sizeof(double));
    B_random = (double *)malloc(SIZEOF_TEST * sizeof(double));
    ne_random = (double *)malloc(SIZEOF_TEST * sizeof(double));
    Te_random = (double *)malloc(SIZEOF_TEST * sizeof(double));

    int i;

    // Allocating memory in device memory.
    double *d_H_test;
    cudaMalloc(&d_H_test, SIZEOF_H * sizeof(double));
    double *d_B_test;
    cudaMalloc(&d_B_test, SIZEOF_B * sizeof(double));
    double *d_ne_test;
    cudaMalloc(&d_ne_test, SIZEOF_NE * sizeof(double));
    double *d_Te_test;
    cudaMalloc(&d_Te_test, SIZEOF_TE * sizeof(double));
    double *d_cool_test;
    cudaMalloc(&d_cool_test, TABLE_SIZE * sizeof(double));

    double *d_H_random;
    cudaMalloc(&d_H_random, SIZEOF_TEST * sizeof(double));
    double *d_B_random;
    cudaMalloc(&d_B_random, SIZEOF_TEST * sizeof(double));
    double *d_Ne_random;
    cudaMalloc(&d_Ne_random, SIZEOF_TEST * sizeof(double));
    double *d_Te_random;
    cudaMalloc(&d_Te_random, SIZEOF_TEST * sizeof(double));
    double *d_results;
    cudaMalloc(&d_results, SIZEOF_TEST * sizeof(double));
    printf("Initializing GLOBAL MEMORY testing\n");
    FILE *file_height_test;
    file_height_test = fopen("scale_height.txt", "r");
    FILE *file_e_density_test;
    file_e_density_test = fopen("ne.txt", "r");
    FILE *file_temperature_test;
    file_temperature_test = fopen("te.txt", "r");
    FILE *file_mag_field_test;
    file_mag_field_test = fopen("mag.txt", "r");
    FILE *file_cooling_test;
    file_cooling_test = fopen("cooling_table.bin", "rb");
    for (i = 0; fscanf(file_height_test, "%lf", H_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_mag_field_test, "%lf", B_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_e_density_test, "%lf", ne_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_temperature_test, "%lf", Te_test + i) == 1; i++)
    {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; i < TABLE_SIZE; i++)
    {
        // printf("i = %d \n", i);
        fread(&cool_test[i], sizeof(double), 1, file_cooling_test);
        // printf("cool_test[%d] = %lf\n", i, cool_test[i]);
    }
    logspace(1e5, 1e8, SIZEOF_TEST, H_random);
    logspace(1e0, 1e10, SIZEOF_TEST, B_random);
    logspace(1e2, 1e15, SIZEOF_TEST, Te_random);
    logspace(1e2, 1e25, SIZEOF_TEST, ne_random);
    printf("Transfering data from Host to Device... \n");
    cudaMemcpy(d_H_test, H_test, SIZEOF_H * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_test, B_test, SIZEOF_B * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ne_test, ne_test, SIZEOF_NE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Te_test, Te_test, SIZEOF_TE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cool_test, cool_test, TABLE_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H_random, H_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_random, B_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ne_random, ne_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Te_random, Te_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    CreateTexture();

    printf("Starting table lookup...\n");
    start_time = clock();
    global_memory_reading<<<1, 1>>>(d_H_random, d_B_random, d_Ne_random, d_Te_random, d_H_test, d_B_test, d_ne_test, d_Te_test, d_cool_test, d_results);
    cudaDeviceSynchronize();
    end_time = clock();
    duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Number of values analyzed = %d, duration global: %.6f seconds\n", SIZEOF_TEST, duration);

    start_time = clock();
    cooling_function_comparison_global<<<1, 1>>>(coolTexObj, d_H_random, d_B_random, d_Ne_random, d_Te_random, d_results);
    cudaDeviceSynchronize();
    end_time = clock();
    duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Number of values analyzed = %d, duration texture: %.6f seconds\n", SIZEOF_TEST, duration);

#endif
    return 0;
}
