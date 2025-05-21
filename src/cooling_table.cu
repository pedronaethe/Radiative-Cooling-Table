#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define NBLOCKS 1792
#define NTHREADS 256
#define SIZEOF_H (40 + 1) /*Size of H's in your cooling table*/
#define SIZEOF_B (40 + 1) /*Size of B's in your cooling table*/
#define SIZEOF_TE (40 + 1) /*Size of Te's in your cooling table*/
#define SIZEOF_NE (40 + 1)/*Size of Ne's in your cooling table*/
#define N_RESOLUTION 65536 /*This is for resolution_test and is the number of cells in your simulation*/
#define DT 7.336005915070878e-07 /*This is an approximation of the timestep for coulomb test*/

#define THOMSON_CGS (6.652e-25) /*Thomson's cross section in CGS*/
#define BOLTZ_CGS (1.3806504e-16) /*Boltzmann constant in CGS*/
#define TABLE_SIZE (SIZEOF_H * SIZEOF_B * SIZEOF_TE * SIZEOF_NE) /*Total size of the table*/
#define SIZEOF_TEST 50  /*Quad root of number of calculations for GLOBAL_MEMORY_TEST*/

#define SINGLE_TEST (0) /*Single value test*/
#define RESOLUTION_TEST (0) /*Compare analytical values with values from the table*/
#define COMPARISON_MARCEL (0) /*Compare plot A.1 of Marcel et al. 2018: A unified accretion-ejection paradigm for black hole X-ray binaries*/
#define GLOBAL_MEMORY_TEST (1) /*Test texture memory vs global memory efficiency*/
#define INDEX(h, b, ne, te) (((((h) * SIZEOF_B + (b)) * SIZEOF_NE + (ne)) * SIZEOF_TE) + (te))

/*Declaration of both texture objects*/
cudaTextureObject_t coolTexObj;
cudaArray *cuCoolArray = 0;

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

/*This function loads the cooling values from the binary file*/
void Load_Cooling_Tables(float *cooling_table)
{
    fprintf(stderr, "Loading Table...\n");

    int i = 0;
    int nw = SIZEOF_H; //Number of H data
    int nx = SIZEOF_TE; // Number of Te data.
    int ny = SIZEOF_NE; // Number of ne data.
    int nz = SIZEOF_B;  // Number of Bmag data.

    FILE *infile;
    double value;

    // Reading the cooling table
    infile = fopen("../tables/cooling_table_40.bin", "rb");

    if (infile == NULL)
    {
        fprintf(stderr, "Unable to open cooling file.\n");
        exit(1);
    }
    fprintf(stderr, "Reading Data...\n");

    // Opening the binary file and reading the data, while also transferring it to the pointer cooling_table
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

/*This function will transfer the values from the table to the texture object*/
void CreateTexture(void)
{
    float *cooling_table; //Pointer that will hold the cooling values from Load_cooling_table function
    const int nw = SIZEOF_H;  // Number of H data
    const int nx = SIZEOF_TE; // Number of Te data
    const int ny = SIZEOF_NE; // Number of ne data
    const int nz = SIZEOF_B;  // Number of Bmag data
    cooling_table = (float *)malloc(nw * nx * ny * nz * sizeof(float)); //Allocating memory for cooling_table pointer

    Load_Cooling_Tables(cooling_table); // Loading Cooling Values into pointer
    
    // cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    // cuda Array
    cudaArray *cuCoolArray;

    //Creating 3D array in device memory  
    cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx * ny, nz, nw), 0);
    cudaMemcpy3DParms copyParams = {0};

    // Copying cooling values from host memory to device array.
    copyParams.srcPtr = make_cudaPitchedPtr((void *)cooling_table, nx * ny* sizeof(float), nx * ny, nz);
    copyParams.dstArray = cuCoolArray;
    copyParams.extent = make_cudaExtent(nx * ny, nz, nw);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    // Array creation End

    //Defining parameters for the texture object
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(texRes));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuCoolArray;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = false; //Whether to use normalized coordinates or not, this will impact the indexing
    texDescr.filterMode = cudaFilterModeLinear;//cudaFilterModeLinear;//cudaFilterModePoint; // Whether to use nearest-neighbor approximation or trilinear interpolation
    texDescr.addressMode[0] = cudaAddressModeClamp; // Out of boundary conditions in dimension 1
    texDescr.addressMode[1] = cudaAddressModeClamp; // Out of boundary conditions in dimension 2
    texDescr.addressMode[2] = cudaAddressModeClamp; // Out of boundary conditions in dimension 3
    texDescr.readMode = cudaReadModeElementType; // Type of values stored in texture object

    cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL); //Creating the texture object with the channel and parameters described above
    printf("Texture Created!\n");
    return;
}

__device__ double no_interpolation(cudaTextureObject_t my_tex, double H_value, double B_value, double ne_value, double te_value){
    double coord_H, coord_B, Coord;
    float te_index, ne_index;


    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table

    // Calculate both dimensions that are not flattened
    coord_H = ((((H_value - 3.) > 0 ? H_value - 3. : 0) * (nw - 1.) / 9.) + 0.5);
    coord_B = (((B_value - 0.) > 0 ? B_value : 0) * (nz - 1.) / 10. + 0.5);

    // Select maximum values separetly
    if (ne_value > 25)
    {
        ne_value = 25;
    }
    else if (te_value > 15)
    {
        te_value = 15;
    }

    // These will give us the indexing of B and ne from the table, we gotta see if they are integers or not.
    te_index = (((te_value - 2.) > 0 ? te_value - 2. : 0) * (nx - 1.) / 13.);
    ne_index = (((ne_value - 2.) > 0 ? ne_value - 2. : 0) * (ny - 1.) / 23.); 

    Coord = ((te_index) + ne_index * (nx) + 0.5);
    return tex3D<float>(my_tex, Coord, coord_B, coord_H);
}

__device__ double interpolate_ne_te(cudaTextureObject_t my_tex, double H_value, double B_value, double ne_value, double te_value){
    double coord_H, coord_B, Coord;
    float te_index, ne_index;

    //double t_break = 9.472016; //para 101
    //double t_ubreak = 9.540000; //para 101
    //double t_lbreak = 9.410000; //para 101

    // double t_break = 9.472016; //para 33
    // double t_ubreak = 9.71875; //para 33
    // double t_lbreak = 9.3125; //para 33
    
    double t_break = 9.472016; //para 41
    double t_ubreak = 9.4750; //para 41
    double t_lbreak = 9.1500; //para 41
    
    float alpha, beta, Coord_ij, Coord_i1j, Coord_ij1, Coord_i1j1, Coord_i, Coord_i1;
    float Coord_ihalfj, Coord_ihalfj1, Coord_im1j1, Coord_im1j, Coord_iM1j1, Coord_iM1j, Coord_im2j1, Coord_im2j, Coord_iM2j1, Coord_iM2j, frac_break, alpha_lower, alpha_upper;

    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table

    // Calculate both dimensions that are not flattened
    coord_H = ((((H_value - 3.) > 0 ? H_value - 3. : 0) * (nw - 1.) / 9.) + 0.5);
    coord_B = (((B_value - 0.) > 0 ? B_value : 0) * (nz - 1.) / 10. + 0.5);

    // Select maximum values separetly
    if (ne_value > 25)
    {
        ne_value = 25;
    }
    else if (te_value > 15)
    {
        te_value = 15;
    }

    // These will give us the indexing of B and ne from the table, we gotta see if they are integers or not.
    te_index = (((te_value - 2.) > 0 ? te_value - 2. : 0) * (nx - 1.) / 13.);
    ne_index = (((ne_value - 2.) > 0 ? ne_value - 2. : 0) * (ny - 1.) / 23.);

    if (te_index == (int)te_index && ne_index == (int)ne_index)
    {
        Coord = ((te_index) + ne_index * (nx) + 0.5);
        return tex3D<float>(my_tex, Coord, coord_B, coord_H);
    }
    else if (te_index != (int)te_index && ne_index != (int)ne_index)
    {   
        beta = ne_index - floor(ne_index);
        alpha = te_index - floor(te_index);
        if (te_value < t_break && te_value > t_lbreak){
            frac_break = t_break - t_lbreak;
            alpha_lower = (te_value - t_lbreak)/frac_break;
            Coord_ij = (floor(te_index) + floor(ne_index) * (nx) + 0.5);
            Coord_ij1 = ((floor(te_index)) + (floor(ne_index) + 1) * (nx) + 0.5);


            Coord_im2j =((floor(te_index) - 2) + (floor(ne_index))  * (nx) + 0.5);
            Coord_im1j = ((floor(te_index) - 1) + (floor(ne_index)) * (nx) + 0.5);
            Coord_ihalfj = 3 * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) - 3 * tex3D<float>(my_tex, Coord_im1j, coord_B, coord_H) + tex3D<float>(my_tex, Coord_im2j, coord_B, coord_H);

            Coord_im2j1 =((floor(te_index) - 2) + (floor(ne_index) + 1)  * (nx) + 0.5);
            Coord_im1j1 = ((floor(te_index) - 1) + (floor(ne_index) + 1) * (nx) + 0.5);
            Coord_ihalfj1 = 3 * tex3D<float>(my_tex, Coord_ij1, coord_B, coord_H) - 3 * tex3D<float>(my_tex, Coord_im1j1, coord_B, coord_H) + tex3D<float>(my_tex, Coord_im2j1, coord_B, coord_H);

            return (1 - alpha_lower) * (1 - beta) * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) + alpha_lower * (1 - beta) * Coord_ihalfj +
                    (1 - alpha_lower) * beta * tex3D<float>(my_tex, Coord_ij1, coord_B, coord_H) + alpha_lower * beta * Coord_ihalfj1;                


        }else if(te_value >t_break && te_value < t_ubreak){//
            frac_break = t_ubreak - t_break;
            alpha_upper = (te_value - t_break)/(frac_break);
            Coord_ij = ((floor(te_index) + 1) + floor(ne_index) * (nx) + 0.5);
            Coord_ij1 = ((floor(te_index) + 1) + (floor(ne_index) + 1) * (nx) + 0.5);


            Coord_iM2j =((floor(te_index) + 3) + (floor(ne_index))  * (nx) + 0.5);
            Coord_iM1j = ((floor(te_index) + 2) + (floor(ne_index)) * (nx) + 0.5);
            Coord_ihalfj = 3 * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) - 3 * tex3D<float>(my_tex, Coord_iM1j, coord_B, coord_H) + tex3D<float>(my_tex, Coord_iM2j, coord_B, coord_H);

            Coord_iM2j1 =((floor(te_index) + 3) + (floor(ne_index) + 1)  * (nx) + 0.5);
            Coord_iM1j1 = ((floor(te_index) + 2) + (floor(ne_index) + 1) * (nx) + 0.5);
            Coord_ihalfj1 = 3 * tex3D<float>(my_tex, Coord_ij1, coord_B, coord_H) - 3 * tex3D<float>(my_tex, Coord_iM1j1, coord_B, coord_H) + tex3D<float>(my_tex, Coord_iM2j1, coord_B, coord_H);
            return (1 - alpha_upper) * (1 - beta) * Coord_ihalfj + alpha_upper * (1 - beta) * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) +
                    (1 - alpha_upper) * beta * Coord_ihalfj1 + alpha_upper * beta * tex3D<float>(my_tex, Coord_ij1, coord_B, coord_H);  
        }else{//
            Coord_ij = (floor(te_index) + floor(ne_index) * (nx) + 0.5);
            Coord_i1j = ((floor(te_index) + 1) + floor(ne_index) * (nx) + 0.5);
            Coord_ij1 = ((floor(te_index)) + (floor(ne_index) + 1) * (nx) + 0.5);
            Coord_i1j1 = ((floor(te_index) + 1) + (floor(ne_index) + 1) * (nx) + 0.5);
            return (1 - alpha) * (1 - beta) * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) + alpha * (1 - beta) * tex3D<float>(my_tex, Coord_i1j, coord_B, coord_H) +
                    (1 - alpha) * beta * tex3D<float>(my_tex, Coord_ij1, coord_B, coord_H) + alpha * beta * tex3D<float>(my_tex, Coord_i1j1, coord_B, coord_H);
        }
    }
    else if (ne_index != (int)ne_index) //Condition for indexne not integer and indexte being an integer
    {//
        alpha = ne_index - floor(ne_index);
        Coord_i = ((te_index) + floor(ne_index) * (nx) + 0.5);
        Coord_i1 = ((te_index) + (floor(ne_index) + 1) * (nx) + 0.5);
        return (1 - alpha) * tex3D<float>(my_tex, Coord_i, coord_B, coord_H) + alpha * tex3D<float>(my_tex, Coord_i1, coord_B, coord_H);
    }
    else //Condition for indexte not integer and indexne being an integer
    {
        alpha = ne_index - floor(ne_index);
        if (te_value < t_break && te_value > t_lbreak){//
            frac_break = t_break - t_lbreak;
            alpha_lower = (te_value - t_lbreak)/frac_break;
            Coord_ij = (floor(te_index) + (ne_index) * (nx) + 0.5);
            Coord_im2j =((floor(te_index) - 2) + (floor(ne_index))  * (nx) + 0.5);
            Coord_im1j = ((floor(te_index) - 1) + (floor(ne_index)) * (nx) + 0.5);
            Coord_ihalfj = 3 * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) - 3 * tex3D<float>(my_tex, Coord_im1j, coord_B, coord_H) + tex3D<float>(my_tex, Coord_im2j, coord_B, coord_H);

            return (1 - alpha_lower) * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) + alpha_lower * Coord_ihalfj;
        }else if(te_value >t_break && te_value < t_ubreak){//
            alpha_upper = (te_value - t_break)/(t_ubreak - t_break);
            Coord_ij = (floor(te_index + 1) + floor(ne_index) * (nx) + 0.5);
            Coord_iM2j =((floor(te_index) + 3) + (floor(ne_index))  * (nx) + 0.5);
            Coord_iM1j = ((floor(te_index) + 2) + (floor(ne_index)) * (nx) + 0.5);
            Coord_ihalfj = 3 * tex3D<float>(my_tex, Coord_ij, coord_B, coord_H) - 3 * tex3D<float>(my_tex, Coord_iM1j, coord_B, coord_H) + tex3D<float>(my_tex, Coord_iM2j, coord_B, coord_H);
            return (1 - alpha_upper) * Coord_ihalfj + alpha_upper * tex3D<float>(my_tex, Coord_i1j, coord_B, coord_H);

        }else{//
            alpha = te_index - floor(te_index);
            Coord_i = (floor(te_index) + (ne_index) * (nx) + 0.5);
            Coord_i1 = ((floor(te_index) + 1) + (ne_index) * (nx) + 0.5);
            return (1 - alpha) * tex3D<float>(my_tex, Coord_i, coord_B, coord_H) + alpha * tex3D<float>(my_tex, Coord_i1, coord_B, coord_H);
        }
    }
}

__global__ void cooling_function(cudaTextureObject_t my_tex, float H, float B, float ne, float te)
{
    printf("Cooling value = %lf\n", interpolate_ne_te(my_tex, H, B, ne, te));
    return;
}

__global__ void cooling_function_marcel(cudaTextureObject_t my_tex, float H, double *a1, double *a2, double *value)
{
    double ne_test, B_test, mu = 0.1;
    
    for (int i = 0; i < 20; i++)
    {
        ne_test = a1[i] / (H * THOMSON_CGS);
        for (int k = 0; k < 20; k++)
        {
            B_test = sqrt(2 * mu * BOLTZ_CGS * ne_test * a2[k]);
            value[20 * i + k] = interpolate_ne_te(my_tex, log10(H), log10(B_test), log10(ne_test), log10(a2[k]));
        }
    }
    return;
}


__global__ void cooling_function_test(cudaTextureObject_t my_tex, double * H, double * B, double * ne, double *te, double *value)
{
    int i;

    for (i = 0; i < N_RESOLUTION; i++){
        value[i] = interpolate_ne_te(my_tex, H[i], B[i], ne[i], te[i]);
    }
    return;
}

/*This function utilizes binary search to search for the closest index to the table for GLOBAL_MEMORY_TEST, used to find global_memory indexing*/
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

void random_log_value(double start, double end, int num, double *result)
{
    double log_start = log10(start); // Initial value
    double log_end = log10(end);     // End value
    srand(time(NULL));
    int i;
    for (i = 0; i < num; ++i)
    {
        double random_fraction = (double)rand() / RAND_MAX; // Random value between 0 and 1
        result[i] = log_start + random_fraction * (log_end - log_start);
    }
    return;
}

__global__ void cooling_function_comparison_global_v2(cudaTextureObject_t my_tex, double *H, double *B, double *ne, double *te, double *value)
{
    int total_elements = SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < total_elements; i += NBLOCKS * NTHREADS)
    {
        int l = i % SIZEOF_TEST;
        int k = (i / SIZEOF_TEST) % SIZEOF_TEST;
        int j = (i / (SIZEOF_TEST * SIZEOF_TEST)) % SIZEOF_TEST;
        int i_idx = i / (SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST);

        float coord_H = ((H[i_idx] - 3.) * (SIZEOF_H - 1) / 9. + 0.5);
        float coord_B = ((B[j]) * (SIZEOF_B - 1) / 10. + 0.5);
        float coord_ne = ((ne[k] - 2.) * (SIZEOF_NE - 1) / 23. + 0.5);
        float coord_te = ((te[l] - 2.) * (SIZEOF_TE - 1) / 13. + 0.5);
        int indexTe = floor(coord_te);
        int indexNe = floor(coord_ne);

        float fracTe = coord_te - indexTe;
        float fracNe = coord_ne - indexNe;

        float c00 = tex3D<float>(my_tex, indexTe + indexNe * SIZEOF_TE + 0.5, coord_B, coord_H);
        float c01 = tex3D<float>(my_tex, indexTe + 1 + indexNe * SIZEOF_TE + 0.5, coord_B, coord_H);
        float c10 = tex3D<float>(my_tex, indexTe + (indexNe + 1) * SIZEOF_TE + 0.5, coord_B, coord_H);
        float c11 = tex3D<float>(my_tex, indexTe + 1 + (indexNe + 1) * SIZEOF_TE + 0.5, coord_B, coord_H);

        float c0 = c00 * (1 - fracTe) + c01 * fracTe;
        float c1 = c10 * (1 - fracTe) + c11 * fracTe;

        value[i] = c0 * (1 - fracNe) + c1 * fracNe;
        if(i == 20)
        printf("value[%d] = %lf\n", i, value[i]);
    }
    return;
}

__global__ void global_memory_reading_v2(double *parameterH, double *parameterB, double *parameterNe, double *parameterTe, double *cooling, double *value) {
    int total_elements = SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < total_elements; i += NBLOCKS * NTHREADS) {
        int l = i % SIZEOF_TEST;
        int k = (i / SIZEOF_TEST) % SIZEOF_TEST;
        int j = (i / (SIZEOF_TEST * SIZEOF_TEST)) % SIZEOF_TEST;
        int i_idx = i / (SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST);

        int indexH = floor((parameterH[i_idx] - 3.) * (SIZEOF_H - 1) / 9. + 0.5);
        int indexB = floor((parameterB[j]) * (SIZEOF_B - 1) / 10. + 0.5);
        int indexNe = floor((parameterNe[k] - 2.) * (SIZEOF_NE - 1) / 23. + 0.5);
        int indexTe = floor((parameterTe[l] - 2.) * (SIZEOF_TE - 1) / 13. + 0.5);

        indexH  = max(0, min(SIZEOF_H - 1, indexH));
        indexB  = max(0, min(SIZEOF_B - 1, indexB));
        indexNe = max(0, min(SIZEOF_NE - 1, indexNe));
        indexTe = max(0, min(SIZEOF_TE - 1, indexTe));
        // Perform interpolation between the four-dimensional values
        double fracH = (parameterH[i_idx] - 3.) * (SIZEOF_H - 1) / 9. + 0.5 - indexH;
        double fracB = (parameterB[j]) * (SIZEOF_B - 1) / 10. + 0.5 - indexB;
        double fracNe = (parameterNe[k] - 2.) * (SIZEOF_NE - 1) / 23. + 0.5 - indexNe;
        double fracTe = (parameterTe[l] - 2.) * (SIZEOF_TE - 1) / 13. + 0.5 - indexTe;

        double c0000 = cooling[INDEX(indexH, indexB, indexNe, indexTe)];
        double c0001 = cooling[INDEX(indexH, indexB, indexNe, indexTe + 1)];
        double c0010 = cooling[INDEX(indexH, indexB, indexNe + 1, indexTe)];
        double c0011 = cooling[INDEX(indexH, indexB, indexNe + 1, indexTe + 1)];
        double c0100 = cooling[INDEX(indexH, indexB + 1, indexNe, indexTe)];
        double c0101 = cooling[INDEX(indexH, indexB + 1, indexNe, indexTe + 1)];
        double c0110 = cooling[INDEX(indexH, indexB + 1, indexNe + 1, indexTe)];
        double c0111 = cooling[INDEX(indexH, indexB + 1, indexNe + 1, indexTe + 1)];
        double c1000 = cooling[INDEX(indexH + 1, indexB, indexNe, indexTe)];
        double c1001 = cooling[INDEX(indexH + 1, indexB, indexNe, indexTe + 1)];
        double c1010 = cooling[INDEX(indexH + 1, indexB, indexNe + 1, indexTe)];
        double c1011 = cooling[INDEX(indexH + 1, indexB, indexNe + 1, indexTe + 1)];
        double c1100 = cooling[INDEX(indexH + 1, indexB + 1, indexNe, indexTe)];
        double c1101 = cooling[INDEX(indexH + 1, indexB + 1, indexNe, indexTe + 1)];
        double c1110 = cooling[INDEX(indexH + 1, indexB + 1, indexNe + 1, indexTe)];
        double c1111 = cooling[INDEX(indexH + 1, indexB + 1, indexNe + 1, indexTe + 1)];

        double c00 = c0000 * (1 - fracTe) + c0001 * fracTe;
        double c01 = c0010 * (1 - fracTe) + c0011 * fracTe;
        double c10 = c0100 * (1 - fracTe) + c0101 * fracTe;
        double c11 = c0110 * (1 - fracTe) + c0111 * fracTe;

        double c0 = c00 * (1 - fracNe) + c01 * fracNe;
        double c1 = c10 * (1 - fracNe) + c11 * fracNe;

        double c00_next = c1000 * (1 - fracTe) + c1001 * fracTe;
        double c01_next = c1010 * (1 - fracTe) + c1011 * fracTe;
        double c10_next = c1100 * (1 - fracTe) + c1101 * fracTe;
        double c11_next = c1110 * (1 - fracTe) + c1111 * fracTe;

        double c0_next = c00_next * (1 - fracNe) + c01_next * fracNe;
        double c1_next = c10_next * (1 - fracNe) + c11_next * fracNe;

        double c_final = c0 * (1 - fracB) + c1 * fracB;
        double c_final_next = c0_next * (1 - fracB) + c1_next * fracB;

        value[i] = c_final * (1 - fracH) + c_final_next * fracH;
        if(i == 20)
        printf("value[%d] = %lf\n", i, value[i]);
    }
    return;
}

__global__ void cooling_function_comparison_global(cudaTextureObject_t my_tex, double *H, double *B, double *ne, double *te, double *value)
{
    for (int i = 0; i < SIZEOF_TEST; i++)
    {
        for (int j = 0; j < SIZEOF_TEST; j++)
        {
            for (int k = 0; k < SIZEOF_TEST; k++)
            {
                for (int l = 0; l < SIZEOF_TEST; l++)
                {
                    value[INDEX(i, j, k, l)] = no_interpolation(my_tex, H[i], B[j], ne[k], te[l]);
                }
            }
        }
    }
    return;
}
int main()
{
    cudaError_t cudaStatus;
#if (SINGLE_TEST)
    float read0, read1, read2, read3;
    float loop = 100;

    char str[1];
    CreateTexture();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error creating texture %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    while (loop > 1)
    {
        fprintf(stderr,"scale_height value:\n");
        scanf("%f", &read0);
        fprintf(stderr,"Bmag value:\n");
        scanf("%f", &read1);
        fprintf(stderr,"ne value:\n");
        scanf("%f", &read2);
        fprintf(stderr,"Te value:\n");
        scanf("%f", &read3);
        cooling_function<<<1, 1>>>(coolTexObj, read0, read1, read2, read3);
        cudaDeviceSynchronize();
        fprintf(stderr,"Do you want to read other values? y/n\n");
        scanf("%s", str);
        if (strcmp(str, "n") == 0)
        {
            loop = 0;
        }
    }
    cudaDestroyTextureObject(coolTexObj);
#elif (RESOLUTION_TEST)
    double *H_test, *B_test, *ne_test, *Te_test, *cool_test;
    H_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    B_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    ne_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    Te_test = (double *)malloc(N_RESOLUTION * sizeof(double));
    cool_test = (double *)malloc(N_RESOLUTION * sizeof(double));
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


    fprintf(stderr, "Initializing resolution test reading\n");
    char filename[] = "../txts/hamr_test.txt";
    FILE *file_result;
    file_result = fopen(filename, "w");
    FILE *file_height_test;
    file_height_test = fopen("../txts/logH.txt", "r");
    FILE *file_e_density_test;
    file_e_density_test = fopen("../txts/logne.txt", "r");
    FILE *file_temperature_test;
    file_temperature_test = fopen("../txts/logTe.txt", "r");
    FILE *file_mag_field_test;
    file_mag_field_test = fopen("../txts/logB.txt", "r");
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

    cudaMemcpy(d_H_test, H_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_test, B_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ne_test, ne_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Te_test, Te_test, N_RESOLUTION * sizeof(double), cudaMemcpyHostToDevice);
    fprintf(stderr, "Reading and getting values from texture memory...\n");

    cooling_function_test<<<1, 1>>>(coolTexObj, d_H_test, d_B_test, d_ne_test, d_Te_test, d_cool_test);
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
    fprintf(stderr, "Resolution test sucessfull, table generated: %s\n", filename);
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
    cudaMalloc(&d_results, SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST * SIZEOF_TEST * sizeof(double));
    fprintf(stderr,"Initializing GLOBAL MEMORY testing\n");
    FILE *file_height_test;
    file_height_test = fopen("../parameters/scale_height_40.txt", "r");
    FILE *file_e_density_test;
    file_e_density_test = fopen("../parameters/ne_40.txt", "r");
    FILE *file_temperature_test;
    file_temperature_test = fopen("../parameters/te_40.txt", "r");
    FILE *file_mag_field_test;
    file_mag_field_test = fopen("../parameters/mag_40.txt", "r");
    FILE *file_cooling_test;
    file_cooling_test = fopen("../tables/cooling_table_40.bin", "rb");
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
        // fprintf(stderr,"i = %d \n", i);
        fread(&cool_test[i], sizeof(double), 1, file_cooling_test);
        // fprintf(stderr,"cool_test[%d] = %lf\n", i, cool_test[i]);
    }
    fclose(file_height_test);
    fclose(file_e_density_test);
    fclose(file_temperature_test);
    fclose(file_mag_field_test);
    fclose(file_cooling_test);
    
    random_log_value(1e5, 1e8, SIZEOF_TEST, H_random);
    random_log_value(1e0, 1e10, SIZEOF_TEST, B_random);
    random_log_value(1e2, 1e15, SIZEOF_TEST, Te_random);
    random_log_value(1e2, 1e25, SIZEOF_TEST, ne_random);
    fprintf(stderr,"Transfering data from Host to Device... \n");
    cudaMemcpy(d_cool_test, cool_test, TABLE_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    free(cool_test);

    cudaMemcpy(d_H_random, H_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_random, B_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ne_random, ne_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Te_random, Te_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    free(H_random);
    free(B_random);
    free(ne_random);
    free(Te_random);
    CreateTexture();

    fprintf(stderr,"Starting table lookup...\n");
    start_time = clock();
    cooling_function_comparison_global_v2<<<NBLOCKS, NTHREADS>>>(coolTexObj, d_H_random, d_B_random, d_Ne_random, d_Te_random, d_results);
    cudaDeviceSynchronize();
    
    end_time = clock();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    fprintf(stderr,"Number of values analyzed = %d, duration texture: %.6f seconds\n", SIZEOF_TEST, duration);

    start_time = clock();
    global_memory_reading_v2<<<NBLOCKS, NTHREADS>>>(d_H_random, d_B_random, d_Ne_random, d_Te_random, d_cool_test, d_results);
    cudaDeviceSynchronize();
    end_time = clock();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    fprintf(stderr,"Number of values analyzed = %d, duration global: %.6f seconds\n", SIZEOF_TEST, duration);
    //free all the arrays and texture objects used
    cudaDestroyTextureObject(coolTexObj);
    cudaFree(d_cool_test);
    cudaFree(d_H_random);
    cudaFree(d_B_random);
    cudaFree(d_Ne_random);
    cudaFree(d_Te_random);
    cudaFree(d_results);
    

#endif
    return 0;
}
