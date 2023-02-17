#include <math.h>
#include <stdio.h>
#include <stdlib.h>

cudaTextureObject_t coulTexObj;
cudaArray *cucoulArray = 0;

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

// Load the coulomb_table into the CPU Memory.
void Load_coulomb_Tables(float *coulomb_table)
{
    double *ne_arr;
    double *te_arr;
    double *ti_arr;
    double *coul_arr;

    double ti;
    double ne;
    double te;
    double coul;

    int i = 0;
    int nx = 100; // Number of ne data.
    int ny = 100; // Number of Ti data.
    int nz = 100; // Number of Te data.

    FILE *infile;

    // Allocate arrays for temperature, electronic density and scale_height data.
    ne_arr = (double *)malloc( nx * ny * nz * sizeof(double));
    te_arr = (double *)malloc(nx * ny * nz * sizeof(double));
    coul_arr = (double *)malloc(nx * ny *  nz * sizeof(double));
    ti_arr = (double *)malloc(nx * ny * nz * sizeof(double));

    // Reading the coulomb table
    infile = fopen("source_coulomb.txt", "r");

    if (infile == NULL)
    {
        printf("Unable to open coulomb file.\n");
        exit(1);
    }

    fscanf(infile, "%*[^\n]\n"); // this command is to ignore the first line.
    while (fscanf(infile, "%lf, %lf, %lf, %lf", &ne, &ti, &te, &coul) == 4)
    {
        ne_arr[i] = ne;
        te_arr[i] = te;
        ti_arr[i] = ti;
        coul_arr[i] = coul;

        i++;
    }

    fclose(infile);
    // copy data from coulomb array into the table
    for (i = 0; i < nx * ny * nz; i++)
    {
        coulomb_table[i] = float(coul_arr[i]);
    }

    // Free arrays used to read in table data
    free(ne_arr);
    free(te_arr);    
    free(ti_arr);
    free(coul_arr);
    return;
}

void CreateTexture(void)
{

    float *coulomb_table; //Device Array with coulomb floats
    // number of elements in each variable
    const int nx = 100; //ne
    const int ny = 100; //Ti
    const int nz = 100; //Te
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
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
    cudaCreateTextureObject(&coulTexObj, &texRes, &texDescr, NULL);
    return;
}
__global__ void coulomb_function(cudaTextureObject_t my_tex, float a1, float a2, float a3)
{
    float v1, v2, v3, lambda;

    //Values for testing;
    v1 = a1; //Bmag parameter
    v2 = a2; //ne parameter
    v3 = a3; //te parameter
    printf("Values you chose:\n");
    printf("ne = %f, Ti = %f, Te = %f\n", v1, v2, v3);

    // For the normalized version only.
    const int nx = 100; //Number of te used to generate table
    const int ny = 100; //Number of ne used to generate table
    const int nz = 100; //Number of Bmag used to generate table
     v1 = (round((v1 - 12) * (nx - 1)/10) + 0.5)/nx; // ne
     v2 = (round((v2 - 4) * (nz - 1)/11) + 0.5)/nz; // Ti
     v3 = (round((v3 - 4) * (ny - 1)/11) + 0.5)/ny; // Te

    printf("Coordinates in texture grid:\n");
    printf("Ne = %f, Ti = %f, Te = %f\n",v1, v2, v3);

    //For the non normalized version only.
    //lambda = tex3D<float>(coulTexObj, v3 + 0.5f, v2 + 0.5f, v1 + 0.5f); 

    // //For the normalized version only.
    lambda = tex3D<float>(my_tex, v3, v2, v1); 
    printf("coulomb value = %lf\n", lambda);
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
	    printf("Ne value:\n");
	    scanf("%f", &read1);
	    printf("Ti value:\n");
	    scanf("%f", &read2);
	    printf("Te value:\n");
	    scanf("%f", &read3);
	    coulomb_function<<<1, 1>>>(coulTexObj, read1, read2, read3);
	    printf("Do you want to read other values? y/n\n");
	    scanf("%s", str);
	    if (strcmp(str, "n") == 0)
	    {
	    	loop = 0;
	    }
	}
    cudaDestroyTextureObject(coulTexObj);
    return 0;
}
