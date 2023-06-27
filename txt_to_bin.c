
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define COULOMB (0)

int main()
{
    fprintf(stderr, "Loading Table...\n");

    FILE *infile;
    FILE *file_result;
    double scale_height;
    double bmag;
    double ne;
    double te;
    double cool;


    // Reading the cooling table
    infile = fopen("cooling_table_33_05.txt", "r");
    file_result = fopen("cooling_table_33_05.bin", "w");

    
    if (infile == NULL)
    {
        fprintf(stderr, "Unable to open cooling file.\n");
        exit(1);
    }
    fprintf(stderr, "Reading Data...\n");
    fscanf(infile, "%*[^\n]\n"); // this command is to ignore the first line.
    #if(COULOMB)
    while (fscanf(infile, "%lf, %lf, %lf, %lf", &scale_height, &bmag, &ne, &cool) == 4)
    {
        fwrite(&cool, sizeof(double), 1, file_result); // Write cooling_values_all array
    }
    #else
    while (fscanf(infile, "%lf, %lf, %lf, %lf, %lf", &scale_height, &bmag, &ne, &te, &cool) == 5)
    {
        fwrite(&cool, sizeof(double), 1, file_result); // Write cooling_values_all array
    }
    #endif


    fprintf(stderr, "Finished transfering txt to .binary data\n");
    fclose(infile);
    fclose(file_result);

    // Free arrays used to read in table data
    fprintf(stderr, "Table Loaded!\n");

    return 0 ;
}
