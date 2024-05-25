/*Code created by Pedro Naethe Motta to generate the .txt's file for each parameter.*/


#include <stdio.h>
#include <math.h>

#define NTABLE 40
#define HMIN 3
#define HMAX 12
#define BMIN 0
#define BMAX 10
#define NeMIN 2
#define NeMAX 25
#define TeMIN 2
#define TeMAX 15
int main()
{
    double v1 = 0;
	double v2 = 0;
	double v3 = 0;
	double v4 = 0;
	double i = 0, j = 0, k = 0, l = 0;
	char filename_height[50], filename_temperature[50], filename_electronicdensity[50], filename_mag[50];

    // Create filenames with NTABLE substituted
    sprintf(filename_height, "../parameters/scale_height_%d.txt", NTABLE);
    sprintf(filename_temperature, "../parameters/te_%d.txt", NTABLE);
    sprintf(filename_electronicdensity, "../parameters/ne_%d.txt", NTABLE);
    sprintf(filename_mag, "../parameters/mag_%d.txt", NTABLE);

    FILE *Fheight;
	FILE *Ftemperature;
	FILE *Felectronicdensity;
	FILE *Fmag;
	Fheight = fopen(filename_height, "w");
    Ftemperature = fopen(filename_temperature, "w");
    Felectronicdensity = fopen(filename_electronicdensity, "w");
    Fmag = fopen(filename_mag, "w");
    while (i <= NTABLE)
    {
        v1 = HMIN + (HMAX - HMIN) * i/(NTABLE);
        fprintf(Fheight, "%.7f\n", v1);
        i = i + 1;
    }
	while (k <= NTABLE)
	{
		v2 = NeMIN + (NeMAX - NeMIN) * k/NTABLE;
		fprintf(Felectronicdensity, "%.7f\n", v2);
		k = k + 1;
	}
	while (j <= NTABLE)
	{
		v3 = TeMIN + (TeMAX - TeMIN) * j/NTABLE;
		fprintf(Ftemperature, "%.7f\n", v3);
		j = j + 1;
	}
	while (l <= NTABLE)
	{
		v4 = BMIN + (BMAX - BMIN) * l/NTABLE;
		fprintf(Fmag, "%.7f\n", v4);
		l = l + 1;
	}

    fclose(Fheight);
	fclose(Ftemperature);
	fclose(Felectronicdensity);
	fclose(Fmag);
	return 0;
}
