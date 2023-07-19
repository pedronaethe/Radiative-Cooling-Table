/*Code created by Pedro Naethe Motta to generate the .txt's file for each parameter.*/


#include <stdio.h>
#include <math.h>


int main()
{
    double v1 = 0;
	double v2 = 0;
	double v3 = 0;
	double v4 = 0;
	double i = 0, j = 0, k = 0, l = 0;
    FILE *Fheight;
    Fheight = fopen("scale_height_33.txt", "w");
	FILE *Ftemperature;
	Ftemperature = fopen("te_33.txt", "w");
	FILE *Feletronicdensity;
	Feletronicdensity = fopen("ne_33.txt", "w");
	FILE *Fmag;
	Fmag = fopen("mag_33.txt", "w");

    while (i <= 32)
    {
        v1 = 3 + 5 * i/32;
        fprintf(Fheight, "%.7f\n", v1);
        i = i + 1;
    }
	while (k <= 32)
	{
		v2 = 2 + 23 * k/32;
		fprintf(Feletronicdensity, "%.7f\n", v2);
		k = k + 1;
	}
	while (j <= 32)
	{
		v3 = 2 + 13 * j/32;
		fprintf(Ftemperature, "%.7f\n", v3);
		j = j + 1;
	}
	while (l <= 32)
	{
		v4 = 10 * l/32;
		fprintf(Fmag, "%.7f\n", v4);
		l = l + 1;
	}

    fclose(Fheight);
	fclose(Ftemperature);
	fclose(Feletronicdensity);
	fclose(Fmag);
	return 0;
}
