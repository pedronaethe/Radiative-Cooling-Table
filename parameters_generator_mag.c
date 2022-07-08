
#include <stdio.h>
#include <math.h>


int main()
{
	double v2 = 0;
	double v3 = 0;
	double v4 = 0;
	double j = 0, k = 0, l = 0;
	FILE *Ftemperature;
	Ftemperature = fopen("te.txt", "w");
	FILE *Feletronicdensity;
	Feletronicdensity = fopen("ne.txt", "w");
	FILE *Fmag;
	Fmag = fopen("mag.txt", "w");

	// If you want to generate the table for R, ne and Te out of the log space use the code below:
	//
	// for (R = 3e6; R < 1e12; R = pow(M_E, log(R) + logspacestep(3e6, 1e12, steps)))
	// {
	// 	fprintf(Fradius, "%le\n", R);
	// }
	// for (Te = 1e6; Te < 1e10; Te = pow(M_E, log(Te) + logspacestep(1e6, 1e10, steps)))
	// {
	// 	fprintf(Ftemperature, "%le\n", Te);
	// }
	// for (ne = 1e12; ne < 2e20; ne = pow(M_E, log(ne) + logspacestep(1e12, 2e20, steps)))
	// {
	// 	fprintf(Feletronicdensity, "%le\n", ne);
	// }

	while (k <= 99)
	{
		v2 = 12 + 8 * k/99;
		fprintf(Feletronicdensity, "%.2f\n", v2);
		k = k + 1;
	}
	while (j <= 99)
	{
		v3 = 6 + 4 * j/99;
		fprintf(Ftemperature, "%.2f\n", v3);
		j = j + 1;
	}
	while (l <= 99)
	{
		v4 = 0 + 7 * l/99;
		fprintf(Fmag, "%.2f\n", v4);
		l = l + 1;
	}


	fclose(Ftemperature);
	fclose(Feletronicdensity);
	fclose(Fmag);
	return 0;
}
