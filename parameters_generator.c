
#include <stdio.h>
#include <math.h>

double logspacestep(double initialv, double finalv, int steps)
{
	return ((log(finalv) - log(initialv)) / steps);
}

int main()
{
	double v1 = 0;
	double v2 = 0;
	double v3 = 0;
	double i = 0, j = 0, k = 0;
	FILE *Fradius;
	Fradius = fopen("r.txt", "w");
	FILE *Ftemperature;
	Ftemperature = fopen("te.txt", "w");
	FILE *Feletronicdensity;
	Feletronicdensity = fopen("ne.txt", "w");

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

	while (i <= 69)
	{
		v1 = 6 + 6 * i/69;
		fprintf(Fradius, "%.2f\n", v1);
		i = i + 1;
	}
	while (k <= 69)
	{
		v2 = 12 + 8 * k/69;
		fprintf(Feletronicdensity, "%.2f\n", v2);
		k = k + 1;
	}
	while (j <= 69)
	{
		v3 = 6 + 4 * j/69;
		fprintf(Ftemperature, "%.2f\n", v3);
		j = j + 1;
	}

	fclose(Fradius);
	fclose(Ftemperature);
	fclose(Feletronicdensity);
	return 0;
}
