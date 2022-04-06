
#include <stdio.h>
#include <math.h>

double logspacestep(double initialv, double finalv, int steps)
{
	return ((log(finalv) - log(initialv)) / steps);
}

int main()
{
	double R;
	double ne;
	double Te;
	int count;
	int steps = 70;
	double values[10];
	FILE *Fradius;
	Fradius = fopen("../radius.txt", "w");
	FILE *Ftemperature;
	Ftemperature = fopen("../temperature.txt", "w");
	FILE *Feletronicdensity;
	Feletronicdensity = fopen("../e_density.txt", "w");
	for (R = 3e6; R < 1e12; R = pow(M_E, log(R) + logspacestep(3e6, 1e12, steps)))
	{
		fprintf(Fradius, "%le\n", R);
	}
	for (Te = 1e6; Te < 1e10; Te = pow(M_E, log(Te) + logspacestep(1e6, 1e10, steps)))
	{
		fprintf(Ftemperature, "%le\n", Te);
	}
	for (ne = 1e12; ne < 2e20; ne = pow(M_E, log(ne) + logspacestep(1e12, 2e20, steps)))
	{
		fprintf(Feletronicdensity, "%le\n", ne);
	}

	for (count = 0; count < 10 && fscanf(Fradius, " %lf", &values[count]) != 1 && printf("%lf\n", values[count]); count++)
		;
	fclose(Fradius);
	fclose(Ftemperature);
	fclose(Feletronicdensity);
	return 0;
}
