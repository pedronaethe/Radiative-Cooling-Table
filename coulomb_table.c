// Cooling Functions including Bremmstrahlung, Synchrotron and Comptonized Synchrotron
// Code by Pedro Naethe Motta
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define C_pi (3.14159265358979)    /*value of pi*/
#define C_euler (2.71828182845904) /* euler constant*/
#define C_sigma (5.67051e-5)       /*Stephan Boltzmann constant*/
#define BOLTZ_CGS (1.3806505e-16)  /*Boltzmann constant*/
#define PLANCK_CGS (6.62606876e-27) /*Planck constant*/
#define MH_CGS (1.673534e-24) /*Mass hydrogen molecule*/
#define ME_CGS (9.1094e-28) /*Mass hydrogen molecule*/
#define C_CGS (2.99792458e10)       /*Speed of light*/
#define C_G (6.6726e-8)             /*Gravitational constant*/
#define C_Msun (2e33)              /*Sun mass*/
#define THOMSON_CGS (6.6524e-25)    /*Thomson cross section*/

/*Mass of the Black hole*/
#define C_Mbh (10. * C_Msun)
#define C_GM (C_Mbh * C_G)


#define ITMAX (100)     /* Used in calculating gamma function*/
#define eps (3e-7)
#define maxSteps (1e3)
#define FPMIN (1.0e-30) /* Used in calculating gamma function*/

double edens;
double etemp;
double itemp;

double gammln(double xxgam);

/***********************************************************************************************/
double bessi0(double xbess)
{
    double ax, ans;
    double y;
    if ((ax = fabs(xbess)) < 3.75)
    {
        y = xbess / 3.75;
        y *= y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    }
    else
    {
        y = 3.75 / ax;
        ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 + y *
                                                                                                                                     (-0.2057706e-1 +
                                                                                                                                      y *
                                                                                                                                          (0.2635537e-1 +
                                                                                                                                           y *
                                                                                                                                               (-0.1647633e-1 + y *
                                                                                                                                                                    0.392377e-2))))))));
    }
    return ans;
}

double bessk0(double xbess)
{
    double y, ans;
    if (xbess <= 2.0)
    {
        y = xbess * xbess / 4.0;
        ans = (-log(xbess / 2.0) * bessi0(xbess)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 +
                                                                                          y * (0.3488590e-1 + y * (0.262698e-2 + y *
                                                                                                                                     (0.10750e-3 +
                                                                                                                                      y *
                                                                                                                                          0.74e-5))))));
    }
    else
    {
        y = 2.0 / xbess;
        ans = (exp(-xbess) / sqrt(xbess)) * (1.25331414 + y * (-0.7832358e-1 +
                                                               y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2 + y *
                                                                                                                               (-0.251540e-2 +
                                                                                                                                y *
                                                                                                                                    0.53208e-3))))));
    }
    return ans;
}

double bessi1(double xbess)
{
    double ax, ans;
    double y;
    if ((ax = fabs(xbess)) < 3.75)
    {
        y = xbess / 3.75;
        y *= y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 +
                                                                                     y * (0.301532e-2 + y * 0.32411e-3))))));
    }
    else
    {
        y = 3.75 / ax;
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
        ans *= (exp(ax) / sqrt(ax));
    }
    return xbess < 0.0 ? -ans : ans;
}

double bessk1(double xbess)
{
    double y, ans;
    if (xbess <= 2.0)
    {
        y = xbess * xbess / 4.0;
        ans = (log(xbess / 2.0) * bessi1(xbess)) + (1.0 / xbess) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 +
                                                                                                                    y *
                                                                                                                        (-0.1919402e-1 + y *
                                                                                                                                             (-0.110404e-2 +
                                                                                                                                              y *
                                                                                                                                                  (-0.4686e-4)))))));
    }
    else
    {
        y = 2.0 / xbess;
        ans = (exp(-xbess) / sqrt(xbess)) * (1.25331414 + y * (0.23498619 + y *
                                                                                (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2 + y *
                                                                                                                                             (0.325614e-2 +
                                                                                                                                              y *
                                                                                                                                                  (-0.68245e-3)))))));
    }
    return ans;
}

double bessk2(double xbess)
{
    int n, j;
    double bk, bkm, bkp, tox;
    n = 2;
    tox = 2.0 / xbess;
    bkm = bessk0(xbess);
    bk = bessk1(xbess);
    for (j = 1; j < n; j++)
    {
        bkp = bkm + j * tox * bk;
        bkm = bk;
        bk = bkp;
    }
    return bk;
}

/***********************************************************************************************/
double theta_e(double etemp)
{
    double result = (BOLTZ_CGS)* etemp / ((ME_CGS) * (pow(C_CGS, 2.)));
    return result;
}
double theta_i(double itemp)
{
    double result = (BOLTZ_CGS)* itemp / ((MH_CGS) * (pow(C_CGS, 2.)));
    return result;
}


double coulomb_heating(double etemp, double itemp, double edens)
{
    double coeff, th_sum, th_mean, result, K2i, K2e, K0, K1;  
    double theta_min = 1.e-2;
    double coulog = 20.;   // Coulomb logarithm ( ln Lambda )
    double Theta_i = theta_i(itemp);
    double Theta_e = theta_e(etemp); 
	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
    coeff *= pow(edens, 2.) * (itemp - etemp);
    //printf("coef = %.2e \n", coeff);
    th_sum = theta_e(etemp) + theta_i(itemp);
	th_mean = theta_e(etemp) * theta_i(itemp) / (theta_e(etemp) + theta_i(itemp));
    if (Theta_i < theta_min && Theta_e < theta_min) // approximated equations at small theta
	{
		result = coeff / sqrt(0.5 * C_pi * th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
        // printf("Oii\n");
        // printf("result= %.2e \n", result);
        // printf("coeff= %.2e \n", coeff);
        // printf("th_sum = %.2e \n", th_sum);
        // printf("th_mean = %.2e \n", th_mean);
        // printf("Theta_e = %.2e \n", Theta_e);
        // printf("Theta_i = %.2e \n", Theta_i);
	}
	else if (Theta_i < theta_min)
	{
		//bessel function
	    //printf("Oii-2\n");
		K2e = bessk2(1. / Theta_e);
		result = coeff / K2e / exp(1. / Theta_e) * sqrt(Theta_e) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
    }
	else if (Theta_e < theta_min)
	{
		//bessel function
		K2i = bessk2(1. / Theta_i);
	    //printf("Oii-3\n");
		result = coeff / K2i / exp(1. / Theta_i) * sqrt(Theta_i) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else // general form in Sadowski+17 (eq 20)
	{
	    //printf("Oii-4 \n");
		//bessel functions
		K2e = bessk2(1. / Theta_e);
		K2i = bessk2(1. / Theta_i);
		K0 = bessk0(1.0 / th_mean);
		K1 = bessk1(1.0 / th_mean);

		result = coeff / (K2e * K2i) * ((2. * th_sum * th_sum + 1.) / th_sum * K1 + 2. * K0);
	}
	//if (!isfinite(result)) result = 0.;
    //printf("result far = %.2e\n", result);

	return (result);

}


int main()
{
    FILE *file_e_density;
    file_e_density = fopen("ne.txt", "r");
    FILE *file_temperature;
    file_temperature = fopen("te.txt", "r");
    FILE *file_itemperature;
    file_itemperature = fopen("te.txt", "r");
    FILE *file_result;
    file_result = fopen("source_coulomb.txt", "w");

    float coulomb;

    if (file_e_density == NULL || file_temperature == NULL)
    {
        printf("Error Reading File\n");
        exit(0);
    }

    fprintf(file_result, "e_density, i_temperature, e_temperature, cooling\n");
    // run for logarithm values
    while (fscanf(file_e_density, "%lf,", &edens) == 1)
    {
        rewind(file_itemperature);
        while (fscanf(file_itemperature, "%lf,", &itemp) == 1)
        {
            rewind(file_temperature);
            while (fscanf(file_temperature, "%lf,", &etemp) == 1)
            {
                    fprintf(file_result, "%.2f, %.2f, %.2f,", edens, itemp, etemp);
                    edens = pow(10., edens);
                    etemp = pow(10., etemp);
                    itemp= pow(10., itemp);
                    coulomb = coulomb_heating(etemp, itemp, edens); 
                    edens = log10(edens);
                    etemp = log10(etemp);
                    itemp = log10(itemp);
                    fprintf(file_result, "%.2f\n", coulomb);
            }
        }
    }
    fclose(file_e_density);
    fclose(file_temperature);
    fclose(file_result);

    return 0;
}
