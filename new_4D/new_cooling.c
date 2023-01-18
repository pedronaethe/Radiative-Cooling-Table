// Cooling Functions including Bremmstrahlung, Synchrotron and Comptonized Synchrotron
// Code by Pedro Naethe Motta
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define MH_CGS (1.6726231e-24) /*hydrogen mass*/
#define ARAD_CGS (7.5646e-15) /*radiative constant*/
#define C_pi (3.14159265358979)    /*value of pi*/
#define C_euler (2.71828182845904) /* euler constant*/
#define C_sigma (5.67051e-5)       /*Stephan Boltzmann constant*/
#define BOLTZ_CGS (1.3806505e-16)  /*Boltzmann constant*/
#define PLANCK_CGS (6.62606876e-27) /*Planck constant*/
#define ERM_CGS (9.1093826e-28)     /*Electron rest mass CGS*/
#define C_CGS (2.99792458e10)       /*Speed of light*/
#define C_G (6.6726e-8)             /*Gravitational constant*/
#define C_Msun (2e33)              /*Sun mass*/
#define THOMSON_CGS (6.6524e-25)    /*Thomson cross section*/

/*Mass of the Black hole*/
#define C_Mbh (10. * C_Msun)
#define C_GM (C_Mbh * C_G)

#define R_CGS (C_G * C_Mbh/(C_CGS**2))


#define ITMAX (100)     /* Used in calculating gamma function*/
#define eps (3e-7)
#define maxSteps (1e3)
#define FPMIN (1.0e-30) /* Used in calculating gamma function*/


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

void gcf(double *gammcf, double agam, double xgam,
         double *gln)
{ // Function used in the calculation of incomplete gamma function
    int i;
    double an, b, c, d, del, h;
    *gln = gammln(agam);
    b = xgam + 1.0 - agam;
    c = 1.0 / FPMIN;
    d = 1.0 / b;
    h = d;
    for (i = 1; i <= ITMAX; i++)
    {
        an = -i * (i - agam);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < FPMIN)
            d = FPMIN;
        c = b + an / c;
        if (fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (fabs(del - 1.0) < eps)
            break;
    }
    *gammcf = exp(-xgam + agam * log(xgam) - (*gln)) * h;
}

void gser(double *gamser, double agam, double xgam,
          double *gln)
{ // Function used in the calculation of incomplete gamma function
    int n;
    double sum, del, ap;

    *gln = gammln(agam);
    if (xgam <= 0.0)
    {
        if (xgam < 0.0)
            ;
        *gamser = 0.0;
        return;
    }
    else
    {
        ap = agam;
        del = sum = 1.0 / agam;
        for (n = 1; n <= ITMAX; n++)
        {
            ++ap;
            del *= xgam / ap;
            sum += del;
            if (fabs(del) < fabs(sum) * eps)
            {
                *gamser = sum * exp(-xgam + agam * log(xgam) - (*gln));
                return;
            }
        }
        return;
    }
}

double gammp(double agam, double xgam)
{
    int n;
    if (agam < 0)
    {
        agam = -agam;
        if (n < 1)
        {
            printf("Valor negativo para a na função gammp!!! a= %le\n", -agam);
            n = 2;
        }
    }
    double gamser, gammcf, gln;
    if (xgam < (agam + 1.0))
    {
        gser(&gamser, agam, xgam, &gln);
        return gamser;
    }
    else
    {
        gcf(&gammcf, agam, xgam, &gln);
        return 1.0 - gammcf;
    }
}

double gammq(double agam, double xgam)
{
    double gamser, gammcf, gln;
    if (xgam < (agam + 1.0))
    {
        gser(&gamser, agam, xgam, &gln);
        return 1.0 - gamser;
    }
    else
    {
        gcf(&gammcf, agam, xgam, &gln);
        return gammcf;
    }
}

double gammln(double xxgam)
{
    double x, y, tmp, ser;
    static double cof[6] = {76.18009172947146, -86.50532032941677,
                            24.01409824083091, -1.231739572450155,
                            0.1208650973866179e-2, -0.5395239384953e-5};
    int j;
    y = x = xxgam;
    tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    ser = 1.000000000190015;
    for (j = 0; j <= 5; j++)
        ser += cof[j] / ++y;
    return -tmp + log(2.5066282746310005 * ser / x);
}

/***********************************************************************************************/
double thetae(double etemp)
{
    double result = (BOLTZ_CGS)* etemp / ((ERM_CGS) * (pow(C_CGS, 2.)));
    return result;
}

double scale_height(double radius, double edens, double etemp)
{
    double rho = edens * MH_CGS;
    if(ARAD_CGS * pow(etemp, 3.)/(edens * BOLTZ_CGS) > 1.){
	    return pow(ARAD_CGS * pow(etemp,4.)/(3 * rho * C_G * C_Mbh) ,1/2) * pow(radius, 3/2);
    }
    else{
        return pow(BOLTZ_CGS * etemp/(MH_CGS * C_G * C_Mbh),1/2) * pow(radius, 3/2);
    }
}

double f(double radius, double x, double edens, double etemp, double mag_field)
{
    if (thetae(etemp) < 0.03)
    {
        return pow(C_euler, 1.8899 * pow(x, 1. / 3.)) -
               2.49 * pow(10., -10.) * 12. * C_pi * edens * scale_height(radius, edens, etemp) / (mag_field) * 1 / (2 * pow(thetae(etemp), 5.)) *
                   (1 / pow(x, 7. / 6.) + 0.4 / pow(x, 17. / 12.) + 0.5316 / pow(x, 5. / 3.));
    }
    else
    {
        return pow(C_euler, 1.8899 * pow(x, 1. / 3.)) - 2.49 * pow(10., -10.) * 12. * C_pi * edens * scale_height(radius, edens, etemp) / (mag_field) * 1 /
                                                            (pow(thetae(etemp), 3.) * bessk2(1 / thetae(etemp))) *
                                                            (1 / pow(x, 7. / 6.) + 0.4 / pow(x, 17. / 12.) +
                                                             0.5316 / pow(x, 5. / 3.));
    }
}

/*Function that returns the root from Secant Method*/
double secant(double radius, double edens, double etemp, double mag_field)
{
    int iter = 1;
    double x1 = 1.;
    double x2 = 20000.;
    double x3;
    do
    {
        x3 = (x1 * f(radius, x2, edens, etemp, mag_field) - x2 * f(radius, x1, edens, etemp, mag_field)) / (f(radius, x2, edens, etemp, mag_field) - f(radius, x1, edens, etemp, mag_field));
        if (isnan(x3) || isinf(x3))
        {
            x1 = x2;
            break;
        }
        x1 = x2;
        x2 = x3;
        iter++;
        // printf("x3 = %lf\n", x3);
        // printf("x1 = %lf\n", x1);
    } while (fabs(f(radius, x3, edens, etemp, mag_field)) > eps && iter <= maxSteps);
    return x1;
}

double bremmstrahlung_ee(double edens, double etemp)
{
    double th_e = thetae(etemp);
    double result;
    if (th_e < 1)
    {
        result = 2.56 * pow(10., -22.) * pow(edens, 2.) * pow(thetae(etemp), 3. / 2.) *
                 (1 + 1.10 * thetae(etemp) + pow(thetae(etemp), 2.) - 1.25 * pow(thetae(etemp), 5. / 2.));
    }
    else
    {
        result = 3.42 * pow(10., -22.) * pow(edens, 2.) * thetae(etemp) * (log(1.123 * thetae(etemp)) + 1.28);
    }
    return result;
}

double bremmstrahlung_ei(double edens, double etemp)
{
    double th_e = thetae(etemp);
    double result;
    if (th_e >= 1)
    {
        result = 1.48 * pow(10., -22.) * pow(edens, 2.) * (9 * thetae(etemp)) / (2 * C_pi) *
                 (log(1.123 * thetae(etemp) + 0.48) + 1.5);
    }
    else
    {
        result = 1.48 * pow(10., -22.) * pow(edens, 2.) * 4 * pow((2 * thetae(etemp) / pow(C_pi, 3.)), 1. / 2.) *
                 (1 + 1.781 * pow(thetae(etemp), 1.34));
    }
    return result;
}

double bremmscooling_rate(double edens, double etemp)
{
    double result = bremmstrahlung_ee(edens, etemp) + bremmstrahlung_ei(edens, etemp);
    return result;
}

double crit_freq(double radius, double edens, double etemp, double mag_field)
{
    return (1.5) * (2.80 * pow(10,6) * mag_field) * pow(thetae(etemp), 2.) * secant(radius, edens, etemp, mag_field);
}

/*Synchtron radiation calculation*/
double rsync(double radius, double edens, double etemp, double mag_field)
{
    double nuzero = 2.80 * pow(10., 6.) * mag_field;
    double a1 = 2 / (3 * nuzero * pow(thetae(etemp), 2.));
    double a2 = 0.4 / pow(a1, 1. / 4.);
    double a3 = 0.5316 / pow(a1, 1. / 2.);
    double a4 = 1.8899 * pow(a1, 1. / 3.);
    if (thetae(etemp) > 0.03)
    {
        double result =
            2 * C_pi * BOLTZ_CGS * etemp * pow(crit_freq(radius, edens, etemp, mag_field), 3.) / (3 * scale_height(radius, edens, etemp) * pow(C_CGS, 2.)) +
            6.76 * pow(10., -28.) * edens / (bessk2(1 / thetae(etemp)) * pow(a1, 1. / 6.)) *
                (1 / pow(a4, 11. / 2.) * gammq(11. / 2., a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.)) +
                 a2 / pow(a4, 19. / 4.) * gammq(19. / 4., a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.)) + a3 / pow(a4, 4.) * (pow(a4, 3.) * crit_freq(radius, edens, etemp, mag_field) + 3 * pow(a4, 2.) * pow(crit_freq(radius, edens, etemp, mag_field), 2. / 3.) + 6 * a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.) + 6) * pow(C_euler, -a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.)));
        return result;
    }
    else
    {
        double result =
            2 * C_pi * BOLTZ_CGS * etemp * pow(crit_freq(radius, edens, etemp, mag_field), 3.) / (3 * scale_height(radius, edens, etemp) * pow(C_CGS, 2.)) +
            6.76 * pow(10., -28.) * edens / (2 * pow(thetae(etemp), 2.) * pow(a1, 1. / 6.)) *
                (1 / pow(a4, 11. / 2.) * gammq(11. / 2., a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.)) +
                 a2 / pow(a4, 19. / 4.) * gammq(19. / 4., a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.)) + a3 / pow(a4, 4.) * (pow(a4, 3.) * crit_freq(radius, edens, etemp, mag_field) + 3 * pow(a4, 2.) * pow(crit_freq(radius, edens, etemp, mag_field), 2. / 3.) + 6 * a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.) + 6) * pow(C_euler, -a4 * pow(crit_freq(radius, edens, etemp, mag_field), 1. / 3.)));
        return result;
    }
}

double comptonization_factor (double radius, double edens, double etemp, double mag_field){
	double thompson_opticaldepth = 2 * edens * THOMSON_CGS * etemp;
	double Afactor = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
	double maxfactor = 3 * BOLTZ_CGS * etemp/(PLANCK_CGS * crit_freq(radius, edens, etemp, mag_field));
	double jm = log(maxfactor)/log(Afactor);
	double s = thompson_opticaldepth + pow(thompson_opticaldepth, 2.);
	// printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
	// printf("O valor de Afactor é%le\n", Afactor);
	// printf("O valor de maxfactor é%le\n", maxfactor);
    // printf("O valor da critfreq é%le\n", crit_freq(radius, edens, etemp, mag_field));
	// printf("O valor de jm é%le\n", jm);
	// printf("O valor de s é%le\n", s);
	// printf("O valor de gammp(As) é%le\n", gammp(jm - 1, Afactor * s));
	// printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));

	double result = pow(C_euler, s*(Afactor -1))*(1 - gammp(jm - 1, Afactor * s)) + maxfactor * gammp(jm +1, s);
	if (isnan(result)){
	result = maxfactor * gammp(jm +1, s);
}
	/*printf("O valor de resultado é%le\n", result);*/

	return result;

}

/*scattering optical depth*/

double soptical_depth(double radius, double edens, double etemp)
{
    double result = 2. * edens * THOMSON_CGS * scale_height(radius, edens, etemp);
    return result;
}

// double comptonization_factor_artur(double edens, double etemp, double mag_field)
// {
//     double prob = 1 - pow(C_euler, -soptical_depth(edens, etemp));
//     double A = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
//     double result = 1 + prob * (A - 1) / (1 - prob * A) * (1 - pow((PLANCK_CGS * crit_freq(radius, edens, etemp, mag_field) / (3 * thetae(etemp) * ERM_CGS * pow(C_CGS, 2.))), -1 - log(prob) / log(A)));
//     return result;
// }

double totalthincooling_rate(double radius, double edens, double etemp, double mag_field)
{
    double result =
        bremmstrahlung_ee(edens, etemp) + bremmstrahlung_ei(edens, etemp) + rsync(radius, edens, etemp, mag_field) * comptonization_factor(radius, edens, etemp, mag_field);
    return result;
}

/*Absorption optical depth*/
double absoptical_depth(double radius, double edens, double etemp, double mag_field)
{
    double result = 1. / (4. * C_sigma * pow(etemp, 4.)) * scale_height(radius, edens, etemp) * totalthincooling_rate(radius, edens, etemp, mag_field);
    return result;
}

/*Total optical depth*/
double total_optical_depth(double radius, double edens, double etemp, double mag_field)
{
    double result = soptical_depth(radius, edens, etemp) + absoptical_depth(radius, edens, etemp, mag_field);
    return result;
}

/*Total cooling with thin and thick disk*/
double total_cooling(double radius, double edens, double etemp, double mag_field)
{
    return 4. * C_sigma * pow(etemp, 4.) / scale_height(radius, edens, etemp) * 1 /
           (3 * total_optical_depth(radius, edens, etemp, mag_field) / 2. + pow(3., 1. / 2.) + 1. / absoptical_depth(radius, edens, etemp, mag_field));
}

int main()
{

    double r;
    double ne;
    double te;
    double B;

    //float loop = 100;
    //char str[1];
    // while (loop > 1)
    // {
    //     printf("valor de radius\n");
    //     scanf("%le", &r);
    //     r = pow(10, r);
    //     printf("Valor de edens\n");
    //     scanf("%le", &ne);
    //     ne = pow(10, ne);
    //     printf ("valor de etemp\n");
    //     scanf("%le", &te);
    //     te= pow(10, te);
    //     printf ("valor do B\n");
    //     scanf("%le", &B);
    //     B = pow(10, B);
    //     printf("R = %le, ne = %le, Te = %le\n", r, ne, te);
    //     printf("\nOne of the roots is: %lf\n",secant(r, ne, te, B));
    //     printf("o valor do thetae é:%le\n", thetae(te));
    //     printf("o valor do H é:%le\n", scale_height(r, ne, te));
    //     printf("O valor do bremmstrahlung cooling rate é:%le\n", bremmscooling_rate(ne, te));
    //     printf("o valor da freq crit é: %le\n", crit_freq(r, ne, te, B));
    //     printf("o valor do rsync é: %le\n", rsync(r, ne, te, B));
    //     printf("o valor do comptonization factor é: %le\n", comptonization_factor(r, ne, te, B));
    //     printf("o valor do cooling total no disco fino é:%le\n", totalthincooling_rate(r, ne, te, B));
    //     printf("O valor do tau_scat é:%le\n", soptical_depth(r, ne, te));
    //     printf("O valor do tau_abs é:%le\n", absoptical_depth(r, ne, te, B));
    //     printf("O valor do tau_total é:%le\n", total_optical_depth(r, ne, te, B));
    //     printf("o valor do cooling total é:%le\n", total_cooling(r, ne, te, B));
    //     sleep(1);
    //     printf("Do you want to read other values? y/n\n");
    //     scanf("%s", str);
    //     if (strcmp(str, "n") == 0)
    //     {
    //         loop = 0;
    //     }
    // }
    FILE *file_radius;
    file_radius = fopen("radius.txt", "r");
    FILE *file_e_density;
    file_e_density = fopen("ne.txt", "r");
    FILE *file_temperature;
    file_temperature = fopen("te.txt", "r");
    FILE *file_mag_field;
    file_mag_field = fopen("mag.txt", "r");
    FILE *file_result;
    file_result = fopen("cooling_table_new.txt", "w");

    float cooling;

    if (file_e_density == NULL || file_temperature == NULL || file_mag_field == NULL)
    {
        printf("Error Reading File\n");
        exit(0);
    }

    fprintf(file_result, "radius, mag_field, e_density, temperature, cooling\n");
    // run for logarithm values
    while (fscanf(file_radius, "%lf", &r) == 1){
        rewind(file_mag_field);
        while (fscanf(file_mag_field, "%lf,", &B) == 1)
        {
            rewind(file_e_density);
            while (fscanf(file_e_density, "%lf,", &ne) == 1)
            {
                rewind(file_temperature);
                while (fscanf(file_temperature, "%lf,", &te) == 1)
                {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f,", r, B, ne, te);
                        r = pow(10., r);
                        ne = pow(10., ne);
                        te = pow(10., te);
                        B = pow(10., B);
                        cooling = log10(total_cooling(r, ne, te, B));
                        r = log10(r);
                        ne = log10(ne);
                        te = log10(te);
                        B = log10(B);
                        fprintf(file_result, "%.2f\n", cooling);
                }
            }
        }
    }
    fclose(file_radius);
    fclose(file_e_density);
    fclose(file_temperature);
    fclose(file_result);
    fclose(file_mag_field);

    return 0;
}
