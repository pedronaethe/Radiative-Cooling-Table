// Cooling Functions including Bremmstrahlung, Synchrotron and Comptonized Synchrotron
// Code by Pedro Naethe Motta
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define C_pi 3.14159265358979    /*value of pi*/
#define C_euler 2.71828182845904 /* euler constant*/
#define ERM_CGS 9.1093826e-28    /*Electron rest mass CGS*/
#define C_sigma 5.67051e-5       /*Stephan Boltzmann constant*/
#define BOLTZ_CGS 1.3806505e-16  /*Boltzmann constant*/
#define eps 3e-7
#define maxSteps 1e3
#define BOLTZ_CGS 1.3806505e-16   /*Boltzmann constant*/
#define PLANCK_CGS 6.62606876e-27 /*Planck constant*/
#define ERM_CGS 9.1093826e-28     /*Electron rest mass CGS*/
#define C_amu 1.66053886e-24      /*Atomic mass unit*/
#define C_CGS 2.99792458e10       /*Speed of light*/
#define C_G 6.6726e-8             /*Gravitational constant*/
#define C_Msun 2.e33              /*Sun mass*/
#define THOMSON_CGS 6.6524e-25    /*Thomson cross section*/
#define Rs 2. * C_G *C_Mbh / (pow(C_CGS, 2.))
#define beta 10.
#define C_gamma (5. / 3.)

/*Mass of the Black hole*/
#define C_Mbh 10. * C_Msun
#define C_GM C_Mbh *C_G

/*Disk geometry and polytropic constant*/
#define rho_0 5e-7     /*Maximum density of initial condition*/
#define r_0 100. * Rs  /*Radius of maximum density (rho_0)*/
#define r_min 75. * Rs /*Minimum raius of torus*/
#define CONST_2 -C_GM / (r_min - Rs) + C_GM / (2. * pow(r_min, 2.)) * (pow(r_0, 3.) / (pow((r_0 - Rs), 2)))
#define kappa (C_gamma - 1.) / (C_gamma)*pow(rho_0, (1. - C_gamma)) * (CONST_2 + C_GM / (r_0 - Rs) - C_GM / 2. * r_0 / (pow(r_0 - Rs, 2.)))

#define ITMAX 100     /* Used in calculating gamma function*/
#define EPS 3.0e-7    /* Used in calculating gamma function*/
#define FPMIN 1.0e-30 /* Used in calculating gamma function*/

double ne;
double Te;
double R;

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
        if (fabs(del - 1.0) < EPS)
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
            if (fabs(del) < fabs(sum) * EPS)
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
double thetae(double Te)
{
    double result = (BOLTZ_CGS)*Te / ((ERM_CGS) * (pow(C_CGS, 2.)));
    return result;
}

double sound_speed(double ne)
{
    double result = pow((C_gamma) * (kappa)*pow(ne * 1.14 * C_amu, (C_gamma - 1.)), 1. / 2.);
    return result;
}

double Bmag(double ne)
{
    // printf("ne = %le\n", ne);
    double result = pow(8. * C_pi * pow(sound_speed(ne), 2.) * ne * 1.14 * C_amu / (beta + 1.), 1. / 2.);
    return result;
}

double scale_height(double R, double ne, double Te)
{
    //double result = pow(R / C_GM, 1. / 2.) * sound_speed(ne) * (R - Rs);
	//double result = sound_speed(ne)/((C_G * C_Mbh/pow(R,3.)));
	double result = Te/(4/C_euler);
	//printf("Scale height =%le, R=%le\n", result, R);
    return result;
}

double f(double x, double R, double ne, double Te)
{
    if (thetae(Te) < 0.03)
    {
        return pow(C_euler, 1.8899 * pow(x, 1. / 3.)) -
               2.49 * pow(10., -10.) * 12. * C_pi * ne * R / (Bmag(ne)) * 1 / (2 * pow(thetae(Te), 5.)) *
                   (1 / pow(x, 7. / 6.) + 0.4 / pow(x, 17. / 12.) + 0.5316 / pow(x, 5. / 3.));
    }
    else
    {
        return pow(C_euler, 1.8899 * pow(x, 1. / 3.)) - 2.49 * pow(10., -10.) * 12. * C_pi * ne * R / (Bmag(ne)) * 1 /
                                                            (pow(thetae(Te), 3.) * bessk2(1 / thetae(Te))) *
                                                            (1 / pow(x, 7. / 6.) + 0.4 / pow(x, 17. / 12.) +
                                                             0.5316 / pow(x, 5. / 3.));
    }
}

/*Function that returns the root from Secant Method*/
double secant(double f(double x, double R, double ne, double Te))
{
    int iter = 1;
    double x1 = 1.;
    double x2 = 20000.;
    double x3;
    do
    {
        x3 = (x1 * f(x2, R, ne, Te) - x2 * f(x1, R, ne, Te)) / (f(x2, R, ne, Te) - f(x1, R, ne, Te));
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
    } while (fabs(f(x3, R, ne, Te)) > eps && iter <= maxSteps);
    return x1;
}

double bremmstrahlung_ee(double ne, double Te)
{
    double th_e = thetae(Te);
    double result;
    if (th_e < 1)
    {
        result = 2.56 * pow(10., -22.) * pow(ne, 2.) * pow(thetae(Te), 3. / 2.) *
                 (1 + 1.10 * thetae(Te) + pow(thetae(Te), 2.) - 1.25 * pow(thetae(Te), 5. / 2.));
    }
    else
    {
        result = 3.40 * pow(10., -22.) * pow(ne, 2.) * thetae(Te) * (log(1.123 * thetae(Te)) + 1.28);
    }
    return result;
}

double bremmstrahlung_ei(double ne, double Te)
{
    double th_e = thetae(Te);
    double result;
    if (th_e > 1)
    {
        result = 1.48 * pow(10., -22.) * pow(ne, 2.) * (9 * thetae(Te)) / (2 * C_pi) *
                 (log(1.123 * thetae(Te) + 0.48) + 1.5);
    }
    else
    {
        result = 1.48 * pow(10., -22.) * pow(ne, 2.) * 4 * pow((2 * thetae(Te) / pow(C_pi, 3.)), 1. / 2.) *
                 (1 + 1.781 * pow(thetae(Te), 1.34));
    }
    return result;
}

double bremmscooling_rate(double ne, double Te)
{
    double result = bremmstrahlung_ee(ne, Te) + bremmstrahlung_ei(ne, Te);
    return result;
}

double crit_freq(double ne, double Te)
{
    return (1.5) * (2.80e6 * Bmag(ne)) * pow(thetae(Te), 2.) * secant(f);
}

/*Synchtron radiation calculation*/
double rsync(double ne, double Te)
{
    double bsq = pow(Bmag(ne), 2.);
    double nuzero = 2.80 * pow(10., 6.) * pow(bsq, 1. / 2.);
    double a1 = 2 / (3 * nuzero * pow(thetae(Te), 2.));
    double a2 = 0.4 / pow(a1, 1. / 4.);
    double a3 = 0.5316 / pow(a1, 1. / 2.);
    double a4 = 1.8899 * pow(a1, 1. / 3.);
    if (thetae(Te) > 0.03)
    {
        double result =
            2 * C_pi * BOLTZ_CGS * Te * pow(crit_freq(ne, Te), 3.) / (3 * scale_height(R, ne, Te) * pow(C_CGS, 2.)) +
            6.76 * pow(10., -28.) * ne / (bessk2(1 / thetae(Te)) * pow(a1, 1. / 6.)) *
                (1 / pow(a4, 11. / 2.) * gammq(11. / 2., a4 * pow(crit_freq(ne, Te), 1. / 3.)) +
                 a2 / pow(a4, 19. / 4.) * gammq(19. / 4., a4 * pow(crit_freq(ne, Te), 1. / 3.)) + a3 / pow(a4, 4.) * (pow(a4, 3.) * crit_freq(ne, Te) + 3 * pow(a4, 2.) * pow(crit_freq(ne, Te), 2. / 3.) + 6 * a4 * pow(crit_freq(ne, Te), 1. / 3.) + 6) * pow(C_euler, -a4 * pow(crit_freq(ne, Te), 1. / 3.)));
        return result;
    }
    else
    {
        double result =
            2 * C_pi * BOLTZ_CGS * Te * pow(crit_freq(ne, Te), 3.) / (3 * scale_height(R, ne, Te) * pow(C_CGS, 2.)) +
            6.76 * pow(10., -28.) * ne / (2 * pow(thetae(Te), 2.) * pow(a1, 1. / 6.)) *
                (1 / pow(a4, 11. / 2.) * gammq(11. / 2., a4 * pow(crit_freq(ne, Te), 1. / 3.)) +
                 a2 / pow(a4, 19. / 4.) * gammq(19. / 4., a4 * pow(crit_freq(ne, Te), 1. / 3.)) + a3 / pow(a4, 4.) * (pow(a4, 3.) * crit_freq(ne, Te) + 3 * pow(a4, 2.) * pow(crit_freq(ne, Te), 2. / 3.) + 6 * a4 * pow(crit_freq(ne, Te), 1. / 3.) + 6) * pow(C_euler, -a4 * pow(crit_freq(ne, Te), 1. / 3.)));
        return result;
    }
}

// double comptonization_factor (double ne, double Te){
//	double bsq = pow(Bmag(ne), 2.);
//	double thompson_opticaldepth = 2 * ne * THOMSON_CGS * Te;
//	double Afactor = 1 + 4 * thetae(Te) + 16 * pow(thetae(Te), 2.);
//	double maxfactor = 3 * BOLTZ_CGS * Te/(PLANCK_CGS * crit_freq(ne, Te));
//	double jm = log(maxfactor)/log(Afactor);
//	double s = thompson_opticaldepth + pow(thompson_opticaldepth, 2.);
//	/*printf("O valor de bsq é%le\n", bsq);
//	printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
//	printf("O valor de Afactor é%le\n", Afactor);
//	printf("O valor de nuzero é%le\n", nuzero);
//	printf("O valor de maxfactor é%le\n", maxfactor);
//	printf("O valor de jm é%le\n", jm);
//	printf("O valor de s é%le\n", s);
//	printf("O valor de gammp(As) é%le\n", gammp(jm - 1, Afactor * s));
//	printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));*/
//
//	double result = pow(C_euler, s*(Afactor -1))*(1 - gammp(jm - 1, Afactor * s)) + maxfactor * gammp(jm +1, s);
//	if (isnan(result)){
//	result = maxfactor * gammp(jm +1, s);
// }
//	/*printf("O valor de resultado é%le\n", result);*/
//
//	return result;
//
// }

/*scattering optical depth*/

double soptical_depth(double R, double ne)
{
    double result = 2. * ne * THOMSON_CGS * scale_height(R, ne, Te);
    return result;
}

double comptonization_factor_artur(double ne, double Te)
{
    double prob = 1 - pow(C_euler, -soptical_depth(R, ne));
    double A = 1 + 4 * thetae(Te) + 16 * pow(thetae(Te), 2.);
    double result = 1 + prob * (A - 1) / (1 - prob * A) * (1 - pow((PLANCK_CGS * crit_freq(ne, Te) / (3 * thetae(Te) * ERM_CGS * pow(C_CGS, 2.))), -1 - log(prob) / log(A)));
    return result;
}

double totalthincooling_rate(double ne, double Te)
{
    double result =
        bremmstrahlung_ee(ne, Te) + bremmstrahlung_ei(ne, Te) + rsync(ne, Te) * comptonization_factor_artur(ne, Te);
    return result;
}

/*Absorption optical depth*/
double absoptical_depth(double R, double ne, double Te)
{
    double result = 1. / (4. * C_sigma * pow(Te, 4.)) * scale_height(R, ne, Te) * totalthincooling_rate(ne, Te);
    return result;
}

/*Total optical depth*/
double total_optical_depth(double R, double ne, double Te)
{
    double result = soptical_depth(R, ne) + absoptical_depth(R, ne, Te);
    return result;
}

/*Total cooling with thin and thick disk*/
double total_cooling(double R, double ne, double Te)
{
    return 4. * C_sigma * pow(Te, 4.) / scale_height(R, ne, Te) * 1 /
           (3 * total_optical_depth(R, ne, Te) / 2. * pow(3., 1. / 2.) + 1. / absoptical_depth(R, ne, Te));
}

int main()
{
    //	printf("valor de R\n");
    //	scanf("%le", &R);
    //	printf("Valor de ne\n");
    //	scanf("%le", &ne);
    //	printf ("valor de Te\n");
    //	scanf("%le", &Te);
    //        printf("\nOne of the roots is: %lf\n",secant(f));
    //	printf("O valor do bremmstrahlung cooling rate é:%le\n", bremmscooling_rate(ne, Te));
    //	printf("o valor do thetae é:%le\n", thetae(Te));
    //	printf("O valor do sound_speed é:%lf\n", sound_speed(ne));
    //	printf("o valor de Bmag é: %le\n", Bmag(ne));
    //	printf("o valor do cooling total no disco fino é:%le\n", totalthincooling_rate(ne, Te));
    //	printf("o valor do rsync é: %le\n", rsync(ne, Te));
    //	printf("o valor do comptonization factor é: %le\n", comptonization_factor(ne, Te));
    //	printf("o valor do comptonization factor artur é: %le\n", comptonization_factor_artur(ne, Te));
    //	printf("o valor da freq crit é: %le\n", crit_freq(ne, Te));
    //	printf("O valor do tau_scat é:%le\n", soptical_depth(R, ne));
    //	printf("O valor do tau_abs é:%le\n", absoptical_depth(R, ne, Te));
    //	printf("O valor do tau_total é:%le\n", total_optical_depth(R, ne, Te));
    //	printf("o valor do cooling total é:%le\n", total_cooling(R, ne, Te));
    // printf("\ndobro: %lf\n", dobro(f));
    FILE *file_radius;
    file_radius = fopen("r.txt", "r");
    FILE *file_e_density;
    file_e_density = fopen("ne.txt", "r");
    FILE *file_temperature;
    file_temperature = fopen("te.txt", "r");
    FILE *file_result;
    file_result = fopen("cooling_table_log.txt", "w");

    float cooling;
    int a = 0, b = 0, c = 0;

    if (file_radius == NULL || file_e_density == NULL || file_temperature == NULL)
    {
        printf("Error Reading File\n");
        exit(0);
    }

    fprintf(file_result, "radius, e_density, temperature, cooling\n");
    // run for logarithm values
    while (fscanf(file_radius, "%lf,", &R) == 1)
    {
        rewind(file_e_density);
        while (fscanf(file_e_density, "%lf,", &ne) == 1)
        {
            rewind(file_temperature);
            while (fscanf(file_temperature, "%lf,", &Te) == 1)
            {
                fprintf(file_result, "%.2f, %.2f, %.2f, ", R, ne, Te);
                R = pow(10., R);
                ne = pow(10., ne);
                Te = pow(10., Te);
                cooling = log10(total_cooling(R, ne, Te));
                R = log10(R);
                ne = log10(ne);
                Te = log10(Te);
                fprintf(file_result, "%.2f\n", cooling);
            }
        }
    }

    // Run for values in the natural scale (not log scale)
    // while(fscanf(file_radius, "%lf,", &R) == 1) {
    //     rewind(file_e_density);
    //     while(fscanf(file_e_density, "%lf,", &ne) == 1) {
    //         rewind(file_temperature);
    //         while(fscanf(file_temperature, "%lf,", &Te) == 1) {
    //             cooling = total_cooling(R, ne, Te);
    //             fprintf(file_result,"%lf, %lf, %lf, %lf\n", R, ne, Te, cooling);
    //         }
    //     }
    // }

    fclose(file_radius);
    fclose(file_e_density);
    fclose(file_temperature);
    fclose(file_result);

    return 0;
}
