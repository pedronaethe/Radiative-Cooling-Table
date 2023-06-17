// Cooling Functions including Bremmstrahlung, Synchrotron and Comptonized Synchrotron
// Code by Pedro Naethe Motta
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define MH_CGS (1.673534e-24) /*hydrogen mass*/
#define C_sigma (5.67051e-5)       /*Stephan Boltzmann constant*/
#define C_PI (3.141592653589793)       /*Pi*/
#define BOLTZ_CGS (1.3806504e-16)  /*Boltzmann constant*/
#define PLANCK_CGS (6.6260755e-27) /*Planck constant*/
#define ERM_CGS (9.10938215e-28)     /*Electron rest mass CGS*/
#define C_CGS (2.99792458e10)       /*Speed of light*/
#define C_G (6.67259e-8)             /*Gravitational constant*/
#define C_Msun (1.998e33)              /*Sun mass*/
#define THOMSON_CGS (6.652e-25)    /*Thomson cross section*/
#define ME_CGS (9.1094e-28) /*Mass hydrogen molecule*/

/*battery of tests, for normal table generation, just put everything equals to 0*/
#define BLACKBODYTEST (0) //Generates a table for the blackbody equation
#define SYNCHROTRONTEST (0) //Generates a table for synchrotron equation
#define C_SYNCHROTRONTEST (0)// Generates a table for comptonized synchrotron equation
#define COMPTONTEST (0) // Generates a table with the compton values
#define BREMSTRAHLUNGTEST (0) //Generates a table for bremmstrahlung equation
#define ABSORPTIONDEPTHTEST (0) //Generates a table with absorption values
#define RECALCULATE_GRID_TEST (1)
    #define COULOMB_RECALCULATE_GRID (0)
#define SINGLE_VALUE (0) // Individual value of every function for a certain quantity of parameters]
    #define COULOMB_TEST (0)
#define COMPARISON_MARCEL (0) //Compare plot A.1 of Marcel et al. 2018: A unified accretion-ejection paradigm for black hole X-ray binaries

#define PARAMETER_H 101
#define PARAMETER_B 101
#define PARAMETER_NE 101
#define PARAMETER_TE 101

/*Mass of the Black hole*/
#define C_Mbh (10. * C_Msun)
#define C_GM (C_Mbh * C_G)

#define R_CGS (C_G * C_Mbh/(C_CGS**2))


#define ITMAX (1e8)     /* Used in calculating gamma function*/
#define EPS (3e-7)
#define maxSteps (1e3)
#define FPMIN (1.0e-30) /* Used in calculating gamma function*/


double gammln(double xxgam);

/***********************************************************************************************/
void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr,"%s\n",error_text);
    fprintf(stderr,"...now exiting to system...\n");
    exit(1);
}
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

// void gcf(double *gammcf, double agam, double xgam,
//          double *gln)
// { // Function used in the calculation of incomplete gamma function
//     int i;
//     double an, b, c, d, del, h;
//     *gln = gammln(agam);
//     b = xgam + 1.0 - agam;
//     c = 1.0 / FPMIN;
//     d = 1.0 / b;
//     h = d;
//     for (i = 1; i <= ITMAX; i++)
//     {
//         an = -i * (i - agam);
//         b += 2.0;
//         d = an * d + b;
//         if (fabs(d) < FPMIN)
//             d = FPMIN;
//         c = b + an / c;
//         if (fabs(c) < FPMIN)
//             c = FPMIN;
//         d = 1.0 / d;
//         del = d * c;
//         h *= del;
//         if (fabs(del - 1.0) < eps)
//             break;
//     }
//     *gammcf = exp(-xgam + agam * log(xgam) - (*gln)) * h;
// }
void gcf(double *gammcf, double a, double x, double *gln)
/*Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction representation as gammcf. Also returns ln Γ(a) as gln.*/
{
    double gammln(double xx);
    void nrerror(char error_text[]);
    int i;
    double an,b,c,d,del,h;
    *gln=gammln(a);
    b=x+1.0-a; /*Set up for evaluating continued fraction by modified Lentz’s method (§5.2) with b0 = 0.*/
    c=1.0/FPMIN;
    d=1.0/b;
    h=d;
    for (i=1;i<=ITMAX;i++) 
    {//Iterate to convergence.
        an = -i*(i-a);
        b += 2.0;
        d=an*d+b;
        if (fabs(d) < FPMIN) d=FPMIN;
            c=b+an/c;
        if (fabs(c) < FPMIN) c=FPMIN;
            d=1.0/d;
            del=d*c;
            h *= del;
        if (fabs(del-1.0) < EPS) break;
    }
    if (i > ITMAX) nrerror("a too large, ITMAX too small in gcf");
        *gammcf=exp(-x+a*log(x)-(*gln))*h; //Put factors in front.
}

// void gser(double *gamser, double agam, double xgam,
//           double *gln)
// { // Function used in the calculation of incomplete gamma function
//     int n;
//     double sum, del, ap;

//     *gln = gammln(agam);
//     if (xgam <= 0.0)
//     {
//         if (xgam < 0.0)
//             ;
//         *gamser = 0.0;
//         return;
//     }
//     else
//     {
//         ap = agam;
//         del = sum = 1.0 / agam;
//         for (n = 1; n <= ITMAX; n++)
//         {
//             ++ap;
//             del *= xgam / ap;
//             sum += del;
//             if (fabs(del) < fabs(sum) * eps)
//             {
//                 *gamser = sum * exp(-xgam + agam * log(xgam) - (*gln));
//                 return;
//             }
//         }
//         return;
//     }
// }
void gser(double *gamser, double a, double x, double *gln)
/*Returns the incomplete gamma function P(a, x) evaluated by its series representation as gamser.
Also returns ln Γ(a) as gln.*/
{
    double gammln(double xx);
    void nrerror(char error_text[]);
    int n;
    double sum,del,ap;
    *gln=gammln(a);
    if (x <= 0.0) {
        if (x < 0.0) nrerror("x less than 0 in routine gser");
        *gamser=0.0;
        return;
    } else {
        ap=a;
        del=sum=1.0/a;
        for (n=1;n<=ITMAX;n++) {
            ++ap;
            del *= x/ap;
            sum += del;
            if (fabs(del) < fabs(sum)*EPS) 
            {
                *gamser=sum*exp(-x+a*log(x)-(*gln));
                return;
            }
        }
        nrerror("a too large, ITMAX too small in routine gser");
        return;
    }
}
// double gammp(double agam, double xgam)
// {
//     int n;
//     if (agam < 0)
//     {
//         agam = -agam;
//         if (n < 1)
//         {
//             printf("Valor negativo para a na função gammp!!! a= %le\n", -agam);
//             n = 2;
//         }
//     }
//     double gamser, gammcf, gln;
//     if (xgam < (agam + 1.0))
//     {
//         gser(&gamser, agam, xgam, &gln);
//         return gamser;
//     }
//     else
//     {
//         gcf(&gammcf, agam, xgam, &gln);
//         return 1.0 - gammcf;
//     }
// }


double gammp(double a, double x)
//Returns the incomplete gamma function P(a, x).
{
    void gcf(double *gammcf, double a, double x, double *gln);
    void gser(double *gamser, double a, double x, double *gln);
    void nrerror(char error_text[]);
    double gamser,gammcf,gln;
    if (x < 0.0 || a <= 0.0) nrerror("Invalid arguments in routine gammp");
    if (x < (a+1.0)) 
    { //Use the series representation.
        gser(&gamser,a,x,&gln);
        return gamser;
    } else { //Use the continued fraction representation
        gcf(&gammcf,a,x,&gln);
        return 1.0-gammcf; //and take its complement.
    }
}

double gammq(double a, double x)
//Returns the incomplete gamma function Q(a, x) ≡ 1 − P(a, x).
{
    void gcf(double *gammcf, double a, double x, double *gln);
    void gser(double *gamser, double a, double x, double *gln);
    void nrerror(char error_text[]);
    double gamser,gammcf,gln;
    if (x < 0.0 || a <= 0.0) nrerror("Invalid arguments in routine gammq");
    if (x < (a+1.0)) 
    { //Use the series representation
        gser(&gamser,a,x,&gln);
        return 1.0-gamser; //and take its complement.
    } else { //Use the continued fraction representation.
        gcf(&gammcf,a,x,&gln);
        return gammcf;
    }
}

double gammln(double xx)
//Returns the value ln[Γ(xx)] for xx > 0.
{
    /*Internal arithmetic will be done in double precision, a nicety that you can omit if five-figure
    accuracy is good enough.*/
    double x,y,tmp,ser;
    static double cof[6]={76.18009172947146,-86.50532032941677,
    24.01409824083091,-1.231739572450155,
    0.1208650973866179e-2,-0.5395239384953e-5};
    int j;
    y=x=xx;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.000000000190015;
    for (j=0;j<=5;j++) ser += cof[j]/++y;
    return -tmp+log(2.5066282746310005*ser/x);
}
// double gammln(double xxgam)
// {
//     double x, y, tmp, ser;
//     static double cof[6] = {76.18009172947146, -86.50532032941677,
//                             24.01409824083091, -1.231739572450155,
//                             0.1208650973866179e-2, -0.5395239384953e-5};
//     int j;
//     y = x = xxgam;
//     tmp = x + 5.5;
//     tmp -= (x + 0.5) * log(tmp);
//     ser = 1.000000000190015;
//     for (j = 0; j <= 5; j++)
//         ser += cof[j] / ++y;
//     return -tmp + log(2.5066282746310005 * ser / x);
// }

/***********************************************************************************************/
double thetae(double etemp)
{
    double result = (BOLTZ_CGS)* etemp / ((ERM_CGS) * (pow(C_CGS, 2.)));
    return result;
}

double f(double scale_height, double x, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    if (thetae(etemp) < 0.01)
    { 
        double pre_factor = 2.49 * pow(10., -10.) * 12. * C_PI * edens * scale_height / (mag_field) * 1 /(2 * pow(thetae(etemp), 5.)); 
        //printf("Prefactor1 = %le \n", pre_factor);
        return exp(1.8899 * pow(x, 1. / 3.)) - pre_factor * (1 / pow(x, 7. / 6.) + 0.4 / pow(x, 17. / 12.) + 0.5316 / pow(x, 5. / 3.));
    }
    else
    {
        double pre_factor = 2.49 * pow(10., -10.) * 12. * C_PI * edens * scale_height / (mag_field) * 1 /(bessk2(1. / thetae(etemp)) * pow(thetae(etemp), 3.));
        //printf("Prefactor2 = %le \n", pre_factor);
        return exp( 1.8899 * pow(x, 1. / 3.)) - pre_factor * (1. / pow(x, 7. / 6.) + 0.4 / pow(x, 17. / 12.) + 0.5316 / pow(x, 5. / 3.));
    }
}

double secant_bounded(double (*func)(double, double, double, double, double), double scale_height, double edens, double etemp, double mag_field)
//Using the secant method, find the root of a function func thought to lie between x1 and x2.
//The root, returned as rtsec, is refined until its accuracy is ±xacc.
{//All the function was checked step by step, seems to be working good.
    void nrerror(char error_text[]);
    int j;
    double xacc = 1e-6;
    double x1 = 10;
    double x2 = 20;
    double fl,f,dx,swap,xl,rts;
    int interval = 1;
    fl=(*func)(scale_height, x1, edens, etemp, mag_field);
    f=(*func)(scale_height, x2, edens, etemp, mag_field);
    while(interval){
        if (f * fl > 0){
            x1 = x1/2;
            x2 = x2*2;
            //printf("x1 = %le, x2 = %le \n", x1, x2);
        }
        else{
            interval = 0;
            //printf("Finally x1 = %le, x2 = %le \n", x1, x2);
        }
        fl=(*func)(scale_height, x1, edens, etemp, mag_field);
        f=(*func)(scale_height, x2, edens, etemp, mag_field);
        //printf("fl = %le, f = %le\n", fl, f);
    }
    if (fabs(fl) < fabs(f)) { //Pick the bound with the smaller function value as
        rts=x1; //the most recent guess.
        xl=x2;
        swap=fl;
        fl=f;
        f=swap;
    } else {
        xl=x1;
        rts=x2;
    }
    for (j=1;j<=ITMAX;j++) { //Secant loop.
        dx=(xl-rts)*f/(f-fl); //Increment with respect to latest value.
        xl=rts;
        fl=f;
        rts += dx;
        f=(*func)(scale_height, rts, edens, etemp, mag_field);
        //printf("rts = %le \n", rts);
        if (fabs(dx) < xacc || f == 0.0) {
            return rts; //Convergence.
        }
    }
    printf("Error! H = %le, B = %le, ne = %le, Te = %le\n", scale_height, mag_field, edens, etemp);
    nrerror("Maximum number of iterations exceeded in rtsec");
    return 0.0; //Never get here.
}




double bremmstrahlung_ee(double edens, double etemp)
{//All the function was checked step by step, seems to be working good.
    double result;
    if (thetae(etemp) < 1)
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
{//All the function was checked step by step, seems to be working good.
    double result;
    if (thetae(etemp) >= 1)
    { // functioning
        result = 1.48 * pow(10., -22.) * pow(edens, 2.) * (9. * thetae(etemp)) / (2. * C_PI) *
                 (log(1.123 * thetae(etemp) + 0.48) + 1.5);
    }
    else
    {   //functioning
        result = 1.48 * pow(10., -22.) * pow(edens, 2.) * 4. * pow((2 * thetae(etemp) / pow(C_PI, 3.)), 1. / 2.) *
                 (1. + 1.781 * pow(thetae(etemp), 1.34));
    }
    return result;
}

double bremmscooling_rate(double edens, double etemp)
{//All the function was checked step by step, seems to be working good.

    double result = bremmstrahlung_ee(edens, etemp) + bremmstrahlung_ei(edens, etemp);
    return result;
}

double crit_freq(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    return (1.5) * (2.79925 * pow(10.,6.) * mag_field) * pow(thetae(etemp), 2.) * secant_bounded(f,scale_height, edens, etemp, mag_field);
}

/*Synchtron radiation calculation*/
double rsync(double scale_height, double edens, double etemp, double mag_field)
{ //All the function was checked step by step, seems to be working good.
    double nuzero = 2.80 * pow(10., 6.) * mag_field;
    //correcto
    double a1 = 2. / (3. * nuzero * pow(thetae(etemp), 2.));
    double a2 = 0.4 / pow(a1, 1. / 4.);
    double a3 = 0.5316 / pow(a1, 1. / 2.);
    double a4 = 1.8899 * pow(a1, 1. / 3.);
    if (thetae(etemp) > 0.01)
    {
        double self_abs = 2. * C_PI * BOLTZ_CGS * etemp * pow(crit_freq(scale_height, edens, etemp, mag_field), 3.) / (3. * scale_height * pow(C_CGS, 2.));
        double init_term2 =6.76 * pow(10., -28.) * edens / (bessk2(1. / thetae(etemp)) * pow(a1, 1. / 6.));
        double syn_1 = 1. / pow(a4, 11. / 2.) *(exp(gammln(11./2.)))* gammq(11. / 2., a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.));
        double syn_2 = a2 / pow(a4, 19. / 4.) *(exp(gammln(19./4.)))* gammq(19. / 4., a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.));
        double syn_3 = a3 / pow(a4, 4.) * (pow(a4, 3.) * crit_freq(scale_height, edens, etemp, mag_field) + 3. * pow(a4, 2.) * pow(crit_freq(scale_height, edens, etemp, mag_field), 2. / 3.) + 6. * a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.) + 6.) * exp(-a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.));
        double synchrotron = init_term2 * (syn_1+syn_2+syn_3);
        double result = self_abs + synchrotron;
        
        return result;
    }
    else
    {
        double self_abs = 2. * C_PI * BOLTZ_CGS * etemp * pow(crit_freq(scale_height, edens, etemp, mag_field), 3.) / (3. * scale_height * pow(C_CGS, 2.));
        double init_term2 = 6.76 * pow(10., -28.) * edens / (2. * pow(thetae(etemp), 2.) * pow(a1, 1. / 6.));
        double syn_1 = 1. / pow(a4, 11. / 2.) *(exp(gammln(11./2.)))* gammq(11. / 2., a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.));
        double syn_2 = a2 / pow(a4, 19. / 4.) *(exp(gammln(19./4.)))* gammq(19. / 4., a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.));
        double syn_3 = a3 / pow(a4, 4.) * (pow(a4, 3.) * crit_freq(scale_height, edens, etemp, mag_field) + 3. * pow(a4, 2.) * pow(crit_freq(scale_height, edens, etemp, mag_field), 2. / 3.) + 6. * a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.) + 6.) * exp(-a4 * pow(crit_freq(scale_height, edens, etemp, mag_field), 1. / 3.));
        double synchrotron = init_term2 * (syn_1+syn_2+syn_3);
        double result = self_abs + synchrotron;
        return result;
    }
}


/*scattering optical depth*/

double soptical_depth(double scale_height, double edens, double etemp)
{//All the function was checked step by step, seems to be working good.
    double result =  2. * edens * THOMSON_CGS * scale_height;
    return result;
}

double comptonization_factor_ny(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    double prob = 1 - exp(-soptical_depth(scale_height, edens, etemp));
    double A = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
    double eta1 = prob * (A - 1) / (1 - prob * A);
    double eta2 = PLANCK_CGS * crit_freq(scale_height, edens, etemp, mag_field)/(3 * thetae(etemp) * ERM_CGS * pow(C_CGS,2.));
    double eta3 = -1 - log(prob)/log(A);
    double result = 1 + eta1* (1 - pow(eta2, eta3));
    if (eta2 > 1){
        //printf("Compton formula not valid, exiting...");
        //exit(1);
    }
    return result;
}

double comptonization_factor(double scale_height, double edens, double etemp, double mag_field){
	double thompson_opticaldepth = 2 * edens * THOMSON_CGS * etemp;
	double Afactor = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
	double maxfactor = 3 * BOLTZ_CGS * etemp/(PLANCK_CGS * crit_freq(scale_height, edens, etemp, mag_field));
	double jm = log(maxfactor)/log(Afactor);
	double s = thompson_opticaldepth + pow(thompson_opticaldepth, 2.);
    double factor_1 = (1 - gammp(jm + 1, Afactor * s));
    double factor_2 = maxfactor * gammp(jm +1, s);
    if (factor_1 == 0){
        double result = factor_2;
        if (isnan(result)){
            printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
            printf("O valor de Afactor é%le\n", Afactor);
            printf("O valor de maxfactor é%le\n", maxfactor);
            printf("O valor de jm é%le\n", jm);
            printf("O valor de s é%le\n", s);
            printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
            printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
            exit(1);
        }
        return result;
    }
    else{
        double result = pow(M_E, s*(Afactor -1))*factor_1 + factor_2;
        if (isnan(result)){
            printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
            printf("O valor de Afactor é%le\n", Afactor);
            printf("O valor de maxfactor é%le\n", maxfactor);
            printf("O valor de jm é%le\n", jm);
            printf("O valor de s é%le\n", s);
            printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
            printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
            exit(1);
        }
        return result;
    }
}
double totalthincooling_rate(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    double result = bremmscooling_rate(edens, etemp) + rsync(scale_height, edens, etemp, mag_field) * comptonization_factor(scale_height, edens, etemp, mag_field);
    return result;
}

/*Absorption optical depth*/
double absoptical_depth(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    double result = 1. / (4. * C_sigma * pow(etemp, 4.)) * scale_height * totalthincooling_rate(scale_height, edens, etemp, mag_field);
    return result;
}

/*Total optical depth*/
double total_optical_depth(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    double result = soptical_depth(scale_height, edens, etemp) + absoptical_depth(scale_height, edens, etemp, mag_field);
    return result;
}

/*Total cooling with thin and thick disk*/
double total_cooling(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    return 4. * C_sigma * pow(etemp, 4.) / scale_height * 1 /
           (3 * total_optical_depth(scale_height, edens, etemp, mag_field) / 2. + pow(3., 1. / 2.) + 1. / absoptical_depth(scale_height, edens, etemp, mag_field));
}

double bbody(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    return 8. * C_sigma * pow(etemp, 4.) / (3 * scale_height * total_optical_depth(scale_height, edens, etemp, mag_field));
}

void logspace(double start, double end, int num, double* result) {
    double log_start = log10(start); //Initial value
    double log_end = log10(end); //End value
    double step = (log_end - log_start) / (num - 1); // number of steps
    int i;
    for (i = 0; i < num; ++i) {
        result[i] = pow(10.0, log_start + i * step);
    }
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
    double Theta_e = thetae(etemp); 
	coeff = 1.5 * ME_CGS / MH_CGS * coulog * C_CGS * BOLTZ_CGS * THOMSON_CGS;
    coeff *= pow(edens, 2.) * (itemp - etemp);
    //printf("coef = %.2e \n", coeff);
    th_sum = thetae(etemp) + theta_i(itemp);
	th_mean = thetae(etemp) * theta_i(itemp) / (thetae(etemp) + theta_i(itemp));
    if (Theta_i < theta_min && Theta_e < theta_min) // approximated equations at small theta
	{
		result = coeff / sqrt(0.5 * C_PI * th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
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
		K2e = bessk2(1. / Theta_e);
		result = coeff / K2e / exp(1. / Theta_e) * sqrt(Theta_e) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
    }
	else if (Theta_e < theta_min)
	{
		//bessel function
		K2i = bessk2(1. / Theta_i);
		result = coeff / K2i / exp(1. / Theta_i) * sqrt(Theta_i) / sqrt(th_sum * th_sum * th_sum) * (2. * th_sum * th_sum + 2. * th_sum + 1.);
	}
	else // general form in Sadowski+17 (eq 20)
	{
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
    int i,j,k,l = 0;
    double H_values[PARAMETER_H], B_values[PARAMETER_B], ne_values[PARAMETER_NE], Te_values[PARAMETER_TE];
    double ****cooling_values;

    cooling_values = (double ****) malloc(PARAMETER_H * sizeof(double ***));
    for (int i = 0; i < PARAMETER_B ; i++) {
        cooling_values[i] = (double ***) malloc(PARAMETER_B  * sizeof(double **));
        for (int j = 0; j < PARAMETER_NE; j++) {
            cooling_values[i][j] = (double **) malloc(PARAMETER_NE * sizeof(double *));
            for (int k = 0; k < PARAMETER_TE; k++) {
                cooling_values[i][j][k] = (double *) malloc(PARAMETER_TE  * sizeof(double));
            }
        }
    }
    FILE *file_height;
    file_height = fopen("scale_height.txt", "r");
    FILE *file_e_density;
    file_e_density = fopen("ne.txt", "r");
    FILE *file_temperature;
    file_temperature = fopen("te.txt", "r");
    FILE *file_mag_field;
    file_mag_field = fopen("mag.txt", "r");

    if (file_height == NULL || file_e_density == NULL || file_temperature == NULL || file_mag_field == NULL)
    {
        printf("Error Reading File\n");
        exit(0);
    }
    printf("Getting the parameters...\n");
    //Assign value for the parameters
    for (i = 0; fscanf(file_height, "%lf", &H_values[i]) == 1; i++) {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_mag_field, "%lf", &B_values[i]) == 1; i++) {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_e_density, "%lf", &ne_values[i]) == 1; i++) {
        // Do nothing inside the loop body, everything is done in the for loop header
    }
    for (i = 0; fscanf(file_temperature, "%lf", &Te_values[i]) == 1; i++) {
        // Do nothing inside the loop body, everything is done in the for loop header
    }

    #if (COMPARISON_MARCEL)
        double B_test, ne_test, *te_test, H = 0.01 * 1.483366675977058e6 * 5, *tau_test, mu = 0.1, result, P_rad;
        tau_test = malloc(20 * sizeof(double));
        te_test = malloc(20 * sizeof(double));
        double tau_start = 1.e-6, tau_end = 5.e2;
        double te_start = 5.e4, te_end = 2.e11;
        FILE *file_result;
        file_result = fopen("marcel_comp.txt", "w");
        logspace(tau_start, tau_end, 20, tau_test);
        logspace(te_start, te_end, 20, te_test);
        for (i = 0; i < 20; i++) {
            for(k = 0; k < 20; k++){
                ne_test = tau_test[i]/(H * THOMSON_CGS);
                B_test = sqrt(2 * mu *BOLTZ_CGS* ne_test * te_test[k]);
                result = totalthincooling_rate(H, ne_test, te_test[k], B_test);
                //P_rad = total_cooling(H, ne_test, te_test[k], B_test) * H/C_CGS * (total_optical_depth(H, ne_test, te_test[k], B_test) + 4./3.);
                //result = P_rad/(BOLTZ_CGS* ne_test * te_test[k]);
                fprintf(file_result, "%.8e, ", result);

            }
        }

    #elif(SINGLE_VALUE || COULOMB_TEST)
        double H, B, ne, te, ti;
        double loop = 100;
        char str[1];
        while (loop > 1)
        {   
            #if (COULOMB_TEST)
            printf("Valor de edens\n");
            scanf("%le", &ne);
            ne = pow(10, ne);
            printf ("valor de itemp\n");
            scanf("%le", &ti);
            ti= pow(10, ti);
            printf ("valor de etemp\n");
            scanf("%le", &te);
            te= pow(10, te);
            printf("Te = %f, Ti = %f, ne = %f\n", te, ti, ne);
            printf("o valor do coulomb total é:%le\n", coulomb_heating(te, ti, ne));
            #else
            printf("valor do scale_height\n");
            scanf("%le", &H);
            H = pow(10, H);
            printf ("valor do B\n");
            scanf("%le", &B);
            B = pow(10, B);
            printf("Valor de edens\n");
            scanf("%le", &ne);
            ne = pow(10, ne);
            printf ("valor de etemp\n");
            scanf("%le", &te);
            te= pow(10, te);
            printf("H = %le, ne = %le, Te = %le, B = %le\n", H, ne, te, B);
            printf("\nOne of the roots is: %lf\n",secant_bounded(f, H, ne, te, B));
            printf("o valor do thetae =:%.11e\n", thetae(te));
            //printf("o valor do H_aqui =:%le\n", H);
            //printf("o valor do H =:%le\n", scale_height(H, ne, te));
            printf("O valor do bremmstrahlung ee =:%le\n", bremmstrahlung_ee(ne, te));
            printf("O valor do bremmstrahlung ei =:%le\n", bremmstrahlung_ei(ne, te));
            printf("O valor do bremmstrahlung total =:%le\n", bremmscooling_rate(ne, te));
            printf("o valor da freq crit =: %.11e\n", crit_freq(H, ne, te, B));
            printf("o valor do rsync =: %le\n", rsync(H, ne, te, B));
            printf("o valor do comptonization factor_ny =: %.11e\n", comptonization_factor_ny(H, ne, te, B));
            printf("o valor do comptonization factor =: %.11e\n", comptonization_factor_ny(H, ne, te, B));
            printf("o valor do cooling total no disco fino =:%le\n", totalthincooling_rate(H, ne, te, B));
            printf("O valor do tau_scat =:%le\n", soptical_depth(H, ne, te));
            printf("O valor do tau_abs =:%le\n", absoptical_depth(H, ne, te, B));
            printf("O valor do tau_total =:%le\n", total_optical_depth(H, ne, te, B));
            printf("o valor do cooling total =:%le\n", total_cooling(H, ne, te, B));
            printf("o valor do blackbody =:%le\n", bbody(H, ne, te, B));
            printf("o valor do cooling total em log:%le\n", log10(total_cooling(H, ne, te, B)));
            #endif
            printf("Do you want to read other values? y/n\n");
            scanf("%s", str);
            if (strcmp(str, "n") == 0)
            {
                loop = 0;
            }
        }
    #elif(BLACKBODYTEST)
        printf("Starting Black Body Table\n");
        FILE *file_result;
        file_result = fopen("bbody_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(bbody(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #elif(SYNCHROTRONTEST)
        printf("Starting Synch Table\n");
        FILE *file_result;
        file_result = fopen("synch_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(rsync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #elif(C_SYNCHROTRONTEST)
        printf("Starting C_synch Table\n");
        FILE *file_result;
        file_result = fopen("C_synch_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(rsync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])) * comptonization_factor_ny(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #elif(COMPTONTEST)
        printf("Starting Compton Table\n");
        FILE *file_result;
        file_result = fopen("compton_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(comptonization_factor_ny(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #elif(BREMSTRAHLUNGTEST)
        printf("Starting Brems Table\n");
        FILE *file_result;
        file_result = fopen("brems_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(bremmscooling_rate(pow(10,ne_values[k]), pow(10,Te_values[l])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #elif(ABSORPTIONDEPTHTEST)
        printf("Starting Tau_abs Table\n");
        FILE *file_result;
        file_result = fopen("tau_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(absoptical_depth(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #elif(RECALCULATE_GRID_TEST)
        FILE *file_result;
        file_result = fopen("cooling_test_finalfantasy.txt", "w");
        FILE *file_result_coulomb;
        file_result_coulomb = fopen("coulomb_test.txt", "w");
        FILE *file_height_test;
        file_height_test = fopen("scaleheight_sim.txt", "r");
        FILE *file_e_density_test;
        file_e_density_test = fopen("electronic_density_sim.txt", "r");
        FILE *file_temperature_test;
        file_temperature_test = fopen("electronic_temperature_sim.txt", "r");
        FILE *file_mag_field_test;
        file_mag_field_test = fopen("magnetic_field_sim.txt", "r");


        #if(COULOMB_RECALCULATE_GRID)
        FILE *file_itemperature_test;
        file_itemperature_test = fopen("ion_temperature_sim.txt", "r");
        if (file_itemperature_test == NULL)
        {
            printf("Error Reading Files from test\n");
            exit(0);
        }
        #endif

        if (file_height == NULL || file_e_density == NULL || file_temperature == NULL || file_mag_field == NULL)
        {
            printf("Error Reading Files from test\n");
            exit(0);
        }
        double H_test[72192], B_test[72192], ne_test[72192], Te_test[72192], cool_test;

        #if(COULOMB_RECALCULATE_GRID)
        double Ti_test[72192], coulomb_analy;
        for (i = 0; fscanf(file_itemperature_test, "%lf", &Ti_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        #endif

        for (i = 0; fscanf(file_height_test, "%lf", &H_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_mag_field_test, "%lf", &B_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_e_density_test, "%lf", &ne_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (i = 0; fscanf(file_temperature_test, "%lf", &Te_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        for (i = 0; i < 72192; i++) {
            cool_test = log10(total_cooling(pow(10,H_test[i]), pow(10,ne_test[i]), pow(10,Te_test[i]), pow(10,B_test[i])));

            #if(COULOMB_RECALCULATE_GRID)
            coulomb_analy = coulomb_heating(pow(10., Te_test[i]), pow(10., Ti_test[i]), pow(10., ne_test[i]));
            //printf("ne = %lf, ti = %lf, te = %lf, coulomb = %le\n", ne_test[i], Ti_test[i], Te_test[i], coulomb_analy);
            fprintf(file_result_coulomb, "%.8e\n", coulomb_analy);
            #endif
        
            //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_test[i], B_test[i], ne_test[i], Te_test[i], cool_test);
            fprintf(file_result, "%.8e\n", cool_test);
        } 
        fclose(file_height_test);
        fclose(file_e_density_test);
        fclose(file_temperature_test);
        fclose(file_mag_field_test);
        #if(COULOMB_RECALCULATE_GRID)
        fclose(file_itemperature_test);
        #endif
        fclose(file_result_coulomb);
    #else
        printf("Starting Cooling Table\n");
        FILE *file_result;
        file_result = fopen("cooling_table.txt", "w");
        printf("Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        #pragma omp parallel for collapse(4)
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        cooling_values[i][j][k][l] = log10(total_cooling(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
        printf("Writing down the table...\n");
        fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n");
        // run for logarithm values
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values[i][j][k][l]);
                    }
                }
            }
        }
    #endif
    fclose(file_height);
    fclose(file_e_density);
    fclose(file_temperature);
    fclose(file_mag_field);
    #if(!SINGLE_VALUE && !COULOMB_TEST)
        fclose(file_result);
    #endif
    // for (int i = 0; i < PARAMETER_B; i++) {
    //     for (int j = 0; j < PARAMETER_NE; j++) {
    //         for (int k = 0; k < PARAMETER_TE; k++) {
    //             free(cooling_values[i][j][k]);
    //         }
    //         free(cooling_values[i][j]);
    //     }
    //     free(cooling_values[i]);
    // }
    // free(cooling_values);
    printf("Table created sucessfully! Exitting...\n");

    return 0;
}
