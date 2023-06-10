// Cooling Functions including Bremmstrahlung, Synchrotron and Comptonized Synchrotron
// Code by Pedro Naethe Motta
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>


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
#define ABSORPTIONTEST (0) //Generates a table with absorption values
#define COULOMBTEST (1)
#define RECALCULATE_GRID_TEST (0)
#define SINGLE_VALUE (0) // Individual value of every function for a certain quantity of parameters
#define COMPARISON_MARCEL (0) //Compare plot A.1 of Marcel et al. 2018: A unified accretion-ejection paradigm for black hole X-ray binaries

#define PARAMETER_H (201)
#define PARAMETER_B (201)
#define PARAMETER_NE (201)
#define PARAMETER_TE (201)

#define INDEX (l + PARAMETER_TE * (k + PARAMETER_NE * (j + PARAMETER_B * i)))
#define INDEX_COULOMB (k + PARAMETER_TE * (j + PARAMETER_TE * i))


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

/*double scale_height(double radius, double edens, double etemp)
{
    double rho = edens * MH_CGS;
    if(ARAD_CGS * pow(etemp, 3.)/(edens * BOLTZ_CGS) > 1.){
	    return pow(ARAD_CGS * pow(etemp,4.)/(3 * rho * C_G * C_Mbh) ,1./2.) * pow(radius, 3./2.);
    }
    else{
        return pow(BOLTZ_CGS * etemp/(MH_CGS * C_G * C_Mbh),1./2.) * pow(radius, 3./2.);
    }
}*/

double f(double scale_height, double x, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    if (thetae(etemp) < 0.03)
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
        if (fabs(dx) < xacc || f == 0.0) return rts; //Convergence.
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
    return (1.5) * (2.80 * pow(10.,6.) * mag_field) * pow(thetae(etemp), 2.) * secant_bounded(f,scale_height, edens, etemp, mag_field);
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
    if (thetae(etemp) > 0.03)
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
    double result = 2. * edens * THOMSON_CGS * scale_height;
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

double totalthincooling_rate(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    double result = bremmscooling_rate(edens, etemp) + rsync(scale_height, edens, etemp, mag_field) * comptonization_factor_ny(scale_height, edens, etemp, mag_field);
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

int main(int argc, char** argv)
{

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time, end_time, elapsed_time;
    start_time = MPI_Wtime();
    int i, j, k, l, flag;
    int local_start, local_end, local_size;
    double H_values[PARAMETER_H], B_values[PARAMETER_B], ne_values[PARAMETER_NE], Te_values[PARAMETER_TE];

    #if(COULOMBTEST)
    local_size = PARAMETER_NE / size;
    int remainder = PARAMETER_NE % size;
    local_start = rank * local_size;
    local_end = local_start + local_size;
    if (rank == size - 1) {
        local_end += remainder;
        local_size += remainder;
    }
    #elif
    local_size = PARAMETER_H / size;
    int remainder = PARAMETER_H % size;
    local_start = rank * local_size;
    local_end = local_start + local_size;
    if (rank == size - 1) {
        local_end += remainder;
        local_size += remainder;
    }
    #endif

    if(rank ==0) fprintf(stderr,"Number of processes: %d \n", size);
    fprintf(stderr, "rank = %d, local_size =%d\n", rank, local_size);
    #if(COULOMBTEST)
    double *coulomb_values_local;
    coulomb_values_local = (double *) malloc (PARAMETER_NE * PARAMETER_TE * PARAMETER_TE * sizeof(double));
    // Set all elements to 0
    memset(coulomb_values_local, 0,  PARAMETER_NE * PARAMETER_TE * PARAMETER_TE * sizeof(double));

    double *coulomb_values_all;
    coulomb_values_all= (double *) malloc (PARAMETER_NE * PARAMETER_TE * PARAMETER_TE * sizeof(double));
    #elif
    double *cooling_values_local;
    cooling_values_local = (double *) malloc (PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE * sizeof(double));
    // Set all elements to 0
    memset(cooling_values_local, 0, PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE * sizeof(double));

    double *cooling_values_all;
    cooling_values_all= (double *) malloc (PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE * sizeof(double));
    #endif

    if (rank == 0) {
        // Only rank 0 opens and reads the files
        #if (COULOMBTEST)
        FILE *file_e_density;
        file_e_density = fopen("ne_200.txt", "r");
        FILE *file_temperature;
        file_temperature = fopen("te_200.txt", "r");

        for (i = 0; fscanf(file_e_density, "%lf", &ne_values[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        // Read Te_values
        for (i = 0; fscanf(file_temperature, "%lf", &Te_values[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        fclose(file_e_density);
        fclose(file_temperature);

        #elif
        FILE *file_height;
        file_height = fopen("scale_height_100.txt", "r");
        FILE *file_e_density;
        file_e_density = fopen("ne_100.txt", "r");
        FILE *file_temperature;
        file_temperature = fopen("te_100.txt", "r");
        FILE *file_mag_field;
        file_mag_field = fopen("mag_100.txt", "r");


        if (file_height == NULL || file_e_density == NULL || file_temperature == NULL || file_mag_field == NULL) {
            printf("Error Reading File\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read H_values
        for (i = 0; fscanf(file_height, "%lf", &H_values[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        // Read B_values
        for (i = 0; fscanf(file_mag_field, "%lf", &B_values[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        // Read ne_values
        for (i = 0; fscanf(file_e_density, "%lf", &ne_values[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        // Read Te_values
        for (i = 0; fscanf(file_temperature, "%lf", &Te_values[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        fclose(file_height);
        fclose(file_e_density);
        fclose(file_temperature);
        fclose(file_mag_field);
        #endif
    }

    // Broadcast parameter values to all ranks
    #if(COULOMBTEST)
    MPI_Bcast(ne_values, PARAMETER_NE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Te_values, PARAMETER_TE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #elif
    MPI_Bcast(H_values, PARAMETER_H, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_values, PARAMETER_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ne_values, PARAMETER_NE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Te_values, PARAMETER_TE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #endif

    #if(BLACKBODYTEST)
    if(rank ==0)fprintf(stderr,"Calculating the bbody table in parallelized way. This can take a while...\n");
    #elif(SYNCHROTRONTEST)
    if(rank ==0)fprintf(stderr,"Calculating the synch table in parallelized way. This can take a while...\n");
    #elif(C_SYNCHROTRONTEST)
    if(rank ==0)fprintf(stderr,"Calculating the csynch table in parallelized way. This can take a while...\n");
    #elif(COMPTONTEST)
    if(rank ==0)fprintf(stderr,"Calculating the compton table in parallelized way. This can take a while...\n");
    #elif(BREMSTRAHLUNGTEST)
    if(rank ==0)fprintf(stderr,"Calculating the bremms table in parallelized way. This can take a while...\n");
    #elif(ABSORPTIONTEST)
    if(rank ==0)fprintf(stderr,"Calculating the tau table in parallelized way. This can take a while...\n");
    #elif(COULOMBTEST)
    if(rank ==0)fprintf(stderr,"Calculating the coulomb table in parallelized way. This can take a while...\n");
    #else
    if(rank ==0)fprintf(stderr,"Calculating the cooling table in parallelized way. This can take a while...\n");
    #endif


    #if(COULOMBTEST)
    omp_set_num_threads(omp_get_max_threads()/size);
    #pragma omp parallel for collapse(3)
    for (i = local_start; i < local_end; i++) {
        for (j = 0; j < PARAMETER_TE; j++) {
            for (k = 0; k < PARAMETER_TE; k++) {
                    if (flag == 0) {
                        printf("Rank %d, Thread %d out of %d threads_per_rank\n", rank,  (omp_get_thread_num()) , omp_get_max_threads());
                        flag = 1; // Set flag to indicate message has been printed
                    }
                    coulomb_values_local[INDEX_COULOMB] = coulomb_heating(pow(10,Te_values[k]), pow(10,Te_values[j]), pow(10,ne_values[i]));
                    //printf("ne_values = %lf, Ti_values = %lf, Te_values = %lf, coulomb = %lf\n", ne_values[i], Te_values[j], Te_values[k], coulomb_values_local[INDEX_COULOMB]);
            }
        }
    }

    #elif
    omp_set_num_threads(omp_get_max_threads()/size);
    #pragma omp parallel for collapse(4)
    for (i = local_start; i < local_end; i++) {
        for (j = 0; j < PARAMETER_B; j++) {
            for (k = 0; k < PARAMETER_NE; k++) {
                for (l = 0; l < PARAMETER_TE; l++) {
                    if (flag == 0) {
                        printf("Rank %d, Thread %d out of %d threads_per_rank\n", rank,  (omp_get_thread_num()) , omp_get_max_threads());
                        flag = 1; // Set flag to indicate message has been printed
                    }
                    #if(BLACKBODYTEST)
                    cooling_values_local[INDEX] = log10(bbody(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                    #elif(SYNCHROTRONTEST)
                    cooling_values_local[INDEX]= log10(rsync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                    #elif(C_SYNCHROTRONTEST)
                    cooling_values_local[INDEX] = log10(rsync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])) * comptonization_factor_ny(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                    #elif(COMPTONTEST)
                    cooling_values_local[INDEX] = log10(comptonization_factor_ny(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                    #elif(BREMSTRAHLUNGTEST)
                    cooling_values_local[INDEX]= log10(bremmscooling_rate(pow(10,ne_values[k]), pow(10,Te_values[l])));
                    #elif(ABSORPTIONTEST)
                    cooling_values_local[INDEX] = log10(absoptical_depth(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                    #else
                    cooling_values_local[INDEX] = log10(total_cooling(pow(10, H_values[i]), pow(10, ne_values[k]), pow(10, Te_values[l]), pow(10, B_values[j])));
                    #endif
                }
            }
        }
    }
    #endif
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    fprintf(stderr, "Rank %d: Elapsed time: %.2f seconds for calculation of the table components\n", rank, elapsed_time);

    MPI_Barrier(MPI_COMM_WORLD);  // Add a barrier synchronization point
    //bring all the values to the global array in rank 0 
    #if(COULOMBTEST)
    MPI_Reduce(coulomb_values_local, coulomb_values_all, PARAMETER_NE * PARAMETER_TE* PARAMETER_TE, MPI_DOUBLE, MPI_SUM, 0 ,MPI_COMM_WORLD);
    #elif
    MPI_Reduce(cooling_values_local, cooling_values_all, PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE, MPI_DOUBLE, MPI_SUM, 0 ,MPI_COMM_WORLD);
    #endif
    if (rank == 0) fprintf(stderr,"Writing down the table...\n");
    MPI_Barrier(MPI_COMM_WORLD);  // Add a barrier synchronization point
    
    if(rank ==0){
        FILE *file_result;
        #if(BLACKBODYTEST)
        file_result = fopen("bbody_table.bin", "w");
        #elif(SYNCHROTRONTEST)
        file_result = fopen("synch_table.bin", "w");
        #elif(C_SYNCHROTRONTEST)
        file_result = fopen("C_synch_table.bin", "w");
        #elif(COMPTONTEST)
        file_result = fopen("compton_table.bin", "w");
        #elif(BREMSTRAHLUNGTEST)
        file_result = fopen("brems_table.bin", "w");
        #elif(ABSORPTIONTEST)
        file_result = fopen("tau_table.bin", "w");
        #elif(COULOMBTEST)
        file_result = fopen("coulomb_table.bin", "w");
        #else
        file_result = fopen("cooling_table.bin", "w");
        #endif
        #if(COULOMBTEST)
        //fprintf(file_result, " e_density, temperature_i, temperature_e , coulomb\n"); //code used to write in .txt file
        for (i = 0; i < PARAMETER_NE; i++) {
            for (j = 0; j < PARAMETER_TE; j++) {
                for (k = 0; k < PARAMETER_TE; k++) {
                    //fprintf(file_result, "%.2f, %.2f, %.2f, %.8e\n", ne_values[i], Te_values[j], Te_values[k], coulomb_values_all[INDEX_COULOMB]); //code used to write in .txt file
                    fwrite(&coulomb_values_all[INDEX_COULOMB], sizeof(double), 1, file_result); // Write cooling_values_all array
                }
            }
        }
        #elif
        //fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n"); //code used to write in .txt file
        for (i = 0; i < PARAMETER_H; i++) {
            for (j = 0; j < PARAMETER_B; j++) {
                for (k = 0; k < PARAMETER_NE; k++) {
                    for (l = 0; l < PARAMETER_TE; l++) {
                        //fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values_all[INDEX]); //code used to write in .txt file
                        fwrite(&cooling_values_all[INDEX], sizeof(double), 1, file_result); // Write cooling_values_all array

                    }
                }
            }
        }
        #endif
        fclose(file_result);
    }
    MPI_Barrier(MPI_COMM_WORLD);  // Add a barrier synchronization point
    MPI_Finalize();
    #if(COULOMBTEST)
    if(rank == 0) free(coulomb_values_all);
    free(coulomb_values_local);
    #elif
    if(rank == 0) free(cooling_values_all);
    free(cooling_values_local);
    #endif
    if(rank ==0) printf("Table created sucessfully! Exitting...\n");

    return 0;
}
