// Cooling Functions including Bremmstrahlung, Synchrotron and Comptonized Synchrotron
// Code by Pedro Naethe Motta
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>


/* Defining Constants in CGS*/
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
#define C_Mbh (10. * C_Msun) /*Mass of the black hole*/

/*Switches to choose between different table generation*/
#define BLACKBODYTEST (0) //Generates a table for the blackbody equation
#define SYNCHROTRONTEST (0) //Generates a table for synchrotron equation
#define C_SYNCHROTRONTEST (0)// Generates a table for comptonized synchrotron equation
#define COMPTONTEST (0) // Generates a table with the compton values
#define BREMSTRAHLUNGTEST (0) //Generates a table for bremmstrahlung equation
#define ABSORPTIONTEST (0) //Generates a table with absorption values
#define COULOMBTEST (0) //Generates a table with coulomb values
#define PRINT_BINARY (0) // Whether to print in binary or txt file


/*Switches to change between table generation and test, for table generation put everything 0*/
#define RECALCULATE_GRID_TEST (0)/*Put simulation values for the 4 parameters and it will calculate*/
    #define COULOMB_RECALCULATE_GRID (0) // Do the same as RECALCULATE_GRID_TEST, but with coulomb, activate both.
    #define N_RESOLUTION 12600 //Total number of cells from simulation.
#define SINGLE_VALUE (1) //Individual value of every function for a defined quantity of parameters
    #define COULOMB_TEST (0) //Do the same as SINGLE_VALUE, but with coulomb, activate both.
#define COMPARISON_MARCEL (0) //Compare plot A.1 of Marcel et al. 2018: A unified accretion-ejection paradigm for black hole X-ray binaries
#define FRAGILEMEIER_TEST (0)


/*Size of the table, depend of parameters*/
#define PARAMETER_H (40 + 1)
#define PARAMETER_B (40 + 1)
#define PARAMETER_NE (40 + 1)
#define PARAMETER_TE (40 + 1)

#define PARAMETER_NE_COULOMB (200 + 1)
#define PARAMETER_TE_COULOMB (200 + 1)

/*Indexing of 3D and 4D arrays for cooling/coulomb tables*/
#define INDEX (l + PARAMETER_TE * (k + PARAMETER_NE * (j + PARAMETER_B * i)))
#define INDEX_COULOMB (k + PARAMETER_TE_COULOMB * (j + PARAMETER_TE_COULOMB * i))

/*Routine parameters for bessel/gamma functions*/
#define ITMAX (1e8)  
#define EPS (3e-7)
#define maxSteps (1e3)
#define FPMIN (1.0e-30) 
#define INTEGRATION_STEPS (1e6)

double gammln(double xxgam);
/*Standard Numerical Recipes error function*/
void nrerror(char error_text[])
{
    fprintf(stderr,"Numerical Recipes run-time error...\n");
    fprintf(stderr,"%s\n",error_text);
    fprintf(stderr,"...now exiting to system...\n");
    exit(1);
}

/*Bessel0 function defined as Numerical Recipes book*/
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

/*Modified bessel0 function defined as Numerical Recipes book*/
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

/*Bessel1 function defined as Numerical Recipes book*/
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

/*Modified bessel1 function defined as Numerical Recipes book*/
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

/*Modified bessel2 function defined as Numerical Recipes book*/
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

/*Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction representation as gammcf. Also returns ln Γ(a) as gln.*/
void gcf(double *gammcf, double a, double x, double *gln)
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

/*Returns the incomplete gamma function P(a, x) evaluated by its series representation as gamser. Also returns ln Γ(a) as gln.*/
void gser(double *gamser, double a, double x, double *gln)
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


/*Returns the incomplete gamma function P(a, x).*/
double gammp(double a, double x)
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

/*Returns the incomplete gamma function Q(a, x) ≡ 1 − P(a, x).*/
double gammq(double a, double x)
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

/*Returns the value ln[Γ(xx)] for xx > 0.*/
double gammln(double xx)
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

/****************From now on, these are the functions to represent the equations of cooling****************/
double thetae(double etemp)
{
    double result = (BOLTZ_CGS)* etemp / ((ERM_CGS) * (pow(C_CGS, 2.)));
    return result;
}

double trans_f(double scale_height, double x, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    if (thetae(etemp) < 0.5)
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
    double xacc = 1e-5;
    double x1 = 10;
    double x2 = 20;
    double fl,f,dx,swap,xl,rts;
    double x1_bef, x2_bef;
    int interval = 1;
    fl=(*func)(scale_height, x1, edens, etemp, mag_field);
    f=(*func)(scale_height, x2, edens, etemp, mag_field);
    //printf("f = %le, fl = %le, H = %le, ne = %le, te = %le, B = %le\n", f, fl, scale_height, edens, etemp, mag_field);
    while(interval){
        if (f * fl > 0){
            x1_bef = x1;
            x1 = x1/2;
            x2_bef = x2;
            x2 = x2*2;
            //printf("x1 = %le, x2 = %le \n", x1, x2);
        }
        else{
            interval = 0;
            // printf("Finally x1 = %le, x2 = %le \n", x1, x2);
            // printf("Finally x1_bef = %le, x2_bef = %le \n", x1_bef, x2_bef);
        }
        fl=(*func)(scale_height, x1, edens, etemp, mag_field);
        f=(*func)(scale_height, x2, edens, etemp, mag_field);
        //printf("fl = %le, f = %le\n", fl, f);
    }

    //printf("Finally x1 = %le, x2 = %le \n", x1, x2);
    //printf("xl = %le, rts = %le, f = %le, fl = %le\n", xl, rts, f, fl);
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
    //printf("xl = %le, rts = %le, f = %le, fl = %le\n", xl, rts, f, fl);
    for (j=1;j<=ITMAX;j++) { //Secant loop.
        dx=(xl-rts)*f/(f-fl); //Increment with respect to latest value.
        xl=rts;
        fl=f;
        rts += dx;
        f=(*func)(scale_height, rts, edens, etemp, mag_field);
        if (fabs(dx) < xacc || f == 0.0) {
            return rts; //Convergence.
        }
    }
    //printf("Error! H = %le, B = %le, ne = %le, Te = %le\n", scale_height, mag_field, edens, etemp);
    //printf("Maximum number of iterations exceeded in rtsec");
    return 0.0; //Never get here.
}



/*Electron-electron bremsstrahlung process*/
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

/*Electron-ion bremsstrahlung process*/
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

/*Sum of both bremsstrahlung*/
double bremmscooling_rate(double edens, double etemp)
{//All the function was checked step by step, seems to be working good.

    double result = bremmstrahlung_ee(edens, etemp) + bremmstrahlung_ei(edens, etemp);
    return result;
}

/*Critical frequency calculation for synchrotron cooling*/
double crit_freq(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    return (1.5) * (2.80 * pow(10.,6.) * mag_field) * pow(thetae(etemp), 2.) * secant_bounded(trans_f,scale_height, edens, etemp, mag_field);
}

/*Synchtron radiation calculation*/
double rsync(double scale_height, double edens, double etemp, double mag_field)
{ //All the function was checked step by step, seems to be working good.
    double nuzero = 2.80 * pow(10., 6.) * mag_field;
    double a1 = 2. / (3. * nuzero * pow(thetae(etemp), 2.));
    double a2 = 0.4 / pow(a1, 1. / 4.);
    double a3 = 0.5316 / pow(a1, 1. / 2.);
    double a4 = 1.8899 * pow(a1, 1. / 3.);
    double critical_frequency = crit_freq(scale_height, edens, etemp, mag_field);
    if (thetae(etemp) > 0.5)
    {
        double self_abs = 2. * C_PI * BOLTZ_CGS * etemp * pow(critical_frequency, 3.) / (3. * scale_height * pow(C_CGS, 2.));
        double init_term2 =6.76 * pow(10., -28.) * edens / (bessk2(1. / thetae(etemp)) * pow(a1, 1. / 6.));
        double syn_1 = 1. / pow(a4, 11. / 2.) *(exp(gammln(11./2.)))* gammq(11. / 2., a4 * pow(critical_frequency, 1. / 3.));
        double syn_2 = a2 / pow(a4, 19. / 4.) *(exp(gammln(19./4.)))* gammq(19. / 4., a4 * pow(critical_frequency, 1. / 3.));
        double syn_3 = a3 / pow(a4, 4.) * (pow(a4, 3.) * critical_frequency + 3. * pow(a4, 2.) * pow(critical_frequency, 2. / 3.) + 6. * a4 * pow(critical_frequency, 1. / 3.) + 6.) * exp(-a4 * pow(critical_frequency, 1. / 3.));
        double synchrotron = init_term2 * (syn_1+syn_2+syn_3);
        double result = self_abs + synchrotron;
        
        return result;
    }
    else
    {
        double self_abs = 2. * C_PI * BOLTZ_CGS * etemp * pow(critical_frequency, 3.) / (3. * scale_height * pow(C_CGS, 2.));
        double init_term2 = 6.76 * pow(10., -28.) * edens / (2. * pow(thetae(etemp), 2.) * pow(a1, 1. / 6.));
        double syn_1 = 1. / pow(a4, 11. / 2.) *(exp(gammln(11./2.)))* gammq(11. / 2., a4 * pow(critical_frequency, 1. / 3.));
        double syn_2 = a2 / pow(a4, 19. / 4.) *(exp(gammln(19./4.)))* gammq(19. / 4., a4 * pow(critical_frequency, 1. / 3.));
        double syn_3 = a3 / pow(a4, 4.) * (pow(a4, 3.) * critical_frequency + 3. * pow(a4, 2.) * pow(critical_frequency, 2. / 3.) + 6. * a4 * pow(critical_frequency, 1. / 3.) + 6.) * exp(-a4 * pow(critical_frequency, 1. / 3.));
        double synchrotron = init_term2 * (syn_1+syn_2+syn_3);
        double result = self_abs + synchrotron;
        return result;
    }
}

/*Comptonization factor defined by Esin et al. (1996)*/
double comptonization_factor_sync(double scale_height, double edens, double etemp, double mag_field){
	double thompson_opticaldepth = 2 * edens * THOMSON_CGS * scale_height;
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

double comptonization_factor_brems(double scale_height, double edens, double etemp, double nu){
	double thompson_opticaldepth = 2 * edens * THOMSON_CGS * scale_height;
	double Afactor = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
	double maxfactor = 3 * BOLTZ_CGS * etemp/(PLANCK_CGS * nu);
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

double comptonized_brems(double scale_height, double edens, double etemp, double mag_field){
    //Considering equation found in https://www.astro.utu.fi/~cflynn/Stars/l6.html and https://www.astro.rug.nl/~etolstoy/radproc/resources/lectures/lecture9pr.pdf
    //Considering gaunt factor = 1, ni = ne and z = 1 for hydrogen gas.
    double prefactor = 3.692345e8 * PLANCK_CGS/(BOLTZ_CGS);
    double electron_sca_op = 0.4; // In units of cm^2/g
    double rho = ERM_CGS * edens; //Mass density
    double gaunt_factor = 1.0;
    double frequency = pow(prefactor *gaunt_factor* pow(edens, 2.)/(pow(etemp, 3./2.) * electron_sca_op * rho), 1./2.);
    double ray_regime = PLANCK_CGS * frequency/(BOLTZ_CGS * etemp);
    double max_integration = BOLTZ_CGS * etemp/PLANCK_CGS;
    double steps = (max_integration - 1e-6)/INTEGRATION_STEPS;
    double result = 0;
    if (ray_regime > 0.1){
        return 1e-50;
    }

    for(double nu = frequency; nu <= max_integration; nu += steps){
        result +=comptonization_factor_brems(scale_height, edens, etemp, nu) * bremmscooling_rate(edens, etemp)/(max_integration - frequency) * steps;
    }
    // if (result == 0){
    //     result = bremmscooling_rate(edens, etemp);
    // }
    return result;
}


/*scattering optical depth*/
double soptical_depth(double scale_height, double edens, double etemp)
{//All the function was checked step by step, seems to be working good.
    double result = 2. * edens * THOMSON_CGS * scale_height;
    return result;
}

/*Comptonization factor defined by Narayan & Yi (1995)*/
double comptonization_factor_ny(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    double prob = 1 - exp(-soptical_depth(scale_height, edens, etemp));
    double A = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
    double eta1 = prob * (A - 1) / (1 - prob * A);
    double eta2 = PLANCK_CGS * crit_freq(scale_height, edens, etemp, mag_field)/(3 * thetae(etemp) * ERM_CGS * pow(C_CGS,2.));
    double eta3 = -1 - log(prob)/log(A);
    double result = 1 + eta1* (1 - pow(eta2, eta3));
    if (eta2 > 1){
        printf("Narayan's Compton formula not valid, exiting...");
        //exit(1);
    }
    return result;
}

/*Cooling rate for optically thin gas*/
double totalthincooling_rate(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //double result = (comptonized_brems(scale_height, edens, etemp, mag_field) > 0? comptonized_brems(scale_height, edens, etemp, mag_field): bremmscooling_rate(edens, etemp)) + rsync(scale_height, edens, etemp, mag_field) * comptonization_factor_sync(scale_height, edens, etemp, mag_field);
    double result = bremmscooling_rate(edens, etemp) + rsync(scale_height, edens, etemp, mag_field) * comptonization_factor_sync(scale_height, edens, etemp, mag_field);
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

/*Total cooling for both optically thin and thick regimes*/
double total_cooling(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    return 4. * C_sigma * pow(etemp, 4.) / scale_height * 1 /
           (3 * total_optical_depth(scale_height, edens, etemp, mag_field) / 2. + pow(3., 1. / 2.) + 1. / absoptical_depth(scale_height, edens, etemp, mag_field));
}

/*Blackbody cooling limit*/
double bbody(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    return 8. * C_sigma * pow(etemp, 4.) / (3 * scale_height * total_optical_depth(scale_height, edens, etemp, mag_field));
}

/*Function to create equally spaced intervals in log*/
void logspace(double start, double end, int num, double* result) {
    double log_start = log10(start); //Initial value
    double log_end = log10(end); //End value
    double step = (log_end - log_start) / (num - 1); // number of steps
    int i;
    for (i = 0; i < num; ++i) {
        result[i] = pow(10.0, log_start + i * step);
    }
}

/*Parameter of coulomb collisions calculation*/
double theta_i(double itemp)
{
    double result = (BOLTZ_CGS)* itemp / ((MH_CGS) * (pow(C_CGS, 2.)));
    return result;
}

/*Coulomb heating calculation*/
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
	    //printf("Oii-4 \n");
		//bessel functions
		K2e = bessk2(1. / Theta_e);
		K2i = bessk2(1. / Theta_i);
		K0 = bessk0(1.0 / th_mean);
		K1 = bessk1(1.0 / th_mean);

		result = coeff / (K2e * K2i) * ((2. * th_sum * th_sum + 1.) / th_sum * K1 + 2. * K0);
	}
	if (!isfinite(result)){
        result = 0.;
        printf("Result in coulomb collisions is inf, setting it to 0\n");
    }
	return (result);

}

int main(int argc, char** argv)
{

    #if(SINGLE_VALUE || COULOMB_TEST)
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
            printf("\nOne of the roots is: %lf\n",secant_bounded(trans_f, H, ne, te, B));
            printf("o valor do thetae =:%.11e\n", thetae(te));
            printf("O valor do bremmstrahlung ee =:%le\n", bremmstrahlung_ee(ne, te));
            printf("O valor do bremmstrahlung ei =:%le\n", bremmstrahlung_ei(ne, te));
            printf("O valor do bremmstrahlung total =:%le\n", bremmscooling_rate(ne, te));
            //printf("o valor do comptonizaed brems =: %le\n", comptonized_brems(H, ne, te, B));
            printf("o valor da freq crit =: %.11e\n", crit_freq(H, ne, te, B));
            printf("o valor do rsync =: %le\n", rsync(H, ne, te, B));
            //printf("o valor do comptonization factor_ny =: %.11e\n", comptonization_factor_ny(H, ne, te, B));
            printf("o valor do comptonization factor =: %.11e\n", comptonization_factor_sync(H, ne, te, B));
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
    #elif(COMPARISON_MARCEL)
        double B_test, ne_test, *te_test, H = 0.01 * 1.483366675977058e6 * 5, *tau_test, mu = 0.1, result, P_rad;
        tau_test = malloc(20 * sizeof(double));
        te_test = malloc(20 * sizeof(double));
        double tau_start = 1.e-6, tau_end = 5.e2;
        double te_start = 5.e4, te_end = 2.e11;
        FILE *file_result;
        file_result = fopen("marcel_comp.txt", "w");
        logspace(tau_start, tau_end, 20, tau_test);
        logspace(te_start, te_end, 20, te_test);
        for (int i = 0; i < 20; i++) {
            for(int k = 0; k < 20; k++){
                ne_test = tau_test[i]/(H * THOMSON_CGS);
                B_test = sqrt(2 * mu *BOLTZ_CGS* ne_test * te_test[k]);
                result = totalthincooling_rate(H, ne_test, te_test[k], B_test);
                //P_rad = total_cooling(H, ne_test, te_test[k], B_test) * H/C_CGS * (total_optical_depth(H, ne_test, te_test[k], B_test) + 4./3.);
                //result = P_rad/(BOLTZ_CGS* ne_test * te_test[k]);
                fprintf(file_result, "%.8e, ", result);

            }
        }
    #elif(FRAGILEMEIER_TEST)
        double B = 8.380e3, H = 2.7e7, ne = 1e-10/ERM_CGS;
        double *brems, *synch, *csynch, *bbody_v, *cbrems, *cool;
        double * Te;
        Te = malloc(1000 * sizeof(double));
        brems = malloc(1000 * sizeof(double));
        synch = malloc(1000 * sizeof(double));
        csynch = malloc(1000 * sizeof(double));
        cbrems = malloc(1000 * sizeof(double));
        cool = malloc(1000 * sizeof(double));
        logspace(1e2, 1e15, 1000, Te);
        char filename[] = "cool_comparison.bin";
        FILE *file_result;
        file_result = fopen(filename, "w");
        #pragma omp parallel for
        for (int i = 0; i < 1000; i++){
            brems[i] = bremmscooling_rate(ne, Te[i]);
            cbrems[i] = comptonized_brems(H, ne, Te[i], B);
            synch[i] = rsync(H, ne, Te[i], B);
            csynch[i] = rsync(H, ne, Te[i], B) * comptonization_factor_sync(H, ne, Te[i], B);
            //bbody_v = bbody(H, ne, Te[i], B);
            cool[i] = totalthincooling_rate(H, ne, Te[i], B);
            printf("cooling [%d] = %le ,T[%d] = %le \n",i, cool[i], i, Te[i]);
        }
        #pragma omp barrier
        printf("Starting to write...\n");

        for (int i = 0; i < 1000; i++){
            fwrite(&brems[i], sizeof(double), 1, file_result);
            fwrite(&cbrems[i], sizeof(double), 1, file_result);
            fwrite(&synch[i], sizeof(double), 1, file_result);
            fwrite(&csynch[i], sizeof(double), 1, file_result);
            //fwrite(&bbody_v, sizeof(double), 1, file_result);
            fwrite(&cool[i], sizeof(double), 1, file_result);
        }
        printf("Created file: %s \n", filename);
        free(Te);
        free(brems);
        free(synch);
        free(csynch);
        free(cbrems);
        free(cool);
        fclose(file_result);

    #elif (RECALCULATE_GRID_TEST)
        char filename[] = "cooling_test_finalfantasy.txt";
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

        if (file_height_test == NULL || file_e_density_test == NULL || file_temperature_test == NULL || file_mag_field_test == NULL)
        {
            printf("Error Reading Files from test\n");
            exit(0);
        }
        double H_test[N_RESOLUTION], B_test[N_RESOLUTION], ne_test[N_RESOLUTION], Te_test[N_RESOLUTION], cool_test;

        #if(COULOMB_RECALCULATE_GRID)
        double Ti_test[72192], coulomb_analy;
        for (int i = 0; fscanf(file_itemperature_test, "%lf", &Ti_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        #endif

        for (int i = 0; fscanf(file_height_test, "%lf", &H_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (int i = 0; fscanf(file_mag_field_test, "%lf", &B_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (int i = 0; fscanf(file_e_density_test, "%lf", &ne_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        for (int i = 0; fscanf(file_temperature_test, "%lf", &Te_test[i]) == 1; i++) {
            // Do nothing inside the loop body, everything is done in the for loop header
        }
        fprintf(stderr, "Calculating the table in parallelized way. This can take a while...\n");
        omp_set_num_threads(omp_get_num_threads());
        for (int i = 0; i < N_RESOLUTION; i++) {
            cool_test = log10(total_cooling(pow(10,H_test[i]), pow(10,ne_test[i]), pow(10,Te_test[i]), pow(10,B_test[i])));

            #if(COULOMB_RECALCULATE_GRID)
            coulomb_analy = coulomb_heating(pow(10., Te_test[i]), pow(10., Ti_test[i]), pow(10., ne_test[i]));
            //printf("ne = %lf, ti = %lf, te = %lf, coulomb = %le\n", ne_test[i], Ti_test[i], Te_test[i], coulomb_analy);
            fprintf(file_result_coulomb, "%.8e\n", coulomb_analy);
            #endif
        
            //printf("H = %le, B = %le, ne = %le, Te = %le, value = %.8e\n", H_test[i], B_test[i], ne_test[i], Te_test[i], cool_test);
            fprintf(file_result, "%.8e\n", cool_test);
        }
        fprintf(stderr, "Table generated: %s\n", filename);

        fclose(file_height_test);
        fclose(file_e_density_test);
        fclose(file_temperature_test);
        fclose(file_mag_field_test);
        #if(COULOMB_RECALCULATE_GRID)
        fclose(file_itemperature_test);
        #endif
        fclose(file_result_coulomb);
    #else
        //Initializing MPI ranks and communication. Rank is the ID of each MPI process and size is the total number of MPI processes.
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        double start_time, end_time, elapsed_time;
        /*Starting MPI time so we can keep track how much time each thread took to do the calculations.*/
        start_time = MPI_Wtime();
        int i, j, k, l, flag;
        int local_start, local_end, local_size;
        double H_values[PARAMETER_H], B_values[PARAMETER_B], ne_values[PARAMETER_NE], Te_values[PARAMETER_TE];

        /*Dividing the calculation among MPI processes equally. If the number of parameters is not evenly divisible among processes, the remainder is assigned to the last rank.
        Coulomb collisions require different distribuition, since it is one parameter shorter*/
        #if(COULOMBTEST)
        double ne_values_C[PARAMETER_NE_COULOMB], Te_values_C[PARAMETER_NE_COULOMB];
        local_size = PARAMETER_NE_COULOMB / size;
        int remainder = PARAMETER_NE_COULOMB % size;
        local_start = rank * local_size;
        local_end = local_start + local_size;
        if (rank == size - 1) {
            local_end += remainder;
            local_size += remainder;
        }
        #else
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


        /*Defining local arrays for each MPI process and a global array that will be established in rank 0 and will hold all the other values*/
        #if(COULOMBTEST)
        double *coulomb_values_local;
        coulomb_values_local = (double *) malloc (PARAMETER_NE_COULOMB * PARAMETER_TE_COULOMB * PARAMETER_TE_COULOMB * sizeof(double));
        // Set all elements to 0
        memset(coulomb_values_local, 0,  PARAMETER_NE_COULOMB * PARAMETER_TE_COULOMB * PARAMETER_TE_COULOMB * sizeof(double));
        double *coulomb_values_all;
        coulomb_values_all= (double *) malloc (PARAMETER_NE_COULOMB * PARAMETER_TE_COULOMB * PARAMETER_TE_COULOMB * sizeof(double));
        #else
        double *cooling_values_local;
        cooling_values_local = (double *) malloc (PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE * sizeof(double));
        // Set all elements to 0
        memset(cooling_values_local, 0, PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE * sizeof(double));

        double *cooling_values_all;
        cooling_values_all= (double *) malloc (PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE * sizeof(double));
        #endif

        /*Reading the parameters list for H, B , ne and Te. Only rank 0 reads and distributes it among the other nodes*/
        if (rank == 0) {
            #if (COULOMBTEST)
            FILE *file_e_density;
            file_e_density = fopen("./parameters/ne_200.txt", "r");
            FILE *file_temperature;
            file_temperature = fopen("./parameters/te_200.txt", "r");

            for (i = 0; fscanf(file_e_density, "%lf", &ne_values_C[i]) == 1; i++) {
                // Do nothing inside the loop body, everything is done in the for loop header
            }
            // Read Te_values
            for (i = 0; fscanf(file_temperature, "%lf", &Te_values_C[i]) == 1; i++) {
                // Do nothing inside the loop body, everything is done in the for loop header
            }
            //printf("It got to here!\n");
            fclose(file_e_density);
            fclose(file_temperature);

            #else
             // Print contents of the directory
            char filename[100];
            snprintf(filename, sizeof(filename), "./parameters/scale_height_%d.txt", PARAMETER_H - 1);

            FILE *file_height;
            file_height = fopen(filename, "r");
            if (file_height == NULL) {
                printf("Error Reading Scale Height File\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            snprintf(filename, sizeof(filename), "./parameters/ne_%d.txt", PARAMETER_NE - 1);
            FILE *file_e_density;
            file_e_density = fopen(filename, "r");
            if (file_e_density == NULL) {
                printf("Error Reading Electron Density File\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            snprintf(filename, sizeof(filename), "./parameters/te_%d.txt", PARAMETER_TE - 1);
            FILE *file_temperature;
            file_temperature = fopen(filename, "r");
            if (file_temperature == NULL) {
                printf("Error Reading Temperature File\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            snprintf(filename, sizeof(filename), "./parameters/mag_%d.txt", PARAMETER_B - 1);
            FILE *file_mag_field;
            file_mag_field = fopen(filename, "r");
            if (file_mag_field == NULL) {
                printf("Error Reading Magnetic Field File\n");
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

        /*Broadcast parameter for the other nodes*/
        #if(COULOMBTEST)
        MPI_Bcast(ne_values, PARAMETER_NE_COULOMB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Te_values, PARAMETER_TE_COULOMB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        #else
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
        if(rank ==0)fprintf(stderr,"Calculating the cooling table in parallelized way for %d values or %d^4 values. This can take a while...\n", PARAMETER_B * PARAMETER_H * PARAMETER_NE * PARAMETER_TE, PARAMETER_NE);
        #endif

        /*Here we do the calculation depending on which table is setted by the switches. We distribute the outer loop between MPI processes and uses 
        OpenMP to distribute it among processor's threads.*/
        #if(COULOMBTEST)
        omp_set_num_threads(omp_get_max_threads()/size);
        #pragma omp parallel for collapse(3)
        for (i = local_start; i < local_end; i++) {
            for (j = 0; j < PARAMETER_TE_COULOMB; j++) {
                for (k = 0; k < PARAMETER_TE_COULOMB; k++) {
                        if (flag == 0) {
                            printf("Rank %d, Thread %d out of %d threads_per_rank\n", rank,  (omp_get_thread_num()) , omp_get_max_threads());
                            flag = 1; // Set flag to indicate message has been printed
                        }
                        coulomb_values_local[INDEX_COULOMB] = coulomb_heating(pow(10,Te_values_C[k]), pow(10,Te_values_C[j]), pow(10,ne_values_C[i]));
                        //printf("ne_values = %lf, Ti_values = %lf, Te_values = %lf, coulomb = %lf\n", ne_values[i], Te_values[j], Te_values[k], coulomb_values_local[INDEX_COULOMB]);
                }
            }
        }

        #else
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
                        cooling_values_local[INDEX] = log10(rsync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])) * comptonization_factor_sync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
                        #elif(COMPTONTEST)
                        cooling_values_local[INDEX] = log10(comptonization_factor_sync(pow(10,H_values[i]), pow(10,ne_values[k]), pow(10,Te_values[l]), pow(10,B_values[j])));
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
        /*Stop MPI wall time and print the elapsed time for each MPI process, so you can keep track of the time.*/
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        fprintf(stderr, "Rank %d: Elapsed time: %.2f seconds for calculation of the table components\n", rank, elapsed_time);

        MPI_Barrier(MPI_COMM_WORLD);  // Add a barrier synchronization point

        /*All the values for cooling are broadcasted back to rank 0.*/
        #if(COULOMBTEST)
        MPI_Reduce(coulomb_values_local, coulomb_values_all, PARAMETER_NE_COULOMB * PARAMETER_TE_COULOMB* PARAMETER_TE_COULOMB, MPI_DOUBLE, MPI_SUM, 0 ,MPI_COMM_WORLD);
        #else
        MPI_Reduce(cooling_values_local, cooling_values_all, PARAMETER_H * PARAMETER_B * PARAMETER_NE * PARAMETER_TE, MPI_DOUBLE, MPI_SUM, 0 ,MPI_COMM_WORLD);
        #endif
        if (rank == 0) fprintf(stderr,"Writing down the table...\n");
        MPI_Barrier(MPI_COMM_WORLD);  // Add a barrier synchronization point
        
        /*Only rank 0 writes the binary/txt file.*/
        if(rank ==0){
            FILE *file_result;
            #if(BLACKBODYTEST)
            file_result = fopen("bbody_table_05.bin", "w");
            #elif(SYNCHROTRONTEST)
            file_result = fopen("synch_table_05.bin", "w");
            #elif(C_SYNCHROTRONTEST)
            file_result = fopen("C_synch_table_05.bin", "w");
            #elif(COMPTONTEST)
            file_result = fopen("compton_table_05.bin", "w");
            #elif(BREMSTRAHLUNGTEST)
            file_result = fopen("brems_table_05.bin", "w");
            #elif(ABSORPTIONTEST)
            file_result = fopen("tau_table_05.bin", "w");
            #elif(COULOMBTEST)
            char filename_result[100];
                #if(!PRINT_BINARY)
                    snprintf(filename_result, sizeof(filename_result), "./tables/coulomb_table_%d.txt", PARAMETER_NE_COULOMB -1);
                    file_result = fopen(filename_result, "w");
                #else
                    snprintf(filename_result, sizeof(filename_result), "./tables/coulomb_table_%d.bin", PARAMETER_NE_COULOMB -1);
                    file_result = fopen(filename_result, "w");
                #endif
            #else
            char filename_result[100];
                #if(!PRINT_BINARY)
                    snprintf(filename_result, sizeof(filename_result), "./tables/cooling_table_%d.txt", PARAMETER_H -1);
                    file_result = fopen(filename_result, "w");
                #else
                    snprintf(filename_result, sizeof(filename_result), "./tables/cooling_table_%d.bin", PARAMETER_H -1);
                    file_result = fopen(filename_result, "w");
                #endif
            #endif
            #if(COULOMBTEST)
                #if(PRINT_BINARY)
                for (i = 0; i < PARAMETER_NE_COULOMB; i++) {
                    for (j = 0; j < PARAMETER_TE_COULOMB; j++) {
                        for (k = 0; k < PARAMETER_TE_COULOMB; k++) {
                            fwrite(&coulomb_values_all[INDEX_COULOMB], sizeof(double), 1, file_result); // Write cooling_values_all array
                        }
                    }
                }
                #else
                fprintf(file_result, " e_density, temperature_i, temperature_e , coulomb\n"); //code used to write in .txt file
                for (i = 0; i < PARAMETER_NE_COULOMB; i++) {
                    for (j = 0; j < PARAMETER_TE_COULOMB; j++) {
                        for (k = 0; k < PARAMETER_TE_COULOMB; k++) {
                            fprintf(file_result, "%.2f, %.2f, %.2f, %.8e\n", ne_values_C[i], Te_values_C[j], Te_values_C[k], coulomb_values_all[INDEX_COULOMB]); //code used to write in .txt file
                        }
                    }
                }
                #endif
            #else
                #if(PRINT_BINARY)
                for (i = 0; i < PARAMETER_H; i++) {
                    for (j = 0; j < PARAMETER_B; j++) {
                        for (k = 0; k < PARAMETER_NE; k++) {
                            for (l = 0; l < PARAMETER_TE; l++) {
                                fwrite(&cooling_values_all[INDEX], sizeof(double), 1, file_result); // Write cooling_values_all array

                            }
                        }
                    }
                }
                #else
                fprintf(file_result, "scale_height, mag_field, e_density, temperature, cooling\n"); //code used to write in .txt file
                for (i = 0; i < PARAMETER_H; i++) {
                    for (j = 0; j < PARAMETER_B; j++) {
                        for (k = 0; k < PARAMETER_NE; k++) {
                            for (l = 0; l < PARAMETER_TE; l++) {
                                fprintf(file_result, "%.2f, %.2f, %.2f, %.2f, %.8e\n", H_values[i], B_values[j], ne_values[k], Te_values[l], cooling_values_all[INDEX]); //code used to write in .txt file
                            }
                        }
                    }
                }
                #endif
            #endif
            fclose(file_result);
        }
        MPI_Barrier(MPI_COMM_WORLD);  // Add a barrier synchronization point
        /*Finalize MPI, free arrays and end the code.*/
        MPI_Finalize();
        #if(COULOMBTEST)
        if(rank == 0) free(coulomb_values_all);
        free(coulomb_values_local);
        #else
        if(rank == 0) free(cooling_values_all);
        free(cooling_values_local);
        #endif
        if(rank ==0) printf("Table created sucessfully! Exitting...\n");
    #endif
    return 0;
}
