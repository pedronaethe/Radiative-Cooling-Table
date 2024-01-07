  
#include <time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

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

#define SIZEOF_TEST 30 /*Quad root of number of calculations for GLOBAL_MEMORY_TEST*/
#define INDEX(i, j, k, l) (l + SIZEOF_TE * (k + SIZEOF_NE * (j + SIZEOF_B * i))) /*4D indexing*/
#define TABLE_SIZE (SIZEOF_H * SIZEOF_B * SIZEOF_TE * SIZEOF_NE) /*Total size of the table*/

#define SIZEOF_H 33 /*Size of H's in your cooling table*/
#define SIZEOF_B 33 /*Size of B's in your cooling table*/
#define SIZEOF_TE 33 /*Size of Te's in your cooling table*/
#define SIZEOF_NE 33/*Size of Ne's in your cooling table*/

#define SINGLE_VALUE (0)
/*Routine parameters for bessel/gamma functions*/
#define ITMAX (1e8)  
#define EPS (3e-7)
#define maxSteps (1e3)
#define FPMIN (1.0e-30) 
#define INTEGRATION_STEPS (1e6)


/*Declaration of both texture objects*/
cudaTextureObject_t coolTexObj;
cudaArray *cuCoolArray = 0;

cudaTextureObject_t coulombTexObj;
cudaArray *cuCoulombArray = 0;

__device__ double gammln(double xxgam);
/*Standard Numerical Recipes error function*/
__device__ void nrerror(char error_text[])
{
    printf("Numerical Recipes run-time error...\n");
    printf("%s\n",error_text);
    printf("...now exiting to system...\n");
    //exit(1);
}

/*Bessel0 function defined as Numerical Recipes book*/
__device__ double bessi0(double xbess)
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
__device__ double bessk0(double xbess)
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
__device__ double bessi1(double xbess)
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
__device__ double bessk1(double xbess)
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
__device__ double bessk2(double xbess)
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
__device__ void gcf(double *gammcf, double a, double x, double *gln)
{
    __device__ double gammln(double xx);
    __device__ void nrerror(char error_text[]);
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
    if (i > ITMAX)
    { 
        printf("a too large, ITMAX too small in gcf");
    }
        *gammcf=exp(-x+a*log(x)-(*gln))*h; //Put factors in front.
}

/*Returns the incomplete gamma function P(a, x) evaluated by its series representation as gamser. Also returns ln Γ(a) as gln.*/
__device__ void gser(double *gamser, double a, double x, double *gln)
{
    __device__ double gammln(double xx);
    __device__ void nrerror(char error_text[]);
    int n;
    double sum,del,ap;
    *gln=gammln(a);
    if (x <= 0.0) {
        if (x < 0.0) printf("x less than 0 in routine gser");
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
        printf("a too large, ITMAX too small in routine gser");
        return;
    }
}


/*Returns the incomplete gamma function P(a, x).*/
__device__ double gammp(double a, double x)
{
    __device__ void gcf(double *gammcf, double a, double x, double *gln);
    __device__ void gser(double *gamser, double a, double x, double *gln);
    __device__ void nrerror(char error_text[]);
    double gamser,gammcf,gln;
    if (x < 0.0 || a <= 0.0) printf("Invalid arguments in routine gammp");
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
__device__ double gammq(double a, double x)
{
    __device__ void gcf(double *gammcf, double a, double x, double *gln);
    __device__ void gser(double *gamser, double a, double x, double *gln);
    __device__ void nrerror(char error_text[]);
    double gamser,gammcf,gln;
    if (x < 0.0 || a <= 0.0) printf("Invalid arguments in routine gammq");
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
__device__ double gammln(double xx)
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
__device__ double thetae(double etemp)
{
    double result = (BOLTZ_CGS)* etemp / ((ERM_CGS) * (pow(C_CGS, 2.)));
    return result;
}

__device__ double trans_f(double scale_height, double x, double edens, double etemp, double mag_field)
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

__device__ double secant_bounded(double (*func)(double, double, double, double, double), double scale_height, double edens, double etemp, double mag_field)
//Using the secant method, find the root of a function func thought to lie between x1 and x2.
//The root, returned as rtsec, is refined until its accuracy is ±xacc.
{//All the function was checked step by step, seems to be working good.
    __device__ void nrerror(char error_text[]);
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
__device__ double bremmstrahlung_ee(double edens, double etemp)
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
__device__ double bremmstrahlung_ei(double edens, double etemp)
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
__device__ double bremmscooling_rate(double edens, double etemp)
{//All the function was checked step by step, seems to be working good.
    //printf(("brems\n");

    double result = bremmstrahlung_ee(edens, etemp) + bremmstrahlung_ei(edens, etemp);
    return result;
}

/*Critical frequency calculation for synchrotron cooling*/
__device__ double crit_freq(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //printf(("Crit Freq\n");

    return (1.5) * (2.80 * pow(10.,6.) * mag_field) * pow(thetae(etemp), 2.) * secant_bounded(trans_f,scale_height, edens, etemp, mag_field);
}

/*Synchtron radiation calculation*/
__device__ double rsync(double scale_height, double edens, double etemp, double mag_field)
{ //All the function was checked step by step, seems to be working good.
    //printf(("rsynch\n");
    double nuzero = 2.80 * pow(10., 6.) * mag_field;
    double a1 = 2. / (3. * nuzero * pow(thetae(etemp), 2.));
    double a2 = 0.4 / pow(a1, 1. / 4.);
    double a3 = 0.5316 / pow(a1, 1. / 2.);
    double a4 = 1.8899 * pow(a1, 1. / 3.);
    double criticalfrequency = crit_freq(scale_height, edens, etemp, mag_field);
    if (criticalfrequency == 0){
        printf("H = %le, B = %le, ne = %le, te = %le \n", scale_height, mag_field, edens, etemp);
    }
    if (thetae(etemp) > 0.5)
    {
        double self_abs = 2. * C_PI * BOLTZ_CGS * etemp * pow(criticalfrequency, 3.) / (3. * scale_height * pow(C_CGS, 2.));
        double init_term2 =6.76 * pow(10., -28.) * edens / (bessk2(1. / thetae(etemp)) * pow(a1, 1. / 6.));
        double syn_1 = 1. / pow(a4, 11. / 2.) *(exp(gammln(11./2.)))* gammq(11. / 2., a4 * pow(criticalfrequency, 1. / 3.));
        double syn_2 = a2 / pow(a4, 19. / 4.) *(exp(gammln(19./4.)))* gammq(19. / 4., a4 * pow(criticalfrequency, 1. / 3.));
        double syn_3 = a3 / pow(a4, 4.) * (pow(a4, 3.) * criticalfrequency + 3. * pow(a4, 2.) * pow(criticalfrequency, 2. / 3.) + 6. * a4 * pow(criticalfrequency, 1. / 3.) + 6.) * exp(-a4 * pow(criticalfrequency, 1. / 3.));
        double synchrotron = init_term2 * (syn_1+syn_2+syn_3);
        double result = self_abs + synchrotron;
        
        return result;
    }
    else
    {
        double self_abs = 2. * C_PI * BOLTZ_CGS * etemp * pow(criticalfrequency, 3.) / (3. * scale_height * pow(C_CGS, 2.));
        double init_term2 = 6.76 * pow(10., -28.) * edens / (2. * pow(thetae(etemp), 2.) * pow(a1, 1. / 6.));
        double syn_1 = 1. / pow(a4, 11. / 2.) *(exp(gammln(11./2.)))* gammq(11. / 2., a4 * pow(criticalfrequency, 1. / 3.));
        double syn_2 = a2 / pow(a4, 19. / 4.) *(exp(gammln(19./4.)))* gammq(19. / 4., a4 * pow(criticalfrequency, 1. / 3.));
        double syn_3 = a3 / pow(a4, 4.) * (pow(a4, 3.) * criticalfrequency + 3. * pow(a4, 2.) * pow(criticalfrequency, 2. / 3.) + 6. * a4 * pow(criticalfrequency, 1. / 3.) + 6.) * exp(-a4 * pow(criticalfrequency, 1. / 3.));
        double synchrotron = init_term2 * (syn_1+syn_2+syn_3);
        double result = self_abs + synchrotron;
        return result;
    }
}

/*Comptonization factor defined by Esin et al. (1996)*/
__device__ double comptonization_factor_sync(double scale_height, double edens, double etemp, double mag_field){
    //printf(("Comptonization_factor_sync\n");
	double thompson_opticaldepth = 2 * edens * THOMSON_CGS * scale_height;
	double Afactor = 1 + 4 * thetae(etemp) + 16 * pow(thetae(etemp), 2.);
	double maxfactor = 3 * BOLTZ_CGS * etemp/(PLANCK_CGS * crit_freq(scale_height, edens, etemp, mag_field));
	double jm = log(maxfactor)/log(Afactor);
	double s = thompson_opticaldepth + pow(thompson_opticaldepth, 2.);
    double factor_1 = (1 - gammp(jm + 1, Afactor * s));
    double factor_2 = maxfactor * gammp(jm +1, s);
    // printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
    // printf("O valor de Afactor é%le\n", Afactor);
    // printf("O valor de maxfactor é%le\n", maxfactor);
    // printf("O valor de jm é%le\n", jm);
    // printf("O valor de s é%le\n", s);
    // printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
    // printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
    if (factor_1 == 0){
        double result = factor_2;
        if (isnan(result)){
            // printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
            // printf("O valor de Afactor é%le\n", Afactor);
            // printf("O valor de maxfactor é%le\n", maxfactor);
            // printf("O valor de jm é%le\n", jm);
            // printf("O valor de s é%le\n", s);
            // printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
            // printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
            //exit(1);
        }
        return result;
    }
    else{
        double result = pow(M_E, s*(Afactor -1))*factor_1 + factor_2;
        if (isnan(result)){
            // printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
            // printf("O valor de Afactor é%le\n", Afactor);
            // printf("O valor de maxfactor é%le\n", maxfactor);
            // printf("O valor de jm é%le\n", jm);
            // printf("O valor de s é%le\n", s);
            // printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
            // printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
            //exit(1);
        }
        return result;
    }
}

__device__ double comptonization_factor_brems(double scale_height, double edens, double etemp, double nu){
    //printf(("Comptonized_factor brems\n");
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
            // printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
            // printf("O valor de Afactor é%le\n", Afactor);
            // printf("O valor de maxfactor é%le\n", maxfactor);
            // printf("O valor de jm é%le\n", jm);
            // printf("O valor de s é%le\n", s);
            // printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
            // printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
            // exit(1);
        }
        return result;
    }
    else{
        double result = pow(M_E, s*(Afactor -1))*factor_1 + factor_2;
        if (isnan(result)){
            // printf("O valor de thompson optical depth é%le\n", thompson_opticaldepth);
            // printf("O valor de Afactor é%le\n", Afactor);
            // printf("O valor de maxfactor é%le\n", maxfactor);
            // printf("O valor de jm é%le\n", jm);
            // printf("O valor de s é%le\n", s);
            // printf("O valor de gammp(As) é%le\n", gammp(jm + 1, Afactor * s));
            // printf("O valor de gammp(s) é%le\n", gammp(jm +1, s));
            // exit(1);
        }
        return result;
    }
}

__device__ double comptonized_brems(double scale_height, double edens, double etemp, double mag_field){
    //Considering equation found in https://www.astro.utu.fi/~cflynn/Stars/l6.html and https://www.astro.rug.nl/~etolstoy/radproc/resources/lectures/lecture9pr.pdf
    //Considering gaunt factor = 1, ni = ne and z = 1 for hydrogen gas.
    //printf(("Comptonized_brems\n");
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
__device__ double soptical_depth(double scale_height, double edens, double etemp)
{//All the function was checked step by step, seems to be working good.
    //printf(("soptical_depth\n");
    double result = 2. * edens * THOMSON_CGS * scale_height;
    return result;
}

/*Comptonization factor defined by Narayan & Yi (1995)*/
__device__ double comptonization_factor_ny(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //printf(("Comptonization_factor_ny\n");
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
__device__ double totalthincooling_rate(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //printf(("Total thin invoked\n");
    //double result = (comptonized_brems(scale_height, edens, etemp, mag_field) > 0? comptonized_brems(scale_height, edens, etemp, mag_field): bremmscooling_rate(edens, etemp)) + rsync(scale_height, edens, etemp, mag_field) * comptonization_factor_sync(scale_height, edens, etemp, mag_field);
    double result = bremmscooling_rate(edens, etemp) + rsync(scale_height, edens, etemp, mag_field) * comptonization_factor_sync(scale_height, edens, etemp, mag_field);
    return result;
}

/*Absorption optical depth*/
__device__ double absoptical_depth(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //printf(("Absorption\n");
    double result = 1. / (4. * C_sigma * pow(etemp, 4.)) * scale_height * totalthincooling_rate(scale_height, edens, etemp, mag_field);
    return result;
}

/*Total optical depth*/
__device__ double total_optical_depth(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //printf(("Total Optical depth invoked\n");
    double result = soptical_depth(scale_height, edens, etemp) + absoptical_depth(scale_height, edens, etemp, mag_field);
    return result;
}

/*Total cooling for both optically thin and thick regimes*/
__device__ double total_cooling(double scale_height, double edens, double etemp, double mag_field)
{//All the function was checked step by step, seems to be working good.
    //printf(("Total cooling invoked\n");
    return 4. * C_sigma * pow(etemp, 4.) / scale_height * 1 /
           (3 * total_optical_depth(scale_height, edens, etemp, mag_field) / 2. + pow(3., 1. / 2.) + 1. / absoptical_depth(scale_height, edens, etemp, mag_field));
           
}



/*This function loads the cooling values from the binary file*/
void Load_Cooling_Tables(float *cooling_table)
{
    fprintf(stderr, "Loading Table...\n");

    int i = 0;
    int nw = SIZEOF_H; //Number of H data
    int nx = SIZEOF_TE; // Number of Te data.
    int ny = SIZEOF_NE; // Number of ne data.
    int nz = SIZEOF_B;  // Number of Bmag data.

    FILE *infile;
    double value;

    // Reading the cooling table
    infile = fopen("cooling_table_33_05.bin", "rb");

    if (infile == NULL)
    {
        fprintf(stderr, "Unable to open cooling file.\n");
        exit(1);
    }
    fprintf(stderr, "Reading Data...\n");

    // Opening the binary file and reading the data, while also transferring it to the pointer cooling_table
    for (i = 0; i < nw * nx * ny * nz; i++)
    {
        fread(&value, sizeof(double), 1, infile);
        cooling_table[i] = float(value);
    }

    fprintf(stderr, "Finished transfering .binary data to memory!\n");
    fclose(infile);

    printf("Table Loaded!\n");

    return;
}

/*This function will transfer the values from the table to the texture object*/
void CreateTexture(void)
{
    float *cooling_table; //Pointer that will hold the cooling values from Load_cooling_table function
    const int nw = SIZEOF_H;  // Number of H data
    const int nx = SIZEOF_TE; // Number of Te data
    const int ny = SIZEOF_NE; // Number of ne data
    const int nz = SIZEOF_B;  // Number of Bmag data
    cooling_table = (float *)malloc(nw * nx * ny * nz * sizeof(float)); //Allocating memory for cooling_table pointer

    Load_Cooling_Tables(cooling_table); // Loading Cooling Values into pointer
    
    // cudaArray Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    // cuda Array
    cudaArray *cuCoolArray;

    //Creating 3D array in device memory  
    cudaMalloc3DArray(&cuCoolArray, &channelDesc, make_cudaExtent(nx * ny, nz, nw), 0);
    cudaMemcpy3DParms copyParams = {0};

    // Copying cooling values from host memory to device array.
    copyParams.srcPtr = make_cudaPitchedPtr((void *)cooling_table, nx * ny* sizeof(float), nx * ny, nz);
    copyParams.dstArray = cuCoolArray;
    copyParams.extent = make_cudaExtent(nx * ny, nz, nw);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    // Array creation End

    //Defining parameters for the texture object
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(texRes));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuCoolArray;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = false; //Whether to use normalized coordinates or not, this will impact the indexing
    texDescr.filterMode = cudaFilterModeLinear; // Whether to use nearest-neighbor approximation or trilinear interpolation
    texDescr.addressMode[0] = cudaAddressModeClamp; // Out of boundary conditions in dimension 1
    texDescr.addressMode[1] = cudaAddressModeClamp; // Out of boundary conditions in dimension 2
    texDescr.addressMode[2] = cudaAddressModeClamp; // Out of boundary conditions in dimension 3
    texDescr.readMode = cudaReadModeElementType; // Type of values stored in texture object

    cudaCreateTextureObject(&coolTexObj, &texRes, &texDescr, NULL); //Creating the texture object with the channel and parameters described above
    printf("Texture Created!\n");
    return;
}
void logspace(double start, double end, int num, double *result)
{
    double log_start = log10(start);                 // Initial value
    double log_end = log10(end);                     // End value
    double step = (log_end - log_start) / (num - 1); // number of steps
    int i;
    for (i = 0; i < num; ++i)
    {
        result[i] = log_start + i * step;
    }
}


__global__ void cooling_function_comparison_global(cudaTextureObject_t my_tex, double *a0, double *a1, double *a2, double *a3, double *value)
{
    
    double v0, v1, v4;
    double lambda;
    float a2_index, a3_index;

    double t_break = 9.472016;
    double t_ubreak = 9.540000;
    double t_lbreak = 9.410000;

    // double t_break = 9.472016;
    // double t_ubreak = 9.71875; //para 101
    // double t_lbreak = 9.3125; //para 101
    
    float alpha, beta, v4_ij, v4_i1j, v4_ij1, v4_i1j1, v4_i, v4_i1;
    float v4_ihalfj, v4_ihalfj1, v4_im1j1, v4_im1j, v4_iM1j1, v4_iM1j, v4_im2j1, v4_im2j, v4_iM2j1, v4_iM2j, frac_break, alpha_lower, alpha_upper;

    // For the normalized version only.
    const int nw = SIZEOF_H;  // Number of H used to generate table
    const int nx = SIZEOF_TE; // Number of te used to generate table
    const int ny = SIZEOF_NE; // Number of ne used to generate table
    const int nz = SIZEOF_B;  // Number of Bmag used to generate table
    for (int i = 0; i < SIZEOF_TEST; i++)
    {
        for (int j = 0; j < SIZEOF_TEST; j++)
        {
            for (int k = 0; k < SIZEOF_TEST; k++)
            {
                for (int l = 0; l < SIZEOF_TEST; l++)
                {
                    // Calculate both dimensions that are not flattened
                    v0 = ((((a0[i] - 3.) > 0 ? a0[i] - 3. : 0) * (nw - 1.) / 5.) + 0.5);
                    v1 = (((a1[j] - 0.) > 0 ? a1[j] : 0) * (nz - 1.) / 10. + 0.5);


                    // Select maximum values separetly
                    if (a2[k] > 25)
                    {
                        a2[k] = 25;
                    }
                    else if (a3[l] > 15)
                    {
                        a3[l] = 15;
                    }

                    // These will give us the indexing of B and ne from the table, we gotta see if they are integers or not.
                    a3_index = (((a3[l] - 2.) > 0 ? a3[l] - 2. : 0) * (nx - 1.) / 13.);
                    a2_index = (((a2[k] - 2.) > 0 ? a2[k] - 2. : 0) * (ny - 1.) / 23.);
            
                    if (a3_index == (int)a3_index && a2_index == (int)a2_index)
                    {
                        v4 = ((a3_index) + a2_index * (nx) + 0.5);
                        lambda = tex3D<float>(my_tex, v4, v1, v0);
                    }
                    else if (a3_index != (int)a3_index && a2_index != (int)a2_index)
                    {   
                        beta = a2_index - floor(a2_index);
                        alpha = a3_index - floor(a3_index);
                        if (a3[l] < t_break && a3[l] > t_lbreak){
                            frac_break = t_break - t_lbreak;
                            alpha_lower = (a3[l] - t_lbreak)/frac_break;
                            v4_ij = (floor(a3_index) + floor(a2_index) * (nx) + 0.5);
                            v4_ij1 = ((floor(a3_index)) + (floor(a2_index) + 1) * (nx) + 0.5);


                            v4_im2j =((floor(a3_index) - 2) + (floor(a2_index))  * (nx) + 0.5);
                            v4_im1j = ((floor(a3_index) - 1) + (floor(a2_index)) * (nx) + 0.5);
                            v4_ihalfj = 3 * tex3D<float>(my_tex, v4_ij, v1, v0) - 3 * tex3D<float>(my_tex, v4_im1j, v1, v0) + tex3D<float>(my_tex, v4_im2j, v1, v0);

                            v4_im2j1 =((floor(a3_index) - 2) + (floor(a2_index) + 1)  * (nx) + 0.5);
                            v4_im1j1 = ((floor(a3_index) - 1) + (floor(a2_index) + 1) * (nx) + 0.5);
                            v4_ihalfj1 = 3 * tex3D<float>(my_tex, v4_ij1, v1, v0) - 3 * tex3D<float>(my_tex, v4_im1j1, v1, v0) + tex3D<float>(my_tex, v4_im2j1, v1, v0);

                            lambda = (1 - alpha_lower) * (1 - beta) * tex3D<float>(my_tex, v4_ij, v1, v0) + alpha_lower * (1 - beta) * v4_ihalfj +
                                    (1 - alpha_lower) * beta * tex3D<float>(my_tex, v4_ij1, v1, v0) + alpha_lower * beta * v4_ihalfj1;                
                    

                        }else if(a3[l] >t_break && a3[l] < t_ubreak){//
                            frac_break = t_ubreak - t_break;
                            alpha_upper = (a3[l] - t_break)/(frac_break);
                            v4_ij = ((floor(a3_index) + 1) + floor(a2_index) * (nx) + 0.5);
                            v4_ij1 = ((floor(a3_index) + 1) + (floor(a2_index) + 1) * (nx) + 0.5);


                            v4_iM2j =((floor(a3_index) + 3) + (floor(a2_index))  * (nx) + 0.5);
                            v4_iM1j = ((floor(a3_index) + 2) + (floor(a2_index)) * (nx) + 0.5);
                            v4_ihalfj = 3 * tex3D<float>(my_tex, v4_ij, v1, v0) - 3 * tex3D<float>(my_tex, v4_iM1j, v1, v0) + tex3D<float>(my_tex, v4_iM2j, v1, v0);

                            v4_iM2j1 =((floor(a3_index) + 3) + (floor(a2_index) + 1)  * (nx) + 0.5);
                            v4_iM1j1 = ((floor(a3_index) + 2) + (floor(a2_index) + 1) * (nx) + 0.5);
                            v4_ihalfj1 = 3 * tex3D<float>(my_tex, v4_ij1, v1, v0) - 3 * tex3D<float>(my_tex, v4_iM1j1, v1, v0) + tex3D<float>(my_tex, v4_iM2j1, v1, v0);
                            lambda = (1 - alpha_upper) * (1 - beta) * v4_ihalfj + alpha_upper * (1 - beta) * tex3D<float>(my_tex, v4_ij, v1, v0) +
                                    (1 - alpha_upper) * beta * v4_ihalfj1 + alpha_upper * beta * tex3D<float>(my_tex, v4_ij1, v1, v0);  
                        }else{//
                            v4_ij = (floor(a3_index) + floor(a2_index) * (nx) + 0.5);
                            v4_i1j = ((floor(a3_index) + 1) + floor(a2_index) * (nx) + 0.5);
                            v4_ij1 = ((floor(a3_index)) + (floor(a2_index) + 1) * (nx) + 0.5);
                            v4_i1j1 = ((floor(a3_index) + 1) + (floor(a2_index) + 1) * (nx) + 0.5);
                            lambda = (1 - alpha) * (1 - beta) * tex3D<float>(my_tex, v4_ij, v1, v0) + alpha * (1 - beta) * tex3D<float>(my_tex, v4_i1j, v1, v0) +
                                    (1 - alpha) * beta * tex3D<float>(my_tex, v4_ij1, v1, v0) + alpha * beta * tex3D<float>(my_tex, v4_i1j1, v1, v0);
                        }
                    }
                    else if (a2_index != (int)a2_index) //Condition for indexne not integer and indexte being an integer
                    {//
                        alpha = a2_index - floor(a2_index);
                        v4_i = ((a3_index) + floor(a2_index) * (nx) + 0.5);
                        v4_i1 = ((a3_index) + (floor(a2_index) + 1) * (nx) + 0.5);
                        lambda = (1 - alpha) * tex3D<float>(my_tex, v4_i, v1, v0) + alpha * tex3D<float>(my_tex, v4_i1, v1, v0);
                    }
                    else //Condition for indexte not integer and indexne being an integer
                    {
                        alpha = a2_index - floor(a2_index);
                        if (a3[l] < t_break && a3[l] > t_lbreak){//
                            frac_break = t_break - t_lbreak;
                            alpha_lower = (a3[l] - t_lbreak)/frac_break;
                            v4_ij = (floor(a3_index) + (a2_index) * (nx) + 0.5);
                            v4_im2j =((floor(a3_index) - 2) + (floor(a2_index))  * (nx) + 0.5);
                            v4_im1j = ((floor(a3_index) - 1) + (floor(a2_index)) * (nx) + 0.5);
                            v4_ihalfj = 3 * tex3D<float>(my_tex, v4_ij, v1, v0) - 3 * tex3D<float>(my_tex, v4_im1j, v1, v0) + tex3D<float>(my_tex, v4_im2j, v1, v0);

                            lambda = (1 - alpha_lower) * tex3D<float>(my_tex, v4_ij, v1, v0) + alpha_lower * v4_ihalfj;
                            lambda = (1 - alpha) * tex3D<float>(my_tex, v4_ij, v1, v0) + alpha * v4_ihalfj;
                        }else if(a3[l] >t_break && a3[l] < t_ubreak){//
                            alpha_upper = (a3[l] - t_break)/(t_ubreak - t_break);
                            v4_ij = (floor(a3_index + 1) + floor(a2_index) * (nx) + 0.5);
                            v4_iM2j =((floor(a3_index) + 3) + (floor(a2_index))  * (nx) + 0.5);
                            v4_iM1j = ((floor(a3_index) + 2) + (floor(a2_index)) * (nx) + 0.5);
                            v4_ihalfj = 3 * tex3D<float>(my_tex, v4_ij, v1, v0) - 3 * tex3D<float>(my_tex, v4_iM1j, v1, v0) + tex3D<float>(my_tex, v4_iM2j, v1, v0);
                            lambda = (1 - alpha_upper) * v4_ihalfj + alpha_upper * tex3D<float>(my_tex, v4_i1j, v1, v0);

                        }else{//
                            alpha = a3_index - floor(a3_index);
                            v4_i = (floor(a3_index) + (a2_index) * (nx) + 0.5);
                            v4_i1 = ((floor(a3_index) + 1) + (a2_index) * (nx) + 0.5);
                            lambda = (1 - alpha) * tex3D<float>(my_tex, v4_i, v1, v0) + alpha * tex3D<float>(my_tex, v4_i1, v1, v0);
                        }
                    }
                    value[INDEX(i, j, k, l)] = lambda;
                    //printf("Lambda Texture = %le",lambda);
                }
            }
        }
    }
    return;
}

__global__ void cooling_function_analytical(double * a0, double * a1, double * a2, double * a3, double * value){
    for (int i = 0; i < SIZEOF_TEST; i++)
    {
        for (int j = 0; j < SIZEOF_TEST; j++)
        {
            for (int k = 0; k < SIZEOF_TEST; k++)
            {
                for (int l = 0; l < SIZEOF_TEST; l++)
                {
                    //printf("H = %le, B = %le, ne = %le, te = %le, cooling = %le \n", pow(10, a0[i]), pow(10, a1[j]), pow(10, a2[k]), pow(10,a3[l]), value[INDEX(i,j,k,l)]);
                    value[INDEX(i, j, k, l)] = total_cooling(pow(10, a0[i]), pow(10, a2[l]), pow(10, a3[l]), pow(10, a1[j]));
                    // printf("One of the roots is: %lf\n",secant_bounded(trans_f, pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor do thetae =:%.11e\n", thetae(pow(10, a3[l])));
                    // printf("O valor do bremmstrahlung ee =:%le\n", bremmstrahlung_ee(pow(10, a2[k]), pow(10, a3[l])));
                    // printf("O valor do bremmstrahlung ei =:%le\n", bremmstrahlung_ei(pow(10, a2[k]), pow(10, a3[l])));
                    // printf("O valor do bremmstrahlung total =:%le\n", bremmscooling_rate(pow(10, a2[k]), pow(10, a3[l])));
                    // //printf("o valor do comptonizaed brems =: %le\n", comptonized_brems(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor da freq crit =: %.11e\n", crit_freq(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor do rsync =: %le\n", rsync(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // //printf("o valor do comptonization factor_ny =: %.11e\n", comptonization_factor_ny(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor do comptonization factor =: %.11e\n", comptonization_factor_sync(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor do cooling total no disco fino =:%le\n", totalthincooling_rate(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("O valor do tau_scat =:%le\n", soptical_depth(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l])));
                    // printf("O valor do tau_abs =:%le\n", absoptical_depth(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("O valor do tau_total =:%le\n", total_optical_depth(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor do cooling total =:%le\n", total_cooling(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j])));
                    // printf("o valor do cooling total em log:%le\n", log10(total_cooling(pow(10, a0[i]), pow(10, a2[k]), pow(10, a3[l]), pow(10, a1[j]))));
                }
            }
        }
    } 
}

__global__ void cooling_function_single(double a0, double a1, double a2, double a3){
    a0 = 3;
    a1 = 2.1428;
    a2 = 18.428571;
    a3 = 11.285714;
    a0 = pow(10, a0);
    a1 = pow(10, a1);
    a2 = pow(10, a2);
    a3 = pow(10, a3);
    printf("H = %le, B = %le, ne = %le, te = %le \n", a0, a1, a2, a3);
    printf("One of the roots is: %lf\n",secant_bounded(trans_f, a0, a2, a3, a1));
    printf("o valor do thetae =:%.11e\n", thetae(a3));
    printf("O valor do bremmstrahlung ee =:%le\n", bremmstrahlung_ee(a2, a3));
    printf("O valor do bremmstrahlung ei =:%le\n", bremmstrahlung_ei(a2, a3));
    printf("O valor do bremmstrahlung total =:%le\n", bremmscooling_rate(a2, a3));
    //printf("o valor do comptonizaed brems =: %le\n", comptonized_brems(a0, a2, a3, a1));
    printf("o valor da freq crit =: %.11e\n", crit_freq(a0, a2, a3, a1));
    printf("o valor do rsync =: %le\n", rsync(a0, a2, a3, a1));
    printf("o valor do comptonization factor =: %.11e\n", comptonization_factor_sync(a0, a2, a3, a1));
    printf("o valor do cooling total no disco fino =:%le\n", totalthincooling_rate(a0, a2, a3, a1));
    printf("O valor do tau_scat =:%le\n", soptical_depth(a0, a2, a3));
    printf("O valor do tau_abs =:%le\n", absoptical_depth(a0, a2, a3, a1));
    printf("O valor do tau_total =:%le\n", total_optical_depth(a0, a2, a3, a1));
    printf("o valor do cooling total =:%le\n", total_cooling(a0, a2, a3, a1));
    printf("o valor do cooling total em log:%le\n", log10(total_cooling(a0, a2, a3, a1)));
}

int main(){
    double *H_random, *B_random, *ne_random, *Te_random;
    clock_t start_time, end_time;
    double duration;

    H_random = (double *)malloc(SIZEOF_TEST * sizeof(double));
    B_random = (double *)malloc(SIZEOF_TEST * sizeof(double));
    ne_random = (double *)malloc(SIZEOF_TEST * sizeof(double));
    Te_random = (double *)malloc(SIZEOF_TEST * sizeof(double));


    // Allocating memory in device memory.

    double *d_H_random;
    cudaMalloc(&d_H_random, SIZEOF_TEST * sizeof(double));
    double *d_B_random;
    cudaMalloc(&d_B_random, SIZEOF_TEST * sizeof(double));
    double *d_Ne_random;
    cudaMalloc(&d_Ne_random, SIZEOF_TEST * sizeof(double));
    double *d_Te_random;
    cudaMalloc(&d_Te_random, SIZEOF_TEST * sizeof(double));
    double *d_results;
    cudaMalloc(&d_results, SIZEOF_TEST * sizeof(double));

    fprintf(stderr,"Initializing analytical vs table speed test\n");

    logspace(1e3, 1e8, SIZEOF_TEST, H_random);
    logspace(1e0, 1e10, SIZEOF_TEST, B_random);
    logspace(1e2, 1e15, SIZEOF_TEST, Te_random);
    logspace(1e2, 1e25, SIZEOF_TEST, ne_random);
    fprintf(stderr,"Transfering data from Host to Device... \n");
    cudaMemcpy(d_H_random, H_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_random, B_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ne_random, ne_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Te_random, Te_random, SIZEOF_TEST * sizeof(double), cudaMemcpyHostToDevice);
    CreateTexture();

    fprintf(stderr,"Starting Analytical Calculation\n");
    start_time = clock();
    #if(SINGLE_VALUE)
        double H, ne, te, B;
        // printf("valor do scale_height\n");
        // scanf("%le", &H);
        // printf ("valor do B\n");
        // scanf("%le", &B);
        // printf("Valor de edens\n");
        // scanf("%le", &ne);
        // printf ("valor de etemp\n");
        // scanf("%le", &te);
        cooling_function_single<<<1,1>>>(H, B, ne, te);
        cudaDeviceSynchronize();

    #endif
    //cooling_function_analytical<<<1, 1>>>(d_H_random, d_B_random, d_Ne_random, d_Te_random, d_results);
    cudaDeviceSynchronize();
    end_time = clock();
    duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    fprintf(stderr,"Number of values analyzed = %d, duration global: %.6f seconds\n", SIZEOF_TEST, duration);
    fprintf(stderr, "Starting texture memory calculations");
    start_time = clock();
    cooling_function_comparison_global<<<1, 1>>>(coolTexObj, d_H_random, d_B_random, d_Ne_random, d_Te_random, d_results);
    cudaDeviceSynchronize();
    end_time = clock();
    duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    fprintf(stderr,"Number of values analyzed = %d, duration texture: %.6f seconds\n", SIZEOF_TEST, duration);
}
