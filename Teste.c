#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* functions*/
double bessi0(double xbess); /* Calculation of the first kind of bessel modified function for n = 0 */ 
double bessi1(double xbess); /* Calculation of the first kind of bessel modified function for n = 1 */
double bessk0(double xbess); /* Calculation of the second kind of bessel modified function for n = 0 */
double bessk1(double xbess); /* Calculation of the second kind of bessel modified function for n = 1 */
double bessk2(double xbess); /* Calculation of the second kind of bessel modified function for n = 2 */
void gcf(double *gammcf, double agam, double xgam, double *gln); /*Function used in the calculation of incomplete gamma function */
void gser(double *gamser, double agam, double xgam, double *gln); /*Function used in the calculation of incomplete gamma function */
double gammp(double a, double xgam); /* Calculation of lower incomplete gamma function*/

double gammq(double a, double xgam); /* Calculation of upper incomplete gamma function*/

double gammln(double xxgam); /* Calculation of the logarithm of the gamma function */

double sound_speed(double ne);

double thetae (double Te);

double scale_height(double R, double ne);

double bremmstrahlung_ee (double ne, double Te);

double bremmstrahlung_ei (double ne, double Te);

double solve_eq_xm(double ne, double Te); /*Solves the transcedental equation for xm in synchrotron radiation in cooling as described by Esin 1996, 1996ApJ...465..312E*/

double crit_freq(double ne, double Te); /*Calculates the critical frequency in synchrotron where below that frequency, radiation is self absorbed as described by Esin 1996, 1996ApJ...465..312E*/

double rsync(double ne, double Te); /*Calculates the synchrotron radiation energy density*/
double bremmscooling_rate (double ne, double Te);
double totalcooling_rate(double ne, double Te);

/*variables*/ 
double Te; /*temperature of the electrons*/
double Bmag(double ne); /*magnetic field module value*/
double ne; /*number density of the electron*/
double R; /*radius*/

/*constants in CGS*/
#define C_sigma  5.67051e-5	/*Stephan Boltzmann constant*/
#define BOLTZ_CGS  1.3806505e-16		/*Boltzmann constant*/
#define PLANCK_CGS 6.62606876e-27		/*Planck constant*/
#define ERM_CGS  9.1093826e-28		/*Electron rest mass CGS*/
#define C_mp  1.67262171e-24		/*Proton mass*/
#define C_amu  1.66053886e-24	/*Atomic mass unit*/
#define C_CGS  2.99792458e10		/*Speed of light*/
#define C_G  6.6726e-8		/*Gravitational constant*/
#define C_Msun  2.e33		/*Sun mass*/
#define THOMSON_CGS  6.6524e-25		/*Thomson cross section*/
#define C_pi  3.14159265358979	/*value of pi*/
#define C_euler  2.71828182845904   /* euler constant*/
#define C_gamma 5./3.             /*Adiabatic index*/
#define beta  10. 		/* value of beta factor*/
#define Rs 2. * C_G * C_Mbh / (pow(C_CGS, 2.))

/*Mass of the Black hole*/
#define C_Mbh  10. * C_Msun
#define C_GM  C_Mbh * C_G

/*Disk geometry and polytropic constant*/
#define rho_0 5e-7				/*Maximum density of initial condition*/
#define r_0 100. * Rs 			/*Radius of maximum density (rho_0)*/
#define r_min  75. * Rs 			/*Minimum raius of torus*/
#define CONST_2 -C_GM/(r_min-Rs) + C_GM/(2.*pow(r_min, 2.))*(pow(r_0, 3.)/(pow((r_0-Rs), 2))
#define kappa (C_gamma-1.)/C_gamma*pow(rho_0, 1.-C_gamma)*(CONST_2 + C_GM/(r_0-Rs) - C_GM/2. * r_0/(pow(r_0-Rs, 2.))))


#define ITMAX 100 /* Used in calculating gamma function*/
#define EPS 3.0e-7 /* Used in calculating gamma function*/
#define FPMIN 1.0e-30 /* Used in calculating gamma function*/

/***********************************************************************************************/
double bessi0(double xbess){
	double ax,ans;
	double y; 
	if ((ax=fabs(xbess)) < 3.75) {
		y=xbess/3.75;
		y*=y;
		ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
			+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
	} else {
		y=3.75/ax;
		ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
			+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
			+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
			+y*0.392377e-2))))))));
	}
	return ans;
}


double bessk0(double xbess){
	double y,ans;
	if (xbess <= 2.0) { 
		y=xbess*xbess/4.0;
		ans=(-log(xbess/2.0)*bessi0(xbess))+(-0.57721566+y*(0.42278420
			+y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
			+y*(0.10750e-3+y*0.74e-5))))));
	} else {
		y=2.0/xbess;
		ans=(exp(-xbess)/sqrt(xbess))*(1.25331414+y*(-0.7832358e-1
			+y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
			+y*(-0.251540e-2+y*0.53208e-3))))));
	}
	return ans;
}


double bessi1(double xbess){
	double ax,ans;
	double y; 
	if ((ax=fabs(xbess)) < 3.75) { 
		y=xbess/3.75;
		y*=y;
		ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
			+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
	} else {
		y=3.75/ax;
		ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
			-y*0.420059e-2));
		ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
			+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
		ans *= (exp(ax)/sqrt(ax));
	}
	return xbess < 0.0 ? -ans : ans;
}


double bessk1(double xbess){
	double y,ans; 
	if (xbess <= 2.0) { 
		y=xbess*xbess/4.0;
		ans=(log(xbess/2.0)*bessi1(xbess))+(1.0/xbess)*(1.0+y*(0.15443144
			+y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
			+y*(-0.110404e-2+y*(-0.4686e-4)))))));
	} else {
		y=2.0/xbess;
		ans=(exp(-xbess)/sqrt(xbess))*(1.25331414+y*(0.23498619
			+y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
			+y*(0.325614e-2+y*(-0.68245e-3)))))));
	}
	return ans;
}


double bessk2(double xbess){
	int n,j;
	double bk,bkm,bkp,tox;
	n = 2;
	tox=2.0/xbess;
	bkm=bessk0(xbess); 
	bk=bessk1(xbess);
	for (j=1;j<n;j++) { 
		bkp=bkm+j*tox*bk;
		bkm=bk;
		bk=bkp;
		}
	return bk;
}

void gcf(double *gammcf, double agam, double xgam, double *gln){
	int i;
	double an,b,c,d,del,h;
	*gln=gammln(agam);
	b=xgam+1.0-agam;
	c=1.0/FPMIN;
	d=1.0/b;
	h=d;
	for (i=1;i<=ITMAX;i++) { 
		an = -i*(i-agam);
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
	*gammcf=exp(-xgam+agam*log(xgam)-(*gln))*h; 
}

void gser(double *gamser, double agam, double xgam, double *gln)
{
	int n;
	double sum,del,ap;
	
	*gln=gammln(agam);
	if (xgam <= 0.0) {
		if (xgam < 0.0);
		*gamser=0.0;
		return;
	} else {
	ap=agam;
	del=sum=1.0/agam;
	for (n=1;n<=ITMAX;n++) {
		++ap;
		del *= xgam/ap;
		sum += del;
		if (fabs(del) < fabs(sum)*EPS) {
			*gamser=sum*exp(-xgam+agam*log(xgam)-(*gln));
			return;
		}
	}
	return;
	}
}

double gammp(double agam, double xgam)

{
	double gamser,gammcf,gln;
	if (xgam < (agam+1.0)) { 
		gser(&gamser,agam,xgam,&gln);
		return gamser;
	} else { 
		gcf(&gammcf,agam,xgam,&gln);
		return 1.0-gammcf; 
	}
}

double gammq(double agam, double xgam)
{
	double gamser,gammcf,gln;
	if (xgam < (agam+1.0)) { 
		gser(&gamser,agam,xgam,&gln);
		return 1.0-gamser; 
	} else { 
		gcf(&gammcf,agam,xgam,&gln);
		return gammcf;
	}
}

double gammln(double xxgam)
{
	double x,y,tmp,ser;
	static double cof[6]={76.18009172947146,-86.50532032941677,
	24.01409824083091,-1.231739572450155,
	0.1208650973866179e-2,-0.5395239384953e-5};
	int j;
	y=x=xxgam;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<=5;j++) ser += cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
}


/***********************************************************************************************/
double thetae (double Te){
	double result = BOLTZ_CGS * Te/(ERM_CGS * pow (C_CGS, 2.));
	return result;
}

double sound_speed(double ne){
	double result = pow(C_gamma * kappa * pow(ne * 1.14 * C_amu, C_gamma-1.), 1./2.);
	return result;
}

double Bmag (double ne){
	double result = pow(8.*C_pi*pow(sound_speed(ne), 2.)*ne*1.14*C_amu/(beta+1.), 1./2.);
	return result;
	
}
double scale_height(double R, double ne){
	double result = pow(R/C_GM, 1./2.)*sound_speed(ne)*(R-Rs);
	return R;
}

double bremmstrahlung_ee (double ne, double Te){
	double th_e = thetae(Te);
	double result;
	if (th_e < 1) {
		result = 2.56 * pow(10., -22.) * pow(ne, 2.) * pow(thetae(Te), 3./2.) * (1 + 1.10 * thetae(Te) + pow(thetae(Te), 2.) - 1.25 * pow(thetae(Te), 5./2.));
		
	} else {
		result = 3.40 * pow(10., -22.) * pow(ne, 2.) * thetae(Te) * (log(1.123 * thetae(Te)) + 1.28);
	}
	return result;
}

double bremmstrahlung_ei (double ne, double Te){
	double th_e = thetae(Te);
	double result;
	if (th_e < 1) {
		result = 1.48 * pow(10., -22.) * pow(ne, 2.) * (9*thetae(Te))/(2 * C_pi) * (log(1.123*thetae(Te) + 0.48) + 1.5);
		
	} else {
		result = 1.48 * pow(10., -22.) * pow(ne, 2.) * 4 * pow((2* thetae(Te)/pow(C_pi,3.)), 1./2.) * (1 + 1.781 * pow(thetae(Te), 1.34));
	}
	return result;
}

double transcedentalxm (double x, double ne, double Te){
	/*double resposta = pow(C_euler, 1.8899 * pow(x, 1./3.)) - 2.49* pow(10., -10.) * 12. * C_pi * ne * scale_height(R, ne)/(Bmag(ne)) * 1/(pow(thetae(Te), 3.) * bessk2(1/thetae(Te))) * (1/pow(x, 7./6.) + 0.4/pow(x, 17./12.) + 0.5316/pow(x, 5./3.));*/
	double resposta = pow(C_euler, 1.8899 * pow(x, 1./3.)) - 2.49* pow(10., -10.) * 12. * C_pi * ne * scale_height(R, ne)/(Bmag(ne)) * 1/(2*pow(thetae(Te), 5.)) * (1/pow(x, 7./6.) + 0.4/pow(x, 17./12.) + 0.5316/pow(x, 5./3.));
	
    return resposta;
}
double transcedentalxmderivative (double x, double ne, double Te){
	double resposta = 0.629967 * pow(C_euler, 1.8899 * pow(x, 1./3.))/pow(x, 2./3.) - 2.49* pow(10., -10.) * 12. * C_pi * ne * scale_height(R, ne)/(Bmag(ne)) * 1/(pow(thetae(Te), 3.) * bessk2(1/thetae(Te))) * (- 1.6667 * (pow(x, 1./2.) + 0.485714 * pow(x, 1./4.) + 0.759429/pow(x, 8./3.)));
    return resposta;
}
double solve_eq_xm(double ne, double Te)
{
    int itr;
    double h, x1;
    double x0 = 200.; /* initial value*/
    double allerr = pow(10, -5.); /* allowed error*/
    double maxmitr = pow(10, 5.); /* maximum iterations*/

    for (itr=1; itr<=maxmitr; itr++)
    {
        h=transcedentalxm(x0, ne, Te)/transcedentalxmderivative(x0, ne, Te);
        x1=x0-h;
        /*printf(" At Iteration no. %3d, x = %9.6f\n", itr, x1);*/
        if (fabs(h) < allerr)
        {
            /*printf("After %3d iterations, root = %8.6f\n", itr, x1);*/
            return x1;
        }
        x0=x1;
    }
    /*printf(" The required solution does not converge or iterations are insufficient\n")*/;
    return 1;
}


double crit_freq (double ne, double Te){
	double bsq = pow(Bmag(ne), 2.);
	double result;
	double nuzero = 2.80 * pow(10., 6.) * Bmag(ne);
	result = 3/2 * nuzero * pow(thetae(Te), 2.) * solve_eq_xm(ne, Te);
	return result;
}

/*Synchtron radiation calculation*/
double rsync (double ne, double Te){
	double bsq = pow(Bmag(ne), 2.);
	double nuzero = 2.80 * pow(10., 6.) * pow(bsq, 1./2.);
	double a1 = 2/(3 * nuzero * pow(thetae(Te), 2.));
	double a2 = 0.4/pow(a1, 1./4.);
	double a3 = 0.5316/pow(a1, 1./2.) ;
	double a4 = 1.8899 *pow(a1, 1./3.);
	/*double result = 2 * C_pi * BOLTZ_CGS * Te * pow(crit_freq(ne, Te), 3.)/(3 * scale_height(R, ne) * pow(C_CGS, 2.)) + 6.76 * pow(10., -28. ) * ne/(bessk2(1/thetae(Te)) * pow(a1, 1./6.)) * (1/pow(a4, 11./2.) * gammq(11./2., a4 * pow(crit_freq(ne, Te), 1./3.)) + a2/pow(a4, 19./4.) * gammq(19./4., a4 * pow(crit_freq(ne, Te), 1./3.)) + a3/pow(a4, 4.) * (pow(a4, 3.) * crit_freq(ne, Te) + 3 * pow(a4, 2.) * pow(crit_freq(ne, Te), 2./3.) + 6 * a4 * pow(crit_freq(ne, Te), 1./3.) + 6) * pow(C_euler, -a4 * pow(crit_freq(ne, Te), 1./3.)));*/
	double result = 2 * C_pi * BOLTZ_CGS * Te * pow(crit_freq(ne, Te), 3.)/(3 * scale_height(R, ne) * pow(C_CGS, 2.)) + 6.76 * pow(10., -28. ) * ne/(2*pow(thetae(Te), 2.) * pow(a1, 1./6.)) * (1/pow(a4, 11./2.) * gammq(11./2., a4 * pow(crit_freq(ne, Te), 1./3.)) + a2/pow(a4, 19./4.) * gammq(19./4., a4 * pow(crit_freq(ne, Te), 1./3.)) + a3/pow(a4, 4.) * (pow(a4, 3.) * crit_freq(ne, Te) + 3 * pow(a4, 2.) * pow(crit_freq(ne, Te), 2./3.) + 6 * a4 * pow(crit_freq(ne, Te), 1./3.) + 6) * pow(C_euler, -a4 * pow(crit_freq(ne, Te), 1./3.)));
	return result;
}

double comptonization_factor (double ne, double Te){
	double bsq = pow(Bmag(ne), 2.);
	double thompson_opticaldepth = 2 * ne * THOMSON_CGS * Te;
	double Afactor = 1 + 4 * thetae(Te) + 16 * pow(thetae(Te), 2.);
	double nuzero = 2.80 * pow(10., 6.) * pow(bsq, 1./2.);
	double maxfactor = 3 * BOLTZ_CGS * Te/(PLANCK_CGS * crit_freq(ne, Te));
	double jm = log(maxfactor)/log(Afactor);
	double s = thompson_opticaldepth + pow(thompson_opticaldepth, 2.);
	double result = pow(C_euler, s*(Afactor -1))*(1 - gammp(jm - 1, Afactor * s)) + maxfactor * gammp(jm +1, s);
	
	return result;
		
}

double bremmscooling_rate (double ne, double Te){
	double result = bremmstrahlung_ee(ne, Te) + bremmstrahlung_ei(ne, Te);
	return result;
}

double totalcooling_rate (double ne, double Te){
	double result = bremmstrahlung_ee(ne, Te) + bremmstrahlung_ei(ne, Te) + rsync(ne, Te) * comptonization_factor(ne, Te);
	return result;
}

int main (double ne, double Te)
{
	printf("valor de R\n");
	scanf("%lf", &R);
	printf("Valor de ne\n");
	scanf("%lf", &ne);
	printf ("valor de Te\n");
	scanf("%lf", &Te);
	printf("O valor do bremmstrahlung cooling rate é:%lf\n", bremmscooling_rate(ne, Te));
	printf("o valor do cooling total é:%lf\n", totalcooling_rate(ne, Te));
	printf("o valor do thetae é:%lf\n", thetae(Te));
	printf("o valor do rsync é: %lf\n", rsync(ne, Te));
	printf("o valor do comptonization factor é: %lf\n", comptonization_factor(ne, Te));
	printf("o valor do xm é: %lf\n", solve_eq_xm(ne, Te));
	printf("o valor da freq crit é: %lf\n", crit_freq(ne, Te));
	printf("o valor de Bmag é: %lf\n", Bmag(ne));
	return 0;
}
		
