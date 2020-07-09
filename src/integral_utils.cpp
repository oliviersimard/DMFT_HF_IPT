#include "integral_utils.hpp"

const unsigned int Integrals::_maxLevel = 20;  // tweek this parameter to get best results with best efficiency.
const unsigned int Integrals::_minLevel = 5;

double Integrals::coarse_app(std::function< double(double) > f, double a, double b) const{
    return (b-a) * (f(a)-f(b))/2.0;
}

std::complex<double> Integrals::coarse_app(std::function< std::complex<double>(double,std::complex<double>) > f, double a, double b, std::complex<double> iwn) const{
    return (b-a) * (f(a,iwn)-f(b,iwn))/2.0;
}

double Integrals::trap_app(std::function< double(double) > f, double a, double b) const{
    double m = (a+b)/2.0;
    return (b-a)/4.0 * (f(a)+2.0*f(m)+f(b));
}

double Integrals::simps_app(std::function< double(double) > f, double a, double b) const{
    double dx = (b-a)/2.0;
    double m = (b+a)/2.0;
    return dx/3.0 * (f(a)+4.0*f(m)+f(b));
}

std::complex<double> Integrals::simps_app(std::function< std::complex<double>(double,std::complex<double>) > f, double a, double b, std::complex<double> iwn) const{
    double dx = (b-a)/2.0;
    double m = (b+a)/2.0;
    return dx/3.0 * (f(a,iwn)+4.0*f(m,iwn)+f(b,iwn));
}

double Integrals::trap(std::function< double(double) > f, double a, double b, double tol, unsigned int currentlevel) const{
    double q = trap_app(f,a,b);
    double r = coarse_app(f,a,b);
    if ( (currentlevel>=_minLevel) && (std::abs(q-r)<=1.0*tol) ){
        return q;
    }
    else if (currentlevel>=_maxLevel) {
        std::cout << "Busted the maximum number of iterations allowed!" << std::endl;
        return q;
    }
    else{
        ++currentlevel;
        return ( trap(f,a,(a+b)/2.0,tol,currentlevel) + trap(f,(a+b)/2.0,b,tol,currentlevel) );
    }
}

double Integrals::simps(std::function< double(double) > f, double a, double b, double tol, unsigned int currentlevel) const{
    double q = simps_app(f,a,b);
    double r = coarse_app(f,a,b);
    //std::cout << currentlevel << "\n";
    if ( (currentlevel >= _minLevel) && (std::abs(q-r)<=1.0*tol) ){
        return q;
    }
    else if (currentlevel >= _maxLevel) {
        //std::cout << "Busted the maximum number of iterations allowed!" << q << std::endl;
        return q;
    }
    else{
        ++currentlevel;
        return ( simps(f,a,(a+b)/2.0,tol,currentlevel) + simps(f,(a+b)/2.0,b,tol,currentlevel) );
    }
}

std::complex<double> Integrals::simps(std::function< std::complex<double>(double,std::complex<double>) > funct,double a,double b,std::complex<double> iwn,double tol,unsigned int curr_level) const{
    std::complex<double> q = simps_app(funct,a,b,iwn);
    std::complex<double> r = coarse_app(funct,a,b,iwn);

    if ( (curr_level>=_minLevel) && (std::abs(q-r)<tol) ){
        return q;
    }
    else if (curr_level >= _maxLevel){
        std::cout << "Busted the maximum number of iterations allowed!" << q << std::endl;
        return q;
    }
    else{
        ++curr_level;
        return ( simps(funct,a,(a+b)/2.0,iwn,tol,curr_level) + simps(funct,(a+b)/2.0,b,iwn,tol,curr_level) );
    }
}

double Integrals::integrate_trap(std::function< double(double) > f, double a, double b, double tol) const{
    return trap(f,a,b,tol,1); // currentlevel starts with number 1, without much surprise.
}

double Integrals::integrate_simps(std::function< double(double) > f, double a, double b, double tol) const{
    return simps(f,a,b,tol,1);
}

std::complex<double> Integrals::integrate_simps(std::function< std::complex<double>(double,std::complex<double>) > funct,double a,double b,std::complex<double> iwn,double tol) const{
    return simps(funct,a,b,iwn,tol,1);
}

std::complex<double> Integrals::I2D(std::function<std::complex<double>(double,double,std::complex<double>)> funct,double x0,double xf,double y0,double yf,std::complex<double> iwn,std::string rule,double tol,unsigned int maxVal,bool is_converged) const noexcept(false){
    double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
    std::complex<double> result, prevResult(0.0,0.0), resultSumNyorNx, resultSumNxNy;
    unsigned int iter=0;
    while (!is_converged && iter<MAX_ITER_INTEGRAL){
        resultSumNyorNx=std::complex<double>(0.0,0.0), resultSumNxNy=std::complex<double>(0.0,0.0);
        dx=(xf-x0)/(double)maxVal;
        dy=(yf-y0)/(double)maxVal;
        prevResult=result;

        if (rule.compare("trapezoidal")==0){
            for (unsigned int i=1; i<maxVal; i++){
                resultSumNyorNx+=funct(x0+i*dx,y0,iwn)+funct(x0+i*dx,yf,iwn)+funct(x0,y0+i*dy,iwn)+funct(xf,y0+i*dy,iwn);
            }
            for (unsigned int i=1; i<maxVal; i++){
                for (unsigned int j=1; j<maxVal; j++){
                    resultSumNxNy+=funct(x0+i*dx,y0+j*dy,iwn);
                }
            }
        
            result=0.25*dx*dy*(funct(x0,y0,iwn)+funct(xf,y0,iwn)+funct(x0,yf,iwn)+funct(xf,yf,iwn))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;
        } else if (rule.compare("simpson")==0){
            for (unsigned int i=1; i<maxVal; i+=2){
                for (unsigned int j=1; j<maxVal; j+=2){
                    resultSumNxNy+=funct(x0+(i-1)*dx,y0+(j-1)*dy,iwn)+4.0*funct(x0+i*dx,y0+(j-1)*dy,iwn)+funct(x0+(i+1)*dx,y0+(j-1)*dy,iwn)+
                    4.0*funct(x0+(i-1)*dx,y0+j*dy,iwn)+16.0*funct(x0+i*dx,y0+j*dy,iwn)+4.0*funct(x0+(i+1)*dx,y0+j*dy,iwn)+
                    funct(x0+(i-1)*dx,y0+(j+1)*dy,iwn)+4.0*funct(x0+i*dx,y0+(j+1)*dy,iwn)+funct(x0+(i+1)*dx,y0+(j+1)*dy,iwn);
                }
            }

            result=1.0/9.0*dx*dy*resultSumNxNy;
        } else{
            throw std::invalid_argument("Unhandled rule chosen in integrals 2D. Choices are \"trapezoidal\" and \"simpson\"");
        }
        
        if (iter>0)
            is_converged = ( ( (std::abs(prevResult-result))>tol ) ) ? false : true;

        if (is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
}

double Integrals::I2D(std::function<double(double,double)> funct,double x0,double xf,double y0,double yf,std::string rule,double tol,unsigned int maxVal,bool is_converged) const noexcept(false){
    double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
    double resultSumNyorNx=0.0, resultSumNxNy=0.0, result, prevResult=0.0;
    unsigned int iter=0;
    while (!is_converged){
        resultSumNxNy=resultSumNyorNx=0.0;
        dx=(xf-x0)/(double)maxVal;
        dy=(yf-y0)/(double)maxVal;
        prevResult=result;

        if (rule.compare("trapezoidal")==0){
            for (unsigned int i=1; i<maxVal; i++){
                resultSumNyorNx+=funct(x0+i*dx,y0)+funct(x0+i*dx,yf)+funct(x0,y0+i*dy)+funct(xf,y0+i*dy);
            }
            for (unsigned int i=1; i<maxVal; i++){
                for (unsigned int j=1; j<maxVal; j++){
                    resultSumNxNy+=funct(x0+i*dx,y0+j*dy);
                }
            }

            result=0.25*dx*dy*(funct(x0,y0)+funct(xf,y0)+funct(x0,yf)+funct(xf,yf))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;
        } else if (rule.compare("simpson")==0){
            for (unsigned int i=1; i<maxVal; i+=2){
                for (unsigned int j=1; j<maxVal; j+=2){
                    resultSumNxNy+=funct(x0+(i-1)*dx,y0+(j-1)*dy)+4.0*funct(x0+i*dx,y0+(j-1)*dy)+funct(x0+(i+1)*dx,y0+(j-1)*dy)+
                    4.0*funct(x0+(i-1)*dx,y0+j*dy)+16.0*funct(x0+i*dx,y0+j*dy)+4.0*funct(x0+(i+1)*dx,y0+j*dy)+
                    funct(x0+(i-1)*dx,y0+(j+1)*dy)+4.0*funct(x0+i*dx,y0+(j+1)*dy)+funct(x0+(i+1)*dx,y0+(j+1)*dy);
                }
            }

            result=1.0/9.0*dx*dy*resultSumNxNy;
        } else{
            throw std::invalid_argument("Unhandled rule chosen in integrals 2D. Choices are \"trapezoidal\" and \"simpson\"");
        }

        if (iter>0)
            is_converged = (std::abs(prevResult-result)<tol) ? true : false;
        
        if (iter>MAX_ITER_INTEGRAL || is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
}

std::complex<double> Integrals::I1D(std::function<std::complex<double>(double,std::complex<double>)> funct,double k0,double kf,std::complex<double> iwn,std::string rule,double tol,unsigned int maxDelta) const noexcept(false){
    double dk;
    std::complex<double> result(0.0,0.0), prevResult(0.0,0.0), resultSumNk;
    unsigned int iter=0;
    bool is_converged=false;
    while(!is_converged && iter<MAX_ITER_INTEGRAL){
        resultSumNk = std::complex<double>(0.0,0.0);
        prevResult = result;
        dk=(kf-k0)/(double)maxDelta;

        if (rule.compare("trapezoidal")==0){
            for (unsigned int i=1; i<maxDelta; i++){
                resultSumNk+=funct(k0+i*dk,iwn);
            }
            result = 0.5*dk*funct(k0,iwn)+0.5*dk*funct(kf,iwn)+dk*resultSumNk;
        } else if (rule.compare("simpson")==0){
            assert(maxDelta%2==0); // The interval has to be devided into an even number of intervals for this to work in this scheme
            for (unsigned int i=1; i<maxDelta; i+=2){
                resultSumNk += funct(k0+(i-1)*dk,iwn) + 4.0*funct(k0+i*dk,iwn) + funct(k0+(i+1)*dk,iwn);
            }
            result = dk/3.0*resultSumNk;
        } else{
            throw std::invalid_argument("Unhandled rule chosen in integrals 1D. Choices are \"trapezoidal\" and \"simpson\"");
        }

        if (iter>0)
            is_converged = (std::abs(prevResult-result)>tol && iter>1) ? false : true;

        if (is_converged)
            break;

        maxDelta*=2;
        iter++;
    }
    
    return result;
}

double Integrals::I1D(std::function<double(double)> funct,double k0,double kf,std::string rule,double tol,unsigned int maxDelta) const noexcept(false){
    double dk=(kf-k0)/(double)maxDelta;
    double result = 0.0, prevResult = 0.0, resultSumNk;
    unsigned int iter=0;
    bool is_converged=false;
    while(!is_converged && iter<MAX_ITER_INTEGRAL){
        resultSumNk = 0.0;
        prevResult = result;
        dk=(kf-k0)/(double)maxDelta;

        if (rule.compare("trapezoidal")==0){
            for (unsigned int i=1; i<maxDelta; i++){
                resultSumNk+=funct(k0+i*dk);
            }
            result = 0.5*dk*funct(k0)+0.5*dk*funct(kf)+dk*resultSumNk;
        } else if (rule.compare("simpson")==0){
            assert(maxDelta%2==0); // The interval has to be devided into an even number of intervals for this to work in this scheme
            for (unsigned int i=1; i<maxDelta; i+=2){
                resultSumNk += funct(k0+(i-1)*dk) + 4.0*funct(k0+i*dk) + funct(k0+(i+1)*dk);
            }
            result = dk/3.0*resultSumNk;
        } else{
            throw std::invalid_argument("Unhandled rule chosen in integrals 1D. Choices are \"trapezoidal\" and \"simpson\"");
        }

        if (iter>0)
            is_converged = (std::abs(prevResult-result)>tol) ? false : true;

        if (is_converged)
            break;

        maxDelta*=2;
        iter++;
    }
    
    return result;
}

std::complex<double> cbrt(std::complex<double> num){
    // computing the cubic root of a complex number using exponential representation of complex number
    double modulus = std::sqrt((num*std::conj(num)).real());
    double angle = std::atan(num.imag()/num.real());

    std::complex<double> cbrt_num = std::cbrt(modulus)*std::exp(std::complex<double>(0.0,1.0/3.0)*angle);

    return cbrt_num;
}

cubic_roots get_cubic_roots(double a, double b, double c, double d){
    // solves eqaution of the form a*x^3 + b*x^2 + c*x + d = 0 using Cardano's formula.
    // Info: https://proofwiki.org/wiki/Cardano%27s_Formula
    constexpr std::complex<double> im(0.0,1.0);
    std::complex<double> S, T;
    double Q = (3.0*a*c-b*b)/(9.0*a*a);
    double R = (9.0*a*b*c-27.0*a*a*d-2.0*b*b*b)/(54.0*a*a*a);
    double D = Q*Q*Q + R*R;
    //std::cout << "The determinant is " << D << std::endl;
    if (D<0){ // all roots are real
        S = cbrt(R+im*std::sqrt(std::abs(D)));
        T = cbrt(R-im*std::sqrt(std::abs(D)));
    } else{ // some roots are imaginary
        S = std::cbrt(R+std::sqrt(D));
        T = std::cbrt(R-std::sqrt(D));
    }
    std::complex<double> x1 = S + T - b/(3.0*a);
    std::complex<double> x2 = -(S+T)/2.0 - b/(3.0*a) + im*std::sqrt(3.0)/2.0*(S-T);
    std::complex<double> x3 = -(S+T)/2.0 - b/(3.0*a) - im*std::sqrt(3.0)/2.0*(S-T);

    // std::cout << "check x1: " << a*x1*x1*x1 + b*x1*x1 + c*x1 + d << std::endl;
    // std::cout << "check x2: " << a*x2*x2*x2 + b*x2*x2 + c*x2 + d << std::endl;
    // std::cout << "check x3: " << a*x3*x3*x3 + b*x3*x3 + c*x3 + d << std::endl;

    return cubic_roots{ x1, x2, x3 };
}

double Integrals::falsePosMethod(std::function<double(double)> funct, double a, double b, const double tol) const noexcept(false){
    if (funct(a)*funct(b)>0.0){
        throw std::range_error("The x-range chosen to sample to find the root is not correct. Modify it such that the function changes sign.");
    }
    double c=10.0*tol; // to start the loop
    unsigned int iter=1;
    while (std::abs(funct(c))>tol && iter<=MAX_ITER_ROOT){
        c=(a-b)*funct(a)/(funct(b)-funct(a))+a;
        // std::cout << "c: " << c << " and funct(c): " << funct(c) << std::endl;
        if (std::abs(funct(c))<=tol){
            break;
        }
        else if (std::abs(funct(c))>tol){
            if (funct(a)*funct(c)<0.0){
                b=c;
            }
            else if (funct(c)*funct(b)<0.0){
                a=c;
            }
            else{
                throw std::runtime_error("Function might not be continuous. Bad behaviour.");
            }
        }
        iter++;
    }

    if (iter>MAX_ITER_ROOT){
        throw std::runtime_error("Exceeded the amount of loops allowed for root finding. Current value is: "+std::to_string(MAX_ITER_ROOT));
    }
    return c;
}