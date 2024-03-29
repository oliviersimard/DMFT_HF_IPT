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

std::complex<double> Integrals::I2D(std::function<std::complex<double>(double,double,std::complex<double>)> funct,double x0,double xf,double y0,double yf,std::complex<double> iwn,const double tol,unsigned int maxVal,bool is_converged) const{
    double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
    std::complex<double> result, prevResult(0.0), resultSumNyorNx, resultSumNxNy;
    unsigned int iter=1;
    while (!is_converged && iter<20){
        resultSumNyorNx=0.0, resultSumNxNy=0.0;
        dx=(xf-x0)/(double)maxVal;
        dy=(yf-y0)/(double)maxVal;
        prevResult=result;
        for (unsigned int i=1; i<maxVal; i++){
            resultSumNyorNx+=funct(x0+i*dx,y0,iwn)+funct(x0+i*dx,yf,iwn)+funct(x0,y0+i*dy,iwn)+funct(xf,y0+i*dy,iwn);
        }
        for (unsigned int i=1; i<maxVal; i++){
            for (unsigned int j=1; j<maxVal; j++){
                resultSumNxNy+=funct(x0+i*dx,y0+j*dy,iwn);
            }
        }
    
        result=0.25*dx*dy*(funct(x0,y0,iwn)+funct(xf,y0,iwn)+funct(x0,yf,iwn)+funct(xf,yf,iwn))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;

        is_converged = ( ( (std::abs(prevResult-result))>tol ) ) ? false : true;

        if (iter>MAX_ITER_INTEGRAL || is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
}

double Integrals::I2D(std::function<double(double,double)> funct,double x0,double xf,double y0,double yf,const double tol,unsigned int maxVal,bool is_converged) const{
    double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
    double resultSumNyorNx=0.0, resultSumNxNy=0.0, result, prevResult=0.0;
    unsigned int iter=1;
    while (!is_converged){
        resultSumNxNy=resultSumNyorNx=0.0;
        dx=(xf-x0)/(double)maxVal;
        dy=(yf-y0)/(double)maxVal;
        prevResult=result;
        for (unsigned int i=1; i<maxVal; i++){
            resultSumNyorNx+=funct(x0+i*dx,y0)+funct(x0+i*dx,yf)+funct(x0,y0+i*dy)+funct(xf,y0+i*dy);
        }
        for (unsigned int i=1; i<maxVal; i++){
            for (unsigned int j=1; j<maxVal; j++){
                resultSumNxNy+=funct(x0+i*dx,y0+j*dy);
            }
        }

        result=0.25*dx*dy*(funct(x0,y0)+funct(xf,y0)+funct(x0,yf)+funct(xf,yf))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;

        is_converged = (std::abs(prevResult-result)<tol) ? true : false;
        
        if (iter>MAX_ITER_INTEGRAL || is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
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

// double Integrals::I2DRec(std::function<double(double,double)> funct,double x0,double xf,double y0,double yf,unsigned int currLevel,double tol,unsigned int maxVal,bool is_converged,double prevResult) const{
//     // same spacing along both axes. No discrimination.
//     double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
//     double resultSumNyorNx=0.0, resultSumNxNy=0.0, result;

//     for (unsigned int i=1; i<maxVal; i++){
//         resultSumNyorNx+=funct(x0+i*dx,y0)+funct(x0+i*dx,yf)+funct(x0,y0+i*dy)+funct(xf,y0+i*dy);
//     }
//     for (unsigned int i=1; i<maxVal; i++){
//         for (unsigned int j=1; j<maxVal; j++){
//             resultSumNxNy+=funct(x0+i*dx,y0+j*dy);
//         }
//     }
//     std::cout << "xf: " << xf << "x0: " <<  x0 << "yf: " << yf  << "y0: " << y0 << std::endl;
//     result=0.25*dx*dy*(funct(x0,y0)+funct(xf,y0)+funct(x0,yf)+funct(xf,yf))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;

//     if ( (currLevel!=1) && ((prevResult-2.0*result)<=tol) ){ // because the segment is always subdivided by half.
//         is_converged=true;
//     }

//     if (_maxLevel>=currLevel && is_converged){
//         return result;
//     }
//     else if (_maxLevel<currLevel){
//         std::cout << "Integral has not converged!" << "\n";
//         std::cout << "Finished with value: " << result << std::endl;
//         return result;
//     }
//     else{
//         std::cout << currLevel << std::endl;
//         currLevel++;
//         return I2DRec(funct,x0,(xf-x0)/2.0,y0,yf,currLevel,tol,maxVal,is_converged,result)+I2DRec(funct,(xf-x0)/2.0,xf,y0,yf,currLevel,tol,maxVal,is_converged,result)+I2DRec(funct,x0,xf,y0,(yf-y0)/2.0,currLevel,tol,maxVal,is_converged,result)+I2DRec(funct,x0,xf,(yf-y0)/2.0,yf,currLevel,tol,maxVal,is_converged,result);
//     }
// }