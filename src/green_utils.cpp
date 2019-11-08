#include "green_utils.hpp"

unsigned int iter=0;
std::vector<double> vecK;
std::vector< std::complex<double> > iwnArr_l;

inline Data::Data(const unsigned int N_tau_, const unsigned int N_k_, const double beta_, const double U_, const double hyb_c){
    if (objectCount<1){ // Maybe try to implement it as a singleton.
        this->N_k=N_k_;
        this->hyb_c=hyb_c;
        this->N_tau=N_tau_;
        this->beta=beta_;
        this->U=U_;
        this->dtau=beta/N_tau;
        this->mu=U/2.0; // starting at half-filling
    }
    objectCount++;    
}

inline Data::Data(const Data& obj){}

Data& Data::operator=(const Data& obj){
    if (&obj == this) return *this;
    else{
        std::cerr << "assignment operator used.\n";
    }
    return *this;
}

// Static variables
unsigned int Data::objectCount=0;
unsigned int Data::N_k=0, Data::N_tau=0;
double Data::mu=0.0, Data::dtau=0.0, Data::U=0.0, Data::beta=0.0, Data::hyb_c=0.0, Data::mu0=0.0;

// Static variables
arma::Cube<double> GreenStuff::dblVec_pos(2,2,1+1,arma::fill::zeros), GreenStuff::dblVec_neg(2,2,1+1,arma::fill::zeros);
arma::Cube< std::complex<double> > GreenStuff::cplxVec(2,2,1,arma::fill::zeros);

inline GreenStuff::GreenStuff() : Data(), matsubara_t_pos(dblVec_pos), matsubara_t_neg(dblVec_neg), matsubara_w(cplxVec){
    std::cerr << "Constructor GreenStuff()\n";
}

GreenStuff::GreenStuff(const unsigned int N_tau, const unsigned int N_k, const double beta, const double U, const double hyb_c, std::vector< std::complex<double> > iwnArr_,
                                                arma::Cube<double>& matsubara_t_pos, arma::Cube<double>& matsubara_t_neg, arma::Cube< std::complex<double> >& matsubara_w) noexcept(false) : Data(N_tau,N_k,beta,U,hyb_c), 
                                                matsubara_t_pos(matsubara_t_pos), matsubara_t_neg(matsubara_t_neg), matsubara_w(matsubara_w), iwnArr(iwnArr_){   
    if (!(matsubara_t_pos.size()==2*2*(2*N_tau+1)) || !(matsubara_t_neg.size()==2*2*(2*N_tau+1))){
        throw std::length_error("Vector matsubara_t has the wrong size!");
    }
    else if (!(matsubara_w.size()==2*2*2*N_tau)){
        throw std::length_error("Vector matsubara_w has the wrong size!");
    }
}

// inline GreenStuff::GreenStuff(const GreenStuff& obj) : Data(obj), matsubara_t_pos(obj.matsubara_t_pos), matsubara_t_neg(obj.matsubara_t_neg),
//                                 matsubara_w(obj.matsubara_w), iwnArr(obj.iwnArr){
//     std::cerr << "Copy Constructor GreenStuff" << std::endl;
// }

// GreenStuff& GreenStuff::operator=(const GreenStuff& obj){ // Should have pointers instead of member references.
//     if (&obj == this){
//         std::cerr << "Call to identical copy assignment function.\n";
//     }
//     else{
//         (*this).Data::operator=(obj);
//         /* Sets the matsubara vectors to obj */
//         std::copy(&obj.iwnArr[0],&obj.iwnArr[0]+obj.iwnArr.size(),&this->iwnArr[0]);
//         //std::copy(&obj.matsubara_t[0],&obj.matsubara_t[0]+obj.matsubara_t.size(),&this->matsubara_t[0]);
//         //std::copy(&obj.matsubara_w[0],&obj.matsubara_w[0]+obj.matsubara_w.size(),&this->matsubara_w[0]);
//         std::cerr << "Call to copy assignment function.\n";
//     }
//     return *this;
// }

arma::Cube< std::complex<double> > GreenStuff::green_inf() const{
    arma::Cube< std::complex<double> > hyb_init(2,2,iwnArr.size(),arma::fill::zeros);
    unsigned int i=0;
    for (auto it=iwnArr.begin(); it!=iwnArr.end(); it++){
        hyb_init.slice(i)(0,0) = 1.0/( *it + mu0 - hyb_c/(*it) );
        i++;
    }
    return hyb_init;
}