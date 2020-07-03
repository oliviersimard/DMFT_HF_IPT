#include "green_utils.hpp"

//unsigned int iter=0;

const arma::Mat< std::complex<double> > ZEROS_(2, 2, arma::fill::zeros);
arma::Mat< std::complex<double> > statMat(2,2);
std::vector<double> vecK;
std::vector< std::complex<double> > iwnArr_l;
std::vector< std::complex<double> > iqnArr_l;

double epsilonk(double kx) noexcept{
    return -2.0*std::cos(kx);
}

double epsilonk(double kx, double ky) noexcept{
    return -2.0*(std::cos(kx)+std::cos(ky));
}

double epsilonk2D(double kx, double ky){
    return -2.0*(std::cos(kx)+std::cos(ky));
}

double epsilonk(double kx, double ky, double kz) noexcept{
    return -2.0*(std::cos(kx)+std::cos(ky)+std::cos(kz));
}

Data& Data::operator=(const Data& obj){
    if (&obj == this) return *this;
    else{
        std::cerr << "assignment operator used.\n"; // Unbuffered.
    }
    return *this;
}

// Static variables
unsigned int Data::objectCount=0;
unsigned int Data::N_k=0, Data::N_tau=0;
double Data::dtau=0.0, Data::U=0.0, Data::beta=0.0, Data::hyb_c=0.0;

// Static variables
arma::Cube<double> GreenStuff::dblVec_pos(2,2,1+1,arma::fill::zeros), GreenStuff::dblVec_neg(2,2,1+1,arma::fill::zeros);
arma::Cube< std::complex<double> > GreenStuff::cplxVec(2,2,1,arma::fill::zeros);

GreenStuff::GreenStuff(const unsigned int N_tau, const unsigned int N_k, const double beta, const double U, const double hyb_c, std::vector< std::complex<double> > iwnArr_,
                                                arma::Cube<double>& matsubara_t_pos, arma::Cube<double>& matsubara_t_neg, arma::Cube< std::complex<double> >& matsubara_w) noexcept(false) : Data(N_tau,N_k,beta,U,hyb_c), 
                                                matsubara_t_pos(matsubara_t_pos), matsubara_t_neg(matsubara_t_neg), matsubara_w(matsubara_w), iwnArr(iwnArr_){   
    #ifndef PARALLEL // Don't need this check when working in parallel env.
    if (!(matsubara_t_pos.size()==2*2*(2*N_tau+1)) || !(matsubara_t_neg.size()==2*2*(2*N_tau+1))){
        throw std::length_error("Vector matsubara_t has the wrong size!");
    }
    else if (!(matsubara_w.size()==2*2*2*N_tau)){
        throw std::length_error("Vector matsubara_w has the wrong size!");
    }
    #endif
}

arma::Cube< std::complex<double> > GreenStuff::green_inf() const{
    arma::Cube< std::complex<double> > hyb_init(2,2,iwnArr.size(),arma::fill::zeros);
    unsigned int i=0;
    for (auto it=iwnArr.begin(); it!=iwnArr.end(); it++){
        hyb_init.slice(i)(0,0) = 1.0/( *it + mu0 - hyb_c/(*it) );
        i++;
    }
    return hyb_init;
}

std::ostream& operator<<(std::ostream& os, const HF::FunctorBuildGk& obj){
    return os << "The content is: \n" << "U: " << obj._u << "\n" << "mu: " << obj._mu << "\n" <<
    "beta: " << obj._beta << "\n" << "n_do: " << obj._ndo << "\n" <<
    "size Matsubara arr: " << obj._size << "\n" << "gridK: " << obj._Nk << std::endl;
}

namespace HF{

    FunctorBuildGk::FunctorBuildGk(double mu,double beta,double u,double ndo,std::vector<double>& kArr_l,unsigned int Nit,unsigned int Nk,std::vector< std::complex<double> >& Gup_k) : 
                                _mu(mu), _u(u), _ndo(ndo), _beta(beta), _Nit(Nit), _Nk(Nk), _kArr_l(kArr_l){
        this->_Gup_k = &Gup_k.front(); 
        this->_size = Gup_k.size();
        for (int j=0; j<_size; j++){
            this->_precomp_wn.push_back(w(j,0.0));
            this->_precomp_qn.push_back(q(j));
        }
    }
    FunctorBuildGk& FunctorBuildGk::operator=(const FunctorBuildGk& obj){
        if (this != &obj){
            this->_beta=obj._beta;
            this->_u=obj._u;
            this->_size=obj._size;
            this->_Nk=obj._Nk;
            this->_Nit=obj._Nit;
            this->_ndo=obj._ndo;
            this->_mu=obj._mu;
            this->_Gup_k=obj._Gup_k;
            this->_kArr_l=obj._kArr_l;
            this->_precomp_qn=obj._precomp_qn;
            this->_precomp_wn=obj._precomp_wn;
        }
        return *this;
    }
    #if DIM == 1
    arma::Mat< std::complex<double> > FunctorBuildGk::buildGkAA_1D(int j, double kx) const{
        statMat(0,0) = 1.0/( w(j,_mu) - _u*_ndo - epsilonk(kx)*epsilonk(kx)/( w(j,_mu) - _u*(1.0-_ndo) ) ); // G^{AA}_{up}
        statMat(1,1) = 1.0/( w(j,_mu) - _u*(1.0-_ndo) - epsilonk(kx)*epsilonk(kx)/( w(j,_mu) - _u*(_ndo) ) ); // G^{AA}_{down}
        statMat(0,1) = 0.0; statMat(1,0) = 0.0;
        return statMat;
    }

    arma::Mat< std::complex<double> > FunctorBuildGk::buildGkAA_1D_w(std::complex<double> w, double kx) const{
        statMat(0,0) = 1.0/( w + _mu - epsilonk(kx) - _u*_ndo );
        statMat(1,1) = 1.0/( w + _mu - epsilonk(kx) - _u*(1.0-_ndo) );
        statMat(0,1) = 0.0; statMat(1,0) = 0.0;
        return statMat;
    }
    #elif DIM == 2
    arma::Mat< std::complex<double> > FunctorBuildGk::buildGkAA_2D(int j, double kx, double ky) const{
        statMat(0,0) = 1.0/( w(j,_mu) - _u*_ndo - epsilonk(kx,ky)*epsilonk(kx,ky)/( w(j,_mu) - _u*(1.0-_ndo) ) ); // G^{AA}_{up} or G^{BB}_{down}
        statMat(1,1) = 1.0/( w(j,_mu) - _u*(1.0-_ndo) - epsilonk(kx,ky)*epsilonk(kx,ky)/( w(j,_mu) - _u*(_ndo) ) ); // G^{AA}_{down} or G^{BB}_{up}
        statMat(0,1) = 0.0; statMat(1,0) = 0.0;
        return statMat;
    }

    arma::Mat< std::complex<double> > FunctorBuildGk::buildGkAA_2D_w(std::complex<double> w, double kx, double ky) const{
        statMat(0,0) = 1.0/( w + _mu - _u*_ndo - epsilonk(kx,ky) ); // G^{AA}_{up} or G^{BB}_{down}
        statMat(1,1) = 1.0/( w + _mu - _u*(1.0-_ndo) - epsilonk(kx,ky) ); // G^{AA}_{down} or G^{BB}_{up}
        statMat(0,1) = 0.0; statMat(1,0) = 0.0;
        return statMat;
    }
    #endif
    arma::Mat< std::complex<double> >& FunctorBuildGk::swap(arma::Mat< std::complex<double> >& M) const{
        arma::Mat< std::complex<double> > tmp_mat = ZEROS_;
        tmp_mat(0,0) = M(1,1);
        tmp_mat(1,1) = M(0,0);
        M = tmp_mat;
        return M;
    }
    #if DIM == 1
    void FunctorBuildGk::update_ndo_1D(){
        
        for (size_t i=0; i<_Nit; i++) {
            double ndo_av=0.0;
            for (size_t kkx=0; kkx<=_Nk; kkx++) {
                // calculate Gup_k in Matsubara space (AFM)
                for (size_t jj=0; jj<_size; jj++){
                    *(_Gup_k+jj) = buildGkAA_1D(jj,_kArr_l[kkx])(0,0);
                    //*(_Gup_k+jj) = buildGkBB_1D(jj,_mu,_beta,_u,_ndo,_kArr[kkx])(0,0); // Modified here to BB
                }

                // calculate ndo_k
                double ndo_k=0.0;
                for (size_t jj=0; jj<_size; jj++)
                    ndo_k += (2./_beta)*( *(_Gup_k+jj)-1./w(jj,0.0) ).real();
                ndo_k -= 0.5;
                ndo_k *= (-1);
            
                if ((kkx==0) || (kkx==_Nk)){
                    ndo_av += 0.5*ndo_k;
                }
                else{
                    ndo_av += ndo_k;
                }
            }
            ndo_av /= (_Nk);
            _ndo = ndo_av;
        }
    }
    #elif DIM == 2
    void FunctorBuildGk::update_ndo_2D(){
        
        for (size_t i=0; i<_Nit; i++) {
            double ndo_av=0.0;
            for (size_t kkx=0; kkx<=_Nk; kkx++) {
                for (size_t kky=0; kky<=_Nk; kky++){
                    // calculate Gup_k in Matsubara space (AFM)
                    for (size_t jj=0; jj<_size; jj++)
                        *(_Gup_k+jj) = buildGkAA_2D(jj,_kArr_l[kkx],_kArr_l[kky])(0,0);
                        // calculate ndo_k
                    double ndo_k=0.0;
                    for (size_t jj=0; jj<_size; jj++)
                        ndo_k += (2./_beta)*( *(_Gup_k+jj)-1./w(jj,0.0) ).real();
                    ndo_k -= 0.5;
                    ndo_k *= (-1);
            
                    if ( (kky==0) || (kky==_Nk) || (kkx==0) || (kkx==_Nk) ){
                        if ( ((kkx==0) || (kkx==_Nk)) && ((kky==0) || (kky==_Nk)) ){
                            ndo_av += 0.25*ndo_k;
                        }
                        ndo_av += 0.5*ndo_k;
                    }
                    else
                        ndo_av += ndo_k;
            
                }
            }
            ndo_av /= (_Nk*_Nk);
            _ndo = ndo_av;
        }
    }
    #endif

    /***************   K's   ******************/

    K_1D K_1D::operator+(const K_1D& rhs) const{
        K_1D obj(_qx,_iwn);
        obj._qx = this->_qx + rhs._qx;
        obj._iwn = this->_iwn + rhs._iwn;

        return obj;
    }

    K_1D K_1D::operator-(const K_1D& rhs) const{
        K_1D obj(_qx,_iwn);
        obj._qx = this->_qx - rhs._qx;
        obj._iwn = this->_iwn - rhs._iwn;

        return obj;
    }

    K_2D K_2D::operator+(const K_2D& rhs) const{
        K_2D obj(_qx,_qy,_iwn);
        obj._qx = this->_qx + rhs._qx;
        obj._qy = this->_qy + rhs._qy;
        obj._iwn = this->_iwn + rhs._iwn;

        return obj;
    }

    K_2D K_2D::operator-(const K_2D& rhs) const{
        K_2D obj(_qx,_qy,_iwn);
        obj._qx = this->_qx - rhs._qx;
        obj._qy = this->_qy - rhs._qy;
        obj._iwn = this->_iwn - rhs._iwn;

        return obj;
    }

} /* end of namespace HF */