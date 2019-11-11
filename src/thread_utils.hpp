#include<mutex>
#include<thread>
#include "green_utils.hpp"

#define NUM_THREADS 4


static std::mutex mutx;
extern arma::Mat< std::complex<double> > matGamma; // Matrices used in case parallel.
extern arma::Mat< std::complex<double> > matWeigths;
extern arma::Mat< std::complex<double> > matTotSus;

namespace ThreadFunctor{

    class ThreadWrapper{
        public:
            //ThreadWrapper(Hubbard::FunctorBuildGk& Gk, Hubbard::K_1D& q, arma::cx_dmat::iterator matPtr, arma::cx_dmat::iterator matWPtr);
            ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_1D& q,double ndo_converged);
            ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_2D& q,double ndo_converged);
            ~ThreadWrapper()=default;
            void operator()(int ktilde, int kbar, double b); // 1D
            void operator()(int kbarx_m_tildex, int kbary_m_tildey); // 2D
            std::complex<double> gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar);
            std::complex<double> gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar);
            std::vector< std::complex<double> > buildGK1D(std::complex<double> ik, double k);
            std::vector< std::complex<double> > buildGK2D(std::complex<double> ik, double kx, double ky);
            void join_all(std::vector<std::thread>& grp);
        private:
            double _ndo_converged;
            HF::FunctorBuildGk _Gk;
            HF::K_1D& _q;
            //arma::cx_dmat::iterator _ktb;
            //arma::cx_dmat::iterator _ktbW;
};

} /* end of namespace ThreadFunctor */