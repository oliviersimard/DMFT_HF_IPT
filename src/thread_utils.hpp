#ifndef Thread_Utils_H_
#define Thread_Utils_H_

#include<mutex>
#include<thread>
#include "green_utils.hpp"

#define NUM_THREADS 4


static std::mutex mutx;
extern arma::Mat< std::complex<double> > matGamma; // Matrices used in case parallel.
extern arma::Mat< std::complex<double> > matWeigths;
extern arma::Mat< std::complex<double> > matTotSus;
extern arma::Mat< std::complex<double> > matCorr;
extern arma::Mat< std::complex<double> > matMidLev;

template<typename T>
inline void calculateSusceptibilitiesParallel(T,const std::string,const std::string,const bool,const bool);

namespace ThreadFunctor{

    class ThreadWrapper{
        public:
            //ThreadWrapper(Hubbard::FunctorBuildGk& Gk, Hubbard::K_1D& q, arma::cx_dmat::iterator matPtr, arma::cx_dmat::iterator matWPtr);
            ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_1D q,double ndo_converged);
            ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_2D q,double ndo_converged);
            ~ThreadWrapper()=default;
            void operator()(size_t ktilde, size_t kbar, double b, bool is_jj); // 1D
            void operator()(size_t kbarx_m_tildex, size_t kbary_m_tildey); // 2D
            std::complex<double> gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar);
            std::complex<double> gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar);
            std::vector< std::complex<double> > buildGK1D(std::complex<double> ik, double k);
            std::vector< std::complex<double> > buildGK2D(std::complex<double> ik, double kx, double ky);
            void join_all(std::vector<std::thread>& grp);
        private:
            double _ndo_converged;
            HF::FunctorBuildGk _Gk;
            HF::K_1D _q;
    };

} /* end of namespace ThreadFunctor */

template<>
inline void calculateSusceptibilitiesParallel<HF::FunctorBuildGk>(HF::FunctorBuildGk Gk,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,double ndo_converged){
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus;// outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspTotSus(pathToDir+customDirName+"/susceptibilities/ChispspTotSus_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    //std::string strOutputChispspBubbleCorr(pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_IPT2_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    size_t it=0;
    HF::K_1D qq(0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    const size_t totSize=vecK.size()*vecK.size(); // Nk+1 * Nk+1
    ThreadFunctor::ThreadWrapper threadObj(Gk,qq,ndo_converged);
    size_t ltot,lkt,lkb;
    while (it<totSize){
        if (totSize % NUM_THREADS != 0){
            if ( (totSize-it)<NUM_THREADS ){
                size_t last_it=totSize-it;
                std::vector<std::thread> tt(last_it);
                for (size_t l=0; l<last_it; l++){
                    ltot=it+l; // Have to make sure spans over the whole array of k-space.
                    lkt = static_cast<size_t>(floor(ltot/vecK.size())); // Samples the rows
                    lkb = (ltot % vecK.size()); // Samples the columns
                    std::thread t(std::ref(threadObj),lkt,lkb,Gk._beta,is_jj);
                    tt[l]=std::move(t);
                    // tt[l]=thread(threadObj,lkt,lkb,beta);
                }
                threadObj.join_all(tt);
            }
            else{
                std::vector<std::thread> tt(NUM_THREADS);
                for (int l=0; l<NUM_THREADS; l++){
                    ltot=it+l; // Have to make sure spans over the whole array of k-space.
                    lkt = static_cast<size_t>(floor(ltot/vecK.size()));
                    lkb = (ltot % vecK.size());
                    std::cout << "lkt: " << lkt << " lkb: " << lkb << "\n";
                    std::thread t(std::ref(threadObj),lkt,lkb,Gk._beta,is_jj);
                    tt[l]=std::move(t);
                    //tt.push_back(static_cast<thread&&>(t));
                }
                threadObj.join_all(tt);
            }
        }
        else{
            std::vector<std::thread> tt(NUM_THREADS);
            for (size_t l=0; l<NUM_THREADS; l++){
                ltot=it+l; // Have to make sure spans over the whole array of k-space.
                lkt = static_cast<int>(floor(ltot/vecK.size()));
                lkb = (ltot % vecK.size());
                std::cout << "lkt: " << lkt << " lkb: " << lkb << "\n";
                std::thread t(std::ref(threadObj),lkt,lkb,Gk._beta,is_jj);
                tt[l]=std::move(t);
                // tt[l]=thread(threadObj,lkt,lkb,beta);
            }
            threadObj.join_all(tt);
        }
        it+=NUM_THREADS;
    }
    outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
    outputChispspWeights.open(strOutputChispspWeights, std::ofstream::out | std::ofstream::app);
    outputChispspTotSus.open(strOutputChispspTotSus, std::ofstream::out | std::ofstream::app);
    for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
        for (size_t kbar=0; kbar<vecK.size(); kbar++){
            outputChispspGamma << matGamma(kbar,ktilde) << " ";
            outputChispspWeights << matWeigths(kbar,ktilde) << " ";
            outputChispspTotSus << matTotSus(kbar,ktilde) << " ";
        }
        outputChispspGamma << "\n";
        outputChispspWeights << "\n";
        outputChispspTotSus << "\n";
    }
    outputChispspGamma.close();
    outputChispspWeights.close();
    outputChispspTotSus.close();
}



#endif /* end of Thread_Utils_H_ */