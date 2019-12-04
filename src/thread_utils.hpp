#ifndef Thread_Utils_H_
#define Thread_Utils_H_

//#define PARALLEL

#include<mpi.h>
#include "susceptibilities.hpp"

#define SEND_DATA_TAG 2000
#define SEND_DATA_TAG_GAMMA 2001
#define SEND_DATA_TAG_WEIGHTS 2002
#define SEND_DATA_TAG_TOT_SUS 2003
#define RETURN_DATA_TAG 3000
#define RETURN_DATA_TAG_GAMMA 3001
#define RETURN_DATA_TAG_WEIGHTS 3002
#define RETURN_DATA_TAG_TOT_SUS 3003
#define RETURN_DATA_TAG_MID_LEV 3004
#define RETURN_DATA_TAG_CORR 3005

extern arma::Mat< std::complex<double> > matGamma; // Matrices used in case parallel.
extern arma::Mat< std::complex<double> > matWeigths;
extern arma::Mat< std::complex<double> > matTotSus;
extern arma::Mat< std::complex<double> > matCorr;
extern arma::Mat< std::complex<double> > matMidLev;
extern int root_process;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecGammaSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecWeightsSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecTotSusSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecCorrSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecMidLevSlaves;


template<typename T> 
inline void calculateSusceptibilitiesParallel(IPT2::SplineInline< std::complex<double> >,std::string,std::string,bool,bool,ThreadFunctor::solver_prototype);
void get_vector_mpi(size_t totSize,bool is_jj,bool is_full,ThreadFunctor::solver_prototype sp,std::vector<mpistruct_t>* vec_root_process);
void fetch_data_from_slaves(int an_id,MPI_Status& status,bool is_full,int ierr,size_t num_elements_per_proc,size_t sizeOfTuple);

namespace ThreadFunctor{
    enum solver_prototype : short { HF_prot, IPT2_prot };
    struct mpistruct{
        size_t _lkt, _lkb;
        bool _is_jj, _is_full;
        solver_prototype _sp;
    };
    class ThreadWrapper{ 
        public:
            //ThreadWrapper(Hubbard::FunctorBuildGk& Gk, Hubbard::K_1D& q, arma::cx_dmat::iterator matPtr, arma::cx_dmat::iterator matWPtr);
            explicit ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_1D q,double ndo_converged);
            explicit ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_2D qq,double ndo_converged);
            explicit ThreadWrapper(HF::K_1D q,IPT2::SplineInline< std::complex<double> > splInline);
            explicit ThreadWrapper(HF::K_2D qq,IPT2::SplineInline< std::complex<double> > splInline);
            ThreadWrapper()=default;
            void operator()(size_t ktilde, size_t kbar, bool is_jj, bool is_full, solver_prototype sp) const; // 1D IPT/HF
            void operator()(solver_prototype sp, size_t kbarx_m_tildex, size_t kbary_m_tildey, bool is_jj, bool is_full) const; // 2D IPT/HF
            std::tuple< std::complex<double>,std::complex<double> > gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const;
            std::tuple< std::complex<double>,std::complex<double> > gamma_oneD_spsp_IPT(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const;
            std::tuple< std::complex<double>,std::complex<double> > gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::tuple< std::complex<double>,std::complex<double> > gamma_twoD_spsp_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::vector< std::complex<double> > buildGK1D(std::complex<double> ik, double k) const;
            std::vector< std::complex<double> > buildGK1D_IPT(std::complex<double> ik, double k) const;
            std::vector< std::complex<double> > buildGK2D(std::complex<double> ik, double kx, double ky) const;
            std::vector< std::complex<double> > buildGK2D_IPT(std::complex<double> ik, double kx, double ky) const;
            std::complex<double> gamma_oneD_spsp_full_lower(double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const;
            std::complex<double> gamma_oneD_spsp_full_lower_IPT(double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const;
            std::complex<double> gamma_twoD_spsp_full_lower(double kpx,double kpy,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> iknp,std::complex<double> wbar) const;
            std::complex<double> gamma_twoD_spsp_full_lower_IPT(double kpx,double kpy,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> iknp,std::complex<double> wbar) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_oneD_spsp_full_middle_plotting(double ktilde,double kbar,std::complex<double> wbar,std::complex<double> wtilde) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_oneD_spsp_full_middle_plotting_IPT(double ktilde,double kbar,std::complex<double> wbar,std::complex<double> wtilde) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_twoD_spsp_full_middle_plotting(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_twoD_spsp_full_middle_plotting_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde) const;
            std::complex<double> getWeightsHF(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::complex<double> getWeightsIPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
        private:
            void save_data_to_local_extern_matrix_instancesIPT(std::complex<double> kt_kb,std::complex<double> weights,std::complex<double> mid_lev,std::complex<double> corr,std::complex<double> tot_sus,
                        size_t k1,size_t k2,bool is_jj,bool is_full,int world_rank) const;
            void save_data_to_local_extern_matrix_instances(std::complex<double> kt_kb,std::complex<double> weights,std::complex<double> mid_lev,std::complex<double> corr,std::complex<double> tot_sus,
                        size_t k1,size_t k2,bool is_jj,bool is_full,int world_rank) const;
            double _ndo_converged {0.0};
            HF::FunctorBuildGk _Gk={};
            HF::K_1D _q={};
            HF::K_2D _qq={};
            IPT2::SplineInline< std::complex<double> > _splInline={};
    };

    inline ThreadWrapper::ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_1D q,double ndo_converged) : _q(q){
        this->_ndo_converged=ndo_converged;
        this->_Gk=Gk;
    }
    inline ThreadWrapper::ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_2D q,double ndo_converged) : _q(q){
        this->_ndo_converged=ndo_converged;
        this->_Gk=Gk;
    }
    inline ThreadWrapper::ThreadWrapper(HF::K_1D q,IPT2::SplineInline< std::complex<double> > splInline){
        this->_q=q;
        this->_splInline=splInline;
    }
    inline ThreadWrapper::ThreadWrapper(HF::K_2D qq,IPT2::SplineInline< std::complex<double> > splInline){
        this->_qq=qq;
        this->_splInline=splInline;
    }
    #if DIM == 1
    // inline std::vector< std::complex<double> > ThreadWrapper::buildGK1D(std::complex<double> ik, double k) const{
    //     std::vector< std::complex<double> > GK = { 1.0/( ik + _Gk._mu - epsilonk(k) - _Gk._u*_ndo_converged ), 1.0/( ik + _Gk._mu - epsilonk(k) - _Gk._u*(1.0-_ndo_converged) ) }; // UP, DOWN
    //     return GK;
    // }

    inline std::vector< std::complex<double> > ThreadWrapper::buildGK1D_IPT(std::complex<double> ik, double k) const{
        std::vector< std::complex<double> > GK = { 1.0/( ik + GreenStuff::mu - epsilonk(k) - _splInline.calculateSpline(ik.imag()) ), 1.0/( ik + GreenStuff::mu - epsilonk(k) - _splInline.calculateSpline(ik.imag()) ) }; // UP, DOWN
        return GK;
    }
    #elif DIM == 2
    inline std::vector< std::complex<double> > ThreadWrapper::buildGK2D(std::complex<double> ik, double kx, double ky) const{
        std::vector< std::complex<double> > GK = { 1.0/( ik + _Gk._mu - epsilonk(kx,ky) - _Gk._u*_ndo_converged ), 1.0/( ik + _Gk._mu - epsilonk(kx,ky) - _Gk._u*(1.0-_ndo_converged) ) }; // UP, DOWN
        return GK;
    }

    inline std::vector< std::complex<double> > ThreadWrapper::buildGK2D_IPT(std::complex<double> ik, double kx, double ky) const{
        std::vector< std::complex<double> > GK = { 1.0/( ik + GreenStuff::mu - epsilonk(kx,ky) - _splInline.calculateSpline( ik.imag() ) ), 1.0/( ik + GreenStuff::mu - epsilonk(kx,ky) - _splInline.calculateSpline( ik.imag() ) ) }; // UP, DOWN
        return GK;
    }
    #endif

} /* end of namespace ThreadFunctor */

template<>
inline void calculateSusceptibilitiesParallel<HF::FunctorBuildGk>(HF::FunctorBuildGk Gk,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,double ndo_converged,ThreadFunctor::solver_prototype sp){
    MPI_Status status;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus, outputChispspBubble, outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspTotSus(pathToDir+customDirName+"/susceptibilities/ChispspTotSus_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspBubble(pathToDir+customDirName+"/susceptibilities/ChispspBubble_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspBubbleCorr;
    if (is_full)
        strOutputChispspBubbleCorr = pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat";
    const size_t totSize=Gk._kArr_l.size()*Gk._kArr_l.size(); // Nk+1 * Nk+1
    int world_rank, world_size, start_arr, end_arr, num_elems_to_send, ierr, num_elems_to_receive;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    const size_t num_elements_per_proc = totSize/world_size;
    std::vector<mpistruct_t>* vec_root_process = new std::vector<mpistruct_t>(totSize);
    std::vector<mpistruct_t>* vec_slave_processes = new std::vector<mpistruct_t>(num_elements_per_proc);
    const size_t sizeOfTuple = sizeof(std::tuple< size_t,size_t,std::complex<double> >);
    #if DIM == 1
    HF::K_1D q(0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(Gk,q,ndo_converged);
    #elif DIM == 2
    HF::K_2D qq(0.0,0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(Gk,qq,ndo_converged);
    #endif
    if (world_rank==root_process){
        // First initialize the data array to be distributed across all the processes called in.
        get_vector_mpi(totSize,is_jj,is_full,sp,vec_root_process);
        /* distribute a portion of the bector to each child process */
        for(int an_id = 1; an_id < world_size; an_id++) {
            start_arr = an_id*num_elements_per_proc + 1;
            end_arr = (an_id + 1)*num_elements_per_proc;
            if((totSize - end_arr) < num_elements_per_proc) // Taking care of the remaining data.
               end_arr = totSize - 1;
            num_elems_to_send = end_arr - start_arr + 1;
            ierr = MPI_Send( &num_elems_to_send, 1 , MPI_INT, an_id, SEND_DATA_TAG, MPI_COMM_WORLD);
            ierr = MPI_Send( (void*)(vec_root_process->data()+start_arr), sizeof(mpistruct_t)*num_elems_to_send, MPI_BYTE,
                  an_id, SEND_DATA_TAG, MPI_COMM_WORLD);
        }
        /* Calculate the susceptilities for the elements assigned to the root process, that is the beginning of the vector. */
        mpistruct_t tmpObj;
        for (int i=0; i<=num_elements_per_proc; i++){
            tmpObj=vec_root_process->at(i);
            #if DIM == 1
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,tmpObj._sp); // Performing the calculations here...
            #elif DIM == 2
            threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full); // Performing the calculations here...
            #endif
            printf("(%li,%li) calculated by root process\n", tmpObj._lkt, tmpObj._lkb);
        }
        MPI_Barrier(MPI_COMM_WORLD); // Wait for the other processes to finish before moving on.
        /* Gather the results from the child processes into the externally linked Math rices meant for this purpose. */
        for(int an_id = 1; an_id < world_size; an_id++) {
            fetch_data_from_slaves(an_id,status,is_full,ierr,num_elements_per_proc,sizeOfTuple);
        }
    } else{
        /* Slave processes receive their part of work from the root process. */
        ierr = MPI_Recv( &num_elems_to_receive, 1, MPI_INT, 
               root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
        ierr = MPI_Recv( (void*)(vec_slave_processes->data()), sizeof(mpistruct_t)*num_elems_to_receive, MPI_BYTE, 
               root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
        /* Calculate the sum of the portion of the array */
        mpistruct_t tmpObj;
        for(int i = 0; i < num_elems_to_receive; i++) {
            tmpObj = vec_slave_processes->at(i);
            #if DIM == 1
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,tmpObj._sp);
            #elif DIM == 2
            threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full);
            #endif
            printf("vec_slave_process el %d: %li, %li, %p\n", world_rank, tmpObj._lkt, tmpObj._lkb, (void*)vec_slave_processes);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        char chars_to_send[50];
        sprintf(chars_to_send,"vec_slave_process el %d completed", world_rank);
        /* Finally send integers to root process to notify the state of the calculations. */
        if (is_full)
            ierr = MPI_Send( (void*)(vecCorrSlaves->data()), sizeOfTuple*vecCorrSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_CORR, MPI_COMM_WORLD);
        ierr = MPI_Send( (void*)(vecMidLevSlaves->data()), sizeOfTuple*vecMidLevSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_MID_LEV, MPI_COMM_WORLD);
        ierr = MPI_Send( (void*)(vecGammaSlaves->data()), sizeOfTuple*vecGammaSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_GAMMA, MPI_COMM_WORLD);
        ierr = MPI_Send( (void*)(vecWeightsSlaves->data()), sizeOfTuple*vecWeightsSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_WEIGHTS, MPI_COMM_WORLD);
        ierr = MPI_Send( (void*)(vecTotSusSlaves->data()), sizeOfTuple*vecTotSusSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_TOT_SUS, MPI_COMM_WORLD);
        ierr = MPI_Send( chars_to_send, 50, MPI_CHAR, root_process, RETURN_DATA_TAG, MPI_COMM_WORLD);
        delete vecGammaSlaves; delete vecMidLevSlaves;
        delete vecWeightsSlaves; delete vecTotSusSlaves;
        if (is_full)
            delete vecCorrSlaves;
    }
    delete vec_root_process;
    delete vec_slave_processes;
    // To make sure not all the processes save at the same time in the same file.
    if (world_rank == root_process){
        outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
        outputChispspWeights.open(strOutputChispspWeights, std::ofstream::out | std::ofstream::app);
        outputChispspTotSus.open(strOutputChispspTotSus, std::ofstream::out | std::ofstream::app);
        outputChispspBubble.open(strOutputChispspBubble, std::ofstream::out | std::ofstream::app);
        if (is_full)
            outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::out | std::ofstream::app);
        for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
            for (size_t kbar=0; kbar<vecK.size(); kbar++){
                outputChispspGamma << matGamma(kbar,ktilde) << " ";
                outputChispspWeights << matWeigths(kbar,ktilde) << " ";
                outputChispspTotSus << matTotSus(kbar,ktilde) << " ";
                outputChispspBubble << matMidLev(kbar,ktilde) << " ";
                if (is_full)
                    outputChispspBubbleCorr << matCorr(kbar,ktilde) << " ";
            }
            outputChispspGamma << "\n";
            outputChispspWeights << "\n";
            outputChispspTotSus << "\n";
            outputChispspBubble << "\n";
            if (is_full)
                outputChispspBubbleCorr << "\n";
        }
        outputChispspGamma.close();
        outputChispspWeights.close();
        outputChispspTotSus.close();
        outputChispspBubble.close();
        if (is_full)
            outputChispspBubbleCorr.close();
    }
    ierr = MPI_Finalize();
}

template<>
inline void calculateSusceptibilitiesParallel<IPT2::DMFTproc>(IPT2::SplineInline< std::complex<double> > splInline,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,ThreadFunctor::solver_prototype sp){
    MPI_Status status;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus, outputChispspBubble, outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspTotSus(pathToDir+customDirName+"/susceptibilities/ChispspTotSus_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspBubble(pathToDir+customDirName+"/susceptibilities/ChispspBubble_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspBubbleCorr;
    if (is_full)
        strOutputChispspBubbleCorr = pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat";
    const size_t totSize=vecK.size()*vecK.size(); // Nk+1 * Nk+1
    int world_rank, world_size, start_arr, end_arr, num_elems_to_send, ierr, num_elems_to_receive;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    const size_t num_elements_per_proc = totSize/world_size; // Investigate this further, because there might prob with rounding.
    std::vector<mpistruct_t>* vec_root_process = new std::vector<mpistruct_t>(totSize);
    std::vector<mpistruct_t>* vec_slave_processes = new std::vector<mpistruct_t>(num_elements_per_proc);
    const size_t sizeOfTuple = sizeof(std::tuple< size_t,size_t,std::complex<double> >);
    #if DIM == 1
    HF::K_1D q(0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(q,splInline);
    #elif DIM == 2 
    HF::K_2D qq(0.0,0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(qq,splInline);
    #endif
    if (world_rank==root_process){
        // First initialize the data array to be distributed across all the processes called in.
        get_vector_mpi(totSize,is_jj,is_full,sp,vec_root_process);
        /* distribute a portion of the bector to each child process */
        for(int an_id = 1; an_id < world_size; an_id++) {
            start_arr = an_id*num_elements_per_proc + 1;
            end_arr = (an_id + 1)*num_elements_per_proc;
            if((totSize - end_arr) < num_elements_per_proc) // Taking care of the remaining data.
               end_arr = totSize - 1;
            num_elems_to_send = end_arr - start_arr + 1;
            ierr = MPI_Send( &num_elems_to_send, 1 , MPI_INT, an_id, SEND_DATA_TAG, MPI_COMM_WORLD);
            ierr = MPI_Send( (void*)(vec_root_process->data()+start_arr), sizeof(mpistruct_t)*num_elems_to_send, MPI_BYTE,
                  an_id, SEND_DATA_TAG, MPI_COMM_WORLD);
        }
        /* Calculate the susceptilities for the elements assigned to the root process, that is the beginning of the vector. */
        mpistruct_t tmpObj;
        for (int i=0; i<=num_elements_per_proc; i++){
            tmpObj=vec_root_process->at(i);
            #if DIM == 1
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,tmpObj._sp); // Performing the calculations here...
            #elif DIM == 2
            threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full);
            #endif
        }
        MPI_Barrier(MPI_COMM_WORLD); // Wait for the other processes to finish before moving on.
        printf("(%li,%li) calculated by root process\n", tmpObj._lkt, tmpObj._lkb);
        /* Gather the results from the child processes into the externally linked matrices meant for this purpose. */
        for(int an_id = 1; an_id < world_size; an_id++) {
            fetch_data_from_slaves(an_id,status,is_full,ierr,num_elements_per_proc,sizeOfTuple);
        }
    } else{
        /* Slave processes receive their part of work from the root process. */
        ierr = MPI_Recv( &num_elems_to_receive, 1, MPI_INT, 
               root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
        ierr = MPI_Recv( (void*)(vec_slave_processes->data()), sizeof(mpistruct_t)*num_elems_to_receive, MPI_BYTE, 
               root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
        /* Calculate the sum of my portion of the array */
        mpistruct_t tmpObj;
        for(int i = 0; i < num_elems_to_receive; i++) {
            tmpObj = vec_slave_processes->at(i);
            #if DIM == 1
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,tmpObj._sp);
            #elif DIM == 2
            threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full);
            #endif
            printf("vec_slave_process el %d: %li, %li, %p\n", world_rank, tmpObj._lkt, tmpObj._lkb, (void*)vec_slave_processes);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        char chars_to_send[50];
        sprintf(chars_to_send,"vec_slave_process el %d completed", world_rank);
        /* Finally send integers to root process to notify the state of the calculations. */
        ierr = MPI_Send( chars_to_send, 50, MPI_CHAR, root_process, RETURN_DATA_TAG, MPI_COMM_WORLD);
        if (is_full){
            ierr = MPI_Send( (void*)(vecCorrSlaves->data()),sizeOfTuple*vecCorrSlaves->size(),MPI_BYTE,root_process,RETURN_DATA_TAG_CORR,MPI_COMM_WORLD );
        }
        ierr = MPI_Send( (void*)(vecMidLevSlaves->data()),sizeOfTuple*vecMidLevSlaves->size(),MPI_BYTE,root_process,RETURN_DATA_TAG_MID_LEV,MPI_COMM_WORLD );
        ierr = MPI_Send( (void*)(vecGammaSlaves->data()),sizeOfTuple*vecGammaSlaves->size(),MPI_BYTE,root_process,RETURN_DATA_TAG_GAMMA,MPI_COMM_WORLD );
        ierr = MPI_Send( (void*)(vecWeightsSlaves->data()),sizeOfTuple*vecWeightsSlaves->size(),MPI_BYTE,root_process,RETURN_DATA_TAG_WEIGHTS,MPI_COMM_WORLD );
        ierr = MPI_Send( (void*)(vecTotSusSlaves->data()),sizeOfTuple*vecTotSusSlaves->size(),MPI_BYTE,root_process,RETURN_DATA_TAG_TOT_SUS,MPI_COMM_WORLD );
        delete vecGammaSlaves; delete vecWeightsSlaves;
        delete vecTotSusSlaves; delete vecMidLevSlaves;
        if (is_full)
            delete vecCorrSlaves;
    }
    delete vec_root_process;
    delete vec_slave_processes;
    // To make sure not all the processes save in the same file at the same time.
    if (world_rank == root_process){
        outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
        outputChispspWeights.open(strOutputChispspWeights, std::ofstream::out | std::ofstream::app);
        outputChispspTotSus.open(strOutputChispspTotSus, std::ofstream::out | std::ofstream::app);
        outputChispspBubble.open(strOutputChispspBubble, std::ofstream::out | std::ofstream::app);
        if (is_full)
            outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::out | std::ofstream::app);
        for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
            for (size_t kbar=0; kbar<vecK.size(); kbar++){
                outputChispspGamma << matGamma(kbar,ktilde) << " ";
                outputChispspWeights << matWeigths(kbar,ktilde) << " ";
                outputChispspTotSus << matTotSus(kbar,ktilde) << " ";
                outputChispspBubble << matMidLev(kbar,ktilde) << " ";
                if (is_full)
                    outputChispspBubbleCorr << matCorr(kbar,ktilde) << " ";
            }
            outputChispspGamma << "\n";
            outputChispspWeights << "\n";
            outputChispspTotSus << "\n";
            outputChispspBubble << "\n";
            if (is_full)
                outputChispspBubbleCorr << "\n";
        }
        outputChispspGamma.close();
        outputChispspWeights.close();
        outputChispspTotSus.close();
        outputChispspBubble.close();
        if (is_full)
            outputChispspBubbleCorr.close();
    }
    ierr = MPI_Finalize();
}

#endif /* end of Thread_Utils_H_ */