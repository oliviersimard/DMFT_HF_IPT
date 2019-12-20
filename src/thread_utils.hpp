#ifndef Thread_Utils_H_
#define Thread_Utils_H_

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
extern std::complex<double>**** gamma_tensor;
extern std::complex<double>**** gamma_full_tensor; // This array construction is only useful when computing the full ladder diagrams.
extern int root_process;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecGammaSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecWeightsSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecTotSusSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecCorrSlaves;
extern std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecMidLevSlaves;

template<typename T> 
inline void calculateSusceptibilitiesParallel(IPT2::SplineInline< std::complex<double> >,std::string,std::string,bool,bool,ThreadFunctor::solver_prototype);

namespace ThreadFunctor{
    enum solver_prototype : short { HF_prot, IPT2_prot };
    struct mpistruct{
        size_t _lkt, _lkb;
        bool _is_jj, _is_full;
        solver_prototype _sp;
    };
    struct gamma_tensor_content{
        size_t _ktilde {0}, _wtilde {0}, _kbar {0}, _wbar {0};
        std::complex<double> _gamma={};
        gamma_tensor_content()=default;
        gamma_tensor_content(size_t ktilde,size_t wtilde,size_t kbar,size_t wbar,std::complex<double> gamma) : _ktilde(ktilde), 
            _wtilde(wtilde), _kbar(kbar), _wbar(wbar), _gamma(gamma) {}
    };
    void get_vector_mpi(size_t totSize,bool is_jj,bool is_full,ThreadFunctor::solver_prototype sp,std::vector<mpistruct_t>* vec_root_process);
    void fetch_data_from_slaves(int an_id,MPI_Status& status,bool is_full,int ierr,size_t num_elements_per_proc,size_t sizeOfTuple,size_t j);
    void send_messages_to_root_process(bool is_full, int ierr, size_t sizeOfTuple, char* chars_to_send, size_t j);
    void create_mpi_data_struct(MPI_Datatype& gamma_tensor_content_type);
    class ThreadWrapper{ 
        template<typename T> friend void ::calculateSusceptibilitiesParallel(T,std::string,std::string,bool,bool,double,ThreadFunctor::solver_prototype);
        template<typename T> friend void ::calculateSusceptibilitiesParallel(IPT2::SplineInline< std::complex<double> >,std::string,std::string,bool,bool,ThreadFunctor::solver_prototype);
        public:
            //ThreadWrapper(Hubbard::FunctorBuildGk& Gk, Hubbard::K_1D& q, arma::cx_dmat::iterator matPtr, arma::cx_dmat::iterator matWPtr);
            explicit ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_1D q,double ndo_converged);
            explicit ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_2D qq,double ndo_converged);
            explicit ThreadWrapper(HF::K_1D q,IPT2::SplineInline< std::complex<double> > splInline);
            explicit ThreadWrapper(HF::K_2D qq,IPT2::SplineInline< std::complex<double> > splInline);
            ThreadWrapper()=default;
            void operator()(size_t ktilde, size_t kbar, bool is_jj, bool is_full, size_t j, solver_prototype sp) const; // 1D IPT/HF
            void operator()(solver_prototype sp, size_t kbarx_m_tildex, size_t kbary_m_tildey, bool is_jj, bool is_full, size_t j) const; // 2D IPT/HF
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
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_oneD_spsp_full_middle_plotting(size_t ktilde,size_t kbar,size_t wbar,size_t wtilde,size_t jj) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_oneD_spsp_full_middle_plotting_IPT(size_t ktilde,size_t kbar,size_t wbar,size_t wtilde, size_t jj) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_twoD_spsp_full_middle_plotting(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde) const;
            std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_twoD_spsp_full_middle_plotting_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde) const;
            std::complex<double> getWeightsHF(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::complex<double> getWeightsIPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::complex<double> lindhard_function(bool is_jj, std::ofstream& ofS, const std::string& strOutput, int world_rank) const;
            std::complex<double> lindhard_functionIPT(bool is_jj, std::ofstream& ofS, const std::string& strOutput, int world_rank) const;
            void fetch_data_gamma_tensor_alltogether(size_t totSizeGammaTensor,int ierr,std::vector<int>* vec_counts, std::vector<int>* vec_disps, size_t sizeOfElMPI_Allgatherv,bool is_full);
        private:
            void save_data_to_local_extern_matrix_instancesIPT(std::complex<double> kt_kb,std::complex<double> weights,std::complex<double> mid_lev,std::complex<double> corr,std::complex<double> tot_sus,
                        size_t k1,size_t k2,bool is_jj,bool is_full,int world_rank,size_t j) const;
            void save_data_to_local_extern_matrix_instances(std::complex<double> kt_kb,std::complex<double> weights,std::complex<double> mid_lev,std::complex<double> corr,std::complex<double> tot_sus,
                        size_t k1,size_t k2,bool is_jj,bool is_full,int world_rank,size_t j) const;
            double _ndo_converged {0.0};
            HF::FunctorBuildGk _Gk={};
            HF::K_1D _q={};
            HF::K_2D _qq={};
            IPT2::SplineInline< std::complex<double> > _splInline={};
            static std::vector< gamma_tensor_content >* vecGammaTensorContent;
            static std::vector< gamma_tensor_content >* vecGammaFullTensorContent;
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
    inline std::vector< std::complex<double> > ThreadWrapper::buildGK1D_IPT(std::complex<double> ik, double k) const{
        std::vector< std::complex<double> > GK = { 1.0/( ik + GreenStuff::mu - epsilonk(k) - _splInline.calculateSpline(ik.imag()) ), 1.0/( ik + GreenStuff::mu - epsilonk(k) - _splInline.calculateSpline(ik.imag()) ) }; // UP, DOWN
        return GK;
    }
    #elif DIM == 2
    inline std::vector< std::complex<double> > ThreadWrapper::buildGK2D_IPT(std::complex<double> ik, double kx, double ky) const{
        std::vector< std::complex<double> > GK = { 1.0/( ik + GreenStuff::mu - epsilonk(kx,ky) - _splInline.calculateSpline( ik.imag() ) ), 1.0/( ik + GreenStuff::mu - epsilonk(kx,ky) - _splInline.calculateSpline( ik.imag() ) ) }; // UP, DOWN
        return GK;
    }
    #endif

} /* end of namespace ThreadFunctor */

template<>
inline void calculateSusceptibilitiesParallel<HF::FunctorBuildGk>(HF::FunctorBuildGk Gk,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,double ndo_converged,ThreadFunctor::solver_prototype sp){
    MPI_Status status;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus, outputChispspBubble, outputChispspBubbleCorr, outputChispsNonInteracting;
    std::string strOutputChispspWeights, strOutputChispspTotSus, strOutputChispspBubble, strOutputChispspGamma, strOutputChispspBubbleCorr, strOutputChispspNonInteracting;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    const size_t totSize=Gk._kArr_l.size()*Gk._kArr_l.size(); // Nk+1 * Nk+1
    const size_t totSizeGammaTensor=totSize*Gk._size*Gk._size;
    const size_t sizeOfElMPI_Allgatherv = Gk._size*Gk._size;
    int world_rank, world_size, start_arr, end_arr, num_elems_to_send, ierr, num_elems_to_receive;
    std::complex<double> sus_non_interacting;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    const size_t num_elements_per_proc = (world_size!=1) ? totSize/world_size : totSize-1; // Otherwise problem when calculating elements assigned to root process.
    std::vector<mpistruct_t>* vec_root_process = new std::vector<mpistruct_t>(totSize);
    std::vector<mpistruct_t>* vec_slave_processes = new std::vector<mpistruct_t>(num_elements_per_proc+1); // Root process has one more element...
    std::vector<int>* vec_counts = new std::vector<int>(world_size); // Hosts the number of values (in bytes) to be sent by MPI_Allgatherv.
    std::vector<int>* vec_disps = new std::vector<int>(world_size); // Displacements in recv buffer from which data is written.
    const size_t sizeOfTuple = sizeof(std::tuple< size_t,size_t,std::complex<double> >);
    #if DIM == 1
    HF::K_1D q1D;
    #elif DIM == 2
    HF::K_2D qq2D;
    #endif
    strOutputChispspNonInteracting = pathToDir+customDirName+"/susceptibilities/ChispspNonInt_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat";
    std::cout << "totSize: " << totSize << "\n";
    std::cout << "num_elements_per_proc: " << num_elements_per_proc << std::endl;
    for (size_t j=0; j<Gk._precomp_qn.size(); j++){ // Looping over the bosonic Matsubara frequencies...
        strOutputChispspWeights=pathToDir+customDirName+"/susceptibilities/ChispspWeights_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+"_iqn_"+std::to_string(Gk._precomp_qn[j].imag())+trailingStr+".dat";
        strOutputChispspTotSus=pathToDir+customDirName+"/susceptibilities/ChispspTotSus_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+"_iqn_"+std::to_string(Gk._precomp_qn[j].imag())+trailingStr+".dat";
        strOutputChispspBubble=pathToDir+customDirName+"/susceptibilities/ChispspBubble_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+"_iqn_"+std::to_string(Gk._precomp_qn[j].imag())+trailingStr+".dat";
        strOutputChispspGamma=pathToDir+customDirName+"/susceptibilities/ChispspGamma_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+"_iqn_"+std::to_string(Gk._precomp_qn[j].imag())+trailingStr+".dat";
        if (is_full)
            strOutputChispspBubbleCorr = pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+"_iqn_"+std::to_string(Gk._precomp_qn[j].imag())+trailingStr+".dat";
        #if DIM == 1
        q1D._iwn = Gk._precomp_qn[j]; // photon 4-vector
        ThreadFunctor::ThreadWrapper threadObj(Gk,q1D,ndo_converged);
        #elif DIM == 2
        qq2D._iwn = Gk._precomp_qn[j]; // photon 4-vector
        ThreadFunctor::ThreadWrapper threadObj(Gk,qq2D,ndo_converged);
        #endif
        if (world_rank == root_process)
            sus_non_interacting=threadObj.lindhard_function(is_jj,outputChispsNonInteracting,strOutputChispspNonInteracting,world_rank); // Non-interacting optical conductivity
        std::cout << "\n\n iqn: " << Gk._precomp_qn[j] << "\n\n";
        if (world_rank==root_process){
            // First initialize the data array to be distributed across all the processes called in.
            ThreadFunctor::get_vector_mpi(totSize,is_jj,is_full,sp,vec_root_process);
            /* MPI_Allgatherv */
            if (j==0){
                vec_counts->at(0)=(num_elements_per_proc+1)*sizeOfElMPI_Allgatherv; // MPI_Allgatherv
                vec_disps->at(0)=0;
            }
            /*  */
            /* distribute a portion of the bector to each child process */
            for(int an_id = 1; an_id < world_size; an_id++){
                start_arr = an_id*num_elements_per_proc + 1;
                end_arr = (an_id + 1)*num_elements_per_proc;
                if((totSize - end_arr) < num_elements_per_proc) // Taking care of the remaining data.
                    end_arr = totSize - 1;
                num_elems_to_send = end_arr - start_arr + 1;
                /* MPI_Allgatherv */
                if (j==0){ // Because the vectors vec_counts and vec_disps are deleted afterwards from heap...
                    vec_counts->at(an_id)=num_elems_to_send*sizeOfElMPI_Allgatherv;
                    vec_disps->at(an_id)=start_arr*sizeOfElMPI_Allgatherv;
                }
                /*  */
                ierr = MPI_Send( &num_elems_to_send, 1 , MPI_INT, an_id, SEND_DATA_TAG, MPI_COMM_WORLD );
                ierr = MPI_Send( (void*)(vec_root_process->data()+start_arr), sizeof(mpistruct_t)*num_elems_to_send, MPI_BYTE,
                    an_id, SEND_DATA_TAG, MPI_COMM_WORLD );
            }
            if (j==0){
                for (auto el : *vec_disps) 
                    std::cout << "vec_disps elements: " << el << std::endl;
            }
            /* Calculate the susceptilities for the elements assigned to the root process, that is the beginning of the vector. */
            mpistruct_t tmpObj;
            for (int i=0; i<=num_elements_per_proc; i++){ // Careful with <=
                tmpObj=vec_root_process->at(i);
                #if DIM == 1
                threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j,tmpObj._sp); // Performing the calculations here...
                #elif DIM == 2
                threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j); // Performing the calculations here...
                #endif
                printf("(%li,%li) calculated by root process\n", tmpObj._lkt, tmpObj._lkb);
            }
            // The root process doesn't append to slave buffers.
            assert(vecWeightsSlaves->size()==0 && vecGammaSlaves->size()==0 && vecMidLevSlaves->size()==0 && vecTotSusSlaves->size()==0 && vecCorrSlaves->size()==0);
            MPI_Barrier(MPI_COMM_WORLD); // Wait for the other processes to finish before moving on.
            /* Gather the results from the child processes into the externally linked Math rices meant for this purpose. */
            for(int an_id = 1; an_id < world_size; an_id++) {
                ThreadFunctor::fetch_data_from_slaves(an_id,status,is_full,ierr,num_elements_per_proc,sizeOfTuple,j);
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
                threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j,tmpObj._sp);
                #elif DIM == 2
                threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j);
                #endif
                printf("vec_slave_process el %d: %li, %li, %p\n", world_rank, tmpObj._lkt, tmpObj._lkb, (void*)vec_slave_processes);
            }
            // The containers for weights and TotSus must remain non-zero throughout the iqn loop (for both full and not full calculations.) 
            assert(vecWeightsSlaves->size()==vecTotSusSlaves->size());
            std::cout << "GETTING CLOSER" << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
            char chars_to_send[50];
            sprintf(chars_to_send,"vec_slave_process el %d completed", world_rank);
            /* Finally send integers to root process to notify the state of the calculations. */
            ThreadFunctor::send_messages_to_root_process(is_full,ierr,sizeOfTuple,chars_to_send,j);
        }
        if (j==0 && world_size>1){
            /* MPI_Allgatherv */
            MPI_Bcast( (void*)(vec_counts->data()), world_size, MPI_INT, root_process, MPI_COMM_WORLD );
            MPI_Bcast( (void*)(vec_disps->data()), world_size, MPI_INT, root_process, MPI_COMM_WORLD );
            /*  */
            /* Sending to all processes their respective content of gamma_tensor */
            threadObj.fetch_data_gamma_tensor_alltogether(totSizeGammaTensor,ierr,vec_counts,vec_disps,sizeOfElMPI_Allgatherv,is_full);
        }
        // To make sure not all the processes save at the same time in the same file.
        if (world_rank == root_process){
            outputChispspWeights.open(strOutputChispspWeights, std::ofstream::out | std::ofstream::app);
            outputChispspTotSus.open(strOutputChispspTotSus, std::ofstream::out | std::ofstream::app);
            if (is_full){
                if (j==0)
                    outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::out | std::ofstream::app);
                outputChispspBubble.open(strOutputChispspBubble, std::ofstream::out | std::ofstream::app);
                outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
            } else{
                if (j==0){
                    outputChispspBubble.open(strOutputChispspBubble, std::ofstream::out | std::ofstream::app);
                    outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
                }
            }
            for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
                for (size_t kbar=0; kbar<vecK.size(); kbar++){
                    outputChispspWeights << matWeigths(kbar,ktilde) << " "; 
                    outputChispspTotSus << matTotSus(kbar,ktilde) + sus_non_interacting  << " "; // Adding the non-interacting susceptibilities...
                    if (is_full){
                        if (j==0)
                            outputChispspBubbleCorr << matCorr(kbar,ktilde) << " ";
                        outputChispspGamma << matGamma(kbar,ktilde) << " ";
                        outputChispspBubble << matMidLev(kbar,ktilde) << " ";
                    } else{
                        if (j==0){
                            outputChispspBubble << matMidLev(kbar,ktilde) << " ";
                            outputChispspGamma << matGamma(kbar,ktilde) << " ";
                        }
                    }
                }
                outputChispspWeights << "\n";
                outputChispspTotSus << "\n";
                if (is_full){
                    if (j==0)
                        outputChispspBubbleCorr << "\n";
                    outputChispspGamma << "\n";
                    outputChispspBubble << "\n";
                } else{
                    if (j==0){
                        outputChispspGamma << "\n";
                        outputChispspBubble << "\n";
                    }
                }
            }
            outputChispspWeights.close();
            outputChispspTotSus.close();
            if (is_full){
                if (j==0)
                    outputChispspBubbleCorr.close();
                outputChispspGamma.close();
                outputChispspBubble.close();
            } else{
                if (j==0){
                    outputChispspGamma.close();
                    outputChispspBubble.close();
                }
            }
        }
    }// Cleaning process
    delete vecGammaSlaves; delete vecMidLevSlaves;
    delete vecWeightsSlaves; delete vecTotSusSlaves;
    if (is_full)
        delete vecCorrSlaves;
    delete vec_root_process;
    delete vec_slave_processes;
    ierr = MPI_Finalize();
}

template<>
inline void calculateSusceptibilitiesParallel<IPT2::DMFTproc>(IPT2::SplineInline< std::complex<double> > splInline,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,ThreadFunctor::solver_prototype sp){
    MPI_Status status;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus, outputChispspBubble, outputChispspBubbleCorr, outputChispspNonInteracting;
    std::string strOutputChispspGamma, strOutputChispspWeights, strOutputChispspTotSus, strOutputChispspBubble, strOutputChispspBubbleCorr, strOutputChispspNonInteracting;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    const size_t totSize=vecK.size()*vecK.size(); // Nk+1 * Nk+1
    const size_t sizeOfElMPI_Allgatherv=GreenStuff::N_tau*GreenStuff::N_tau;
    const size_t totSizeGammaTensor=totSize*sizeOfElMPI_Allgatherv;
    int world_rank, world_size, start_arr, end_arr, num_elems_to_send, ierr, num_elems_to_receive;
    std::complex<double> sus_non_interacting;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    const size_t num_elements_per_proc = (world_size!=1) ? totSize/world_size : totSize-1; // Investigate this further, because there might prob with rounding.
    std::vector<mpistruct_t>* vec_root_process = new std::vector<mpistruct_t>(totSize);
    std::vector<mpistruct_t>* vec_slave_processes = new std::vector<mpistruct_t>(num_elements_per_proc+1); // Root process has one more element...
    std::vector<int>* vec_counts = new std::vector<int>(world_size); // Hosts the number of values (in bytes) to be sent by MPI_Allgatherv.
    std::vector<int>* vec_disps = new std::vector<int>(world_size); // Displacements in recv buffer from which data is written.
    const size_t sizeOfTuple = sizeof(std::tuple< size_t,size_t,std::complex<double> >);
    #if DIM == 1
    HF::K_1D q1D;
    #elif DIM == 2
    HF::K_2D qq2D;
    #endif
    strOutputChispspNonInteracting = pathToDir+customDirName+"/susceptibilities/ChispspNonInt_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat";;
    for (size_t j=0; j<GreenStuff::N_tau; j++){
        strOutputChispspGamma=pathToDir+customDirName+"/susceptibilities/ChispspGamma_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+"_iqn_"+std::to_string(iqnArr_l[j].imag())+trailingStr+".dat";
        strOutputChispspWeights=pathToDir+customDirName+"/susceptibilities/ChispspWeights_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+"_iqn_"+std::to_string(iqnArr_l[j].imag())+trailingStr+".dat";
        strOutputChispspTotSus=pathToDir+customDirName+"/susceptibilities/ChispspTotSus_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+"_iqn_"+std::to_string(iqnArr_l[j].imag())+trailingStr+".dat";
        strOutputChispspBubble=pathToDir+customDirName+"/susceptibilities/ChispspBubble_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+"_iqn_"+std::to_string(iqnArr_l[j].imag())+trailingStr+".dat";
        if (is_full)
            strOutputChispspBubbleCorr = pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+"_iqn_"+std::to_string(iqnArr_l[j].imag())+trailingStr+".dat";
        #if DIM == 1
        q1D._iwn = iqnArr_l[j]; // photon 4-vector
        ThreadFunctor::ThreadWrapper threadObj(q1D,splInline);
        #elif DIM == 2 
        qq2D._iwn = iqnArr_l[j]; // photon 4-vector
        ThreadFunctor::ThreadWrapper threadObj(qq2D,splInline);
        #endif
        if (world_rank == root_process)
            sus_non_interacting=threadObj.lindhard_functionIPT(is_jj,outputChispspNonInteracting,strOutputChispspNonInteracting,world_rank);
        std::cout << "\n\n iqn: " << iqnArr_l[j] << "\n\n";
        if (world_rank==root_process){
            // First initialize the data array to be distributed across all the processes called in.
            ThreadFunctor::get_vector_mpi(totSize,is_jj,is_full,sp,vec_root_process);
            /* MPI_Allgatherv */
            if (j==0){
                vec_counts->at(0)=(num_elements_per_proc+1)*sizeOfElMPI_Allgatherv; // MPI_Allgatherv
                vec_disps->at(0)=0; 
            }
            /*  */
            /* distribute a portion of the bector to each child process */
            for(int an_id = 1; an_id < world_size; an_id++) {
                start_arr = an_id*num_elements_per_proc + 1;
                end_arr = (an_id + 1)*num_elements_per_proc;
                if((totSize - end_arr) < num_elements_per_proc) // Taking care of the remaining data.
                    end_arr = totSize - 1;
                num_elems_to_send = end_arr - start_arr + 1;
                /* MPI_Allgatherv */
                if (j==0){
                    vec_counts->at(an_id)=num_elems_to_send*sizeOfElMPI_Allgatherv;
                    vec_disps->at(an_id)=start_arr*sizeOfElMPI_Allgatherv;
                }
                /*  */
                ierr = MPI_Send( &num_elems_to_send, 1 , MPI_INT, an_id, SEND_DATA_TAG, MPI_COMM_WORLD);
                ierr = MPI_Send( (void*)(vec_root_process->data()+start_arr), sizeof(mpistruct_t)*num_elems_to_send, MPI_BYTE,
                    an_id, SEND_DATA_TAG, MPI_COMM_WORLD);
            }
            if (j==0){
                for (auto el : *vec_disps) 
                    std::cout << "vec_disps' elements: " << el << std::endl;
            }
            /* Calculate the susceptilities for the elements assigned to the root process, that is the beginning of the vector. */
            mpistruct_t tmpObj;
            for (int i=0; i<=num_elements_per_proc; i++){
                tmpObj=vec_root_process->at(i);
                #if DIM == 1
                threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j,tmpObj._sp); // Performing the calculations here...
                #elif DIM == 2
                threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j);
                #endif
                printf("(%li,%li) calculated by root process\n", tmpObj._lkt, tmpObj._lkb);
            }
            MPI_Barrier(MPI_COMM_WORLD); // Wait for the other processes to finish before moving on.
            /* Gather the results from the child processes into the externally linked matrices meant for this purpose. */
            for(int an_id = 1; an_id < world_size; an_id++) {
                ThreadFunctor::fetch_data_from_slaves(an_id,status,is_full,ierr,num_elements_per_proc,sizeOfTuple,j);
            }
        } else{
            /* Slave processes receive their part of work from the root process. */
            ierr = MPI_Recv( &num_elems_to_receive, 1, MPI_INT, 
                root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
            ierr = MPI_Recv( (void*)(vec_slave_processes->data()), sizeof(mpistruct_t)*num_elems_to_receive, MPI_BYTE, 
                root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
            /* Calculate the susceptibility of my portion of the array */
            mpistruct_t tmpObj;
            for(int i = 0; i < num_elems_to_receive; i++) {
                tmpObj = vec_slave_processes->at(i);
                #if DIM == 1
                threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j,tmpObj._sp);
                #elif DIM == 2
                threadObj(tmpObj._sp,tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._is_full,j);
                #endif
                printf("vec_slave_process el %d: %li, %li, %p\n", world_rank, tmpObj._lkt, tmpObj._lkb, (void*)vec_slave_processes);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            char chars_to_send[50];
            sprintf(chars_to_send,"vec_slave_process el %d completed", world_rank);
            /* Finally send integers to root process to notify the state of the calculations. */
            ThreadFunctor::send_messages_to_root_process(is_full,ierr,sizeOfTuple,chars_to_send,j);
        }
        if (j==0 && world_size>1){
            /* MPI_Allgatherv */
            MPI_Bcast( (void*)(vec_counts->data()), world_size, MPI_INT, root_process, MPI_COMM_WORLD );
            MPI_Bcast( (void*)(vec_disps->data()), world_size, MPI_INT, root_process, MPI_COMM_WORLD );
            /*  */
            /* Sending to all processes their respective content of gamma_tensor */
            threadObj.fetch_data_gamma_tensor_alltogether(totSizeGammaTensor,ierr,vec_counts,vec_disps,sizeOfElMPI_Allgatherv,is_full);
        }
        // To make sure not all the processes save in the same file at the same time.
        if (world_rank == root_process){
            outputChispspWeights.open(strOutputChispspWeights, std::ofstream::out | std::ofstream::app);
            outputChispspTotSus.open(strOutputChispspTotSus, std::ofstream::out | std::ofstream::app);
            if (is_full){
                if (j==0)
                    outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::out | std::ofstream::app);
                outputChispspBubble.open(strOutputChispspBubble, std::ofstream::out | std::ofstream::app);
                outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
            } else{
                if (j==0){
                    outputChispspBubble.open(strOutputChispspBubble, std::ofstream::out | std::ofstream::app);
                    outputChispspGamma.open(strOutputChispspGamma, std::ofstream::out | std::ofstream::app);
                }
            }
            for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
                for (size_t kbar=0; kbar<vecK.size(); kbar++){
                    outputChispspWeights << matWeigths(kbar,ktilde) << " ";
                    outputChispspTotSus << matTotSus(kbar,ktilde) + sus_non_interacting << " "; // Considering the non-interacting contribution...
                    if (is_full){
                        if (j==0)
                            outputChispspBubbleCorr << matCorr(kbar,ktilde) << " ";
                        outputChispspGamma << matGamma(kbar,ktilde) << " ";
                        outputChispspBubble << matMidLev(kbar,ktilde) << " ";
                    } else{
                        if (j==0){
                            outputChispspBubble << matMidLev(kbar,ktilde) << " ";
                            outputChispspGamma << matGamma(kbar,ktilde) << " ";
                        }
                    }
                }
                outputChispspWeights << "\n";
                outputChispspTotSus << "\n";
                if (is_full){
                    if (j==0)
                        outputChispspBubbleCorr << "\n";
                    outputChispspGamma << "\n";
                    outputChispspBubble << "\n";
                } else{
                    if (j==0){
                        outputChispspGamma << "\n";
                        outputChispspBubble << "\n";
                    }
                }
            }
            outputChispspWeights.close();
            outputChispspTotSus.close();
            if (is_full){
                if (j==0)
                    outputChispspBubbleCorr.close();
                outputChispspGamma.close();
                outputChispspBubble.close();
            } else{
                if (j==0){
                    outputChispspGamma.close();
                    outputChispspBubble.close();
                }
            }
        }
    }
    delete vecGammaSlaves; delete vecWeightsSlaves;
    delete vecTotSusSlaves; delete vecMidLevSlaves;
    if (is_full)
        delete vecCorrSlaves;
    delete vec_root_process;
    delete vec_slave_processes;
    ierr = MPI_Finalize();
}

#endif /* end of Thread_Utils_H_ */