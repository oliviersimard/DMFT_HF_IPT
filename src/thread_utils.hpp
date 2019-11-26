#ifndef Thread_Utils_H_
#define Thread_Utils_H_

#define PARALLEL

#include<mutex>
#include<thread>
#ifdef PARALLEL
#include<mpi.h>
#endif
#include "susceptibilities.hpp"

#define NUM_THREADS 4
#define SEND_DATA_TAG 2001
#define RETURN_DATA_TAG 2002

static std::mutex mutx;
extern arma::Mat< std::complex<double> > matGamma; // Matrices used in case parallel.
extern arma::Mat< std::complex<double> > matWeigths;
extern arma::Mat< std::complex<double> > matTotSus;
extern arma::Mat< std::complex<double> > matCorr;
extern arma::Mat< std::complex<double> > matMidLev;
extern int root_process;

template<typename T> 
inline void calculateSusceptibilitiesParallel(IPT2::SplineInline< std::complex<double> >,std::string,std::string,bool,bool,ThreadFunctor::solver_prototype);
void get_vector_mpi(size_t totSize,bool is_jj,ThreadFunctor::solver_prototype sp,std::vector<mpistruct_t>* vec_root_process);

namespace ThreadFunctor{
    enum solver_prototype : short { HF_prot, IPT2_prot };
    struct mpistruct{
        size_t _lkt, _lkb;
        bool _is_jj;
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
            void operator()(size_t ktilde, size_t kbar, bool is_jj, solver_prototype sp) const; // 1D IPT/HF
            void operator()(solver_prototype sp, size_t kbarx_m_tildex, size_t kbary_m_tildey, bool is_jj) const; // 2D IPT/HF
            std::complex<double> gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const;
            std::complex<double> gamma_oneD_spsp_IPT(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const;
            std::complex<double> gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::complex<double> gamma_twoD_spsp_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
            std::vector< std::complex<double> > buildGK1D(std::complex<double> ik, double k) const;
            std::vector< std::complex<double> > buildGK1D_IPT(std::complex<double> ik, double k) const;
            std::vector< std::complex<double> > buildGK2D(std::complex<double> ik, double kx, double ky) const;
            std::vector< std::complex<double> > buildGK2D_IPT(std::complex<double> ik, double kx, double ky) const;
            std::complex<double> gamma_twoD_spsp_full_lower(double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const;
            std::complex<double> gamma_twoD_spsp_full_lower_IPT(double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const;
            void join_all(std::vector<std::thread>& grp) const;
        private:
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
    inline std::vector< std::complex<double> > ThreadWrapper::buildGK1D(std::complex<double> ik, double k) const{
        std::vector< std::complex<double> > GK = { 1.0/( ik + _Gk._mu - epsilonk(k) - _Gk._u*_ndo_converged ), 1.0/( ik + _Gk._mu - epsilonk(k) - _Gk._u*(1.0-_ndo_converged) ) }; // UP, DOWN
        return GK;
    }

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

#ifdef PARALLEL

template<>
inline void calculateSusceptibilitiesParallel<HF::FunctorBuildGk>(HF::FunctorBuildGk Gk,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,double ndo_converged,ThreadFunctor::solver_prototype sp){
    MPI_Status status;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus;// outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspTotSus(pathToDir+customDirName+"/susceptibilities/ChispspTotSus_HF_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    //std::string strOutputChispspBubbleCorr(pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_IPT2_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    const size_t totSize=vecK.size()*vecK.size(); // Nk+1 * Nk+1
    int world_rank, world_size, start_arr, end_arr, num_elems_to_send, ierr, sender, num_elems_to_receive;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    const size_t num_elements_per_proc = totSize/world_size;
    std::vector<mpistruct_t>* vec_root_process = new std::vector<mpistruct_t>(totSize);
    std::vector<mpistruct_t>* vec_slave_processes = new std::vector<mpistruct_t>(num_elements_per_proc);
    #if DIM == 1
    HF::K_1D q(0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(Gk,q,ndo_converged);
    if (world_rank==root_process){
        // First initialize the data array to be distributed across all the processes called in.
        get_vector_mpi(totSize,is_jj,sp,vec_root_process);

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
        for (int i=0; i<num_elements_per_proc; i++){
            tmpObj=vec_root_process->at(i);
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._sp); // Performing the calculations here...
        }
        MPI_Barrier(MPI_COMM_WORLD); // Wait for the other processes to finish before moving on.
        printf("(%li,%li) calculated by root process\n", tmpObj._lkt, tmpObj._lkb);

        /* Gather the results from the child processes into the externally linked matrices meant for this purpose. */
        char chars_to_receive[30];
        for(int an_id = 1; an_id < world_size; an_id++) {
            
            ierr = MPI_Recv( chars_to_receive, 30, MPI_CHAR, MPI_ANY_SOURCE,
                  RETURN_DATA_TAG, MPI_COMM_WORLD, &status);
  
            sender = status.MPI_SOURCE;

            printf("Slave process %i returned\n", sender);
     
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
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._sp);
            printf("vec_slave_processes el %d: %li, %li, %p\n", world_rank, tmpObj._lkt, tmpObj._lkb, (void*)vec_slave_processes);
        }
        char chars_to_send[30];
        sprintf(chars_to_send,"vec_slave_processes el %d completed", world_rank);
        /* Finally send integers to root process to notify the state of the calculations. */
        
        ierr = MPI_Send( chars_to_send, 30, MPI_CHAR, root_process, RETURN_DATA_TAG, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    delete vec_root_process;
    delete vec_slave_processes;
    ierr = MPI_Finalize();
    #elif DIM == 2
    HF::K_2D qq(0.0,0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(Gk,qq,ndo_converged);
    while (it<totSize){
        if (totSize % NUM_THREADS != 0){
            if ( (totSize-it)<NUM_THREADS ){
                size_t last_it=totSize-it;
                std::vector<std::thread> tt(last_it);
                for (size_t l=0; l<last_it; l++){
                    ltot=it+l; // Have to make sure spans over the whole array of k-space.
                    lkt = static_cast<size_t>(floor(ltot/vecK.size())); // Samples the rows
                    lkb = (ltot % vecK.size()); // Samples the columns
                    std::thread t(std::ref(threadObj),sp,lkt,lkb,is_jj);
                    tt[l]=std::move(t);
                    // tt[l]=thread(threadObj,lkt,lkb,beta);
                }
                threadObj.join_all(tt);
            }
            else{
                std::vector<std::thread> tt(NUM_THREADS);
                for (size_t l=0; l<NUM_THREADS; l++){
                    ltot=it+l; // Have to make sure spans over the whole array of k-space.
                    lkt = static_cast<size_t>(floor(ltot/vecK.size()));
                    lkb = (ltot % vecK.size());
                    std::cout << "lkt: " << lkt << " lkb: " << lkb << "\n";
                    std::thread t(std::ref(threadObj),sp,lkt,lkb,is_jj);
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
                std::thread t(std::ref(threadObj),sp,lkt,lkb,is_jj);
                tt[l]=std::move(t);
                // tt[l]=thread(threadObj,lkt,lkb,beta);
            }
            threadObj.join_all(tt);
        }
        it+=NUM_THREADS;
    }
    #endif
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

template<>
inline void calculateSusceptibilitiesParallel<IPT2::DMFTproc>(IPT2::SplineInline< std::complex<double> > splInline,std::string pathToDir,std::string customDirName,bool is_full,bool is_jj,ThreadFunctor::solver_prototype sp){
    MPI_Status status;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspTotSus;// outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspTotSus(pathToDir+customDirName+"/susceptibilities/ChispspTotSus_IPT2_parallelized_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    //std::string strOutputChispspBubbleCorr(pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_IPT2_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    const size_t totSize=vecK.size()*vecK.size(); // Nk+1 * Nk+1
    int world_rank, world_size, start_arr, end_arr, num_elems_to_send, ierr, sender, num_elems_to_receive;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    const size_t num_elements_per_proc = totSize/world_size;
    std::vector<mpistruct_t>* vec_root_process = new std::vector<mpistruct_t>(totSize);
    std::vector<mpistruct_t>* vec_slave_processes = new std::vector<mpistruct_t>(num_elements_per_proc);
    #if DIM == 1
    HF::K_1D q(0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(q,splInline);
    if (world_rank==root_process){
        // First initialize the data array to be distributed across all the processes called in.
        get_vector_mpi(totSize,is_jj,sp,vec_root_process);

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
        for (int i=0; i<num_elements_per_proc; i++){
            tmpObj=vec_root_process->at(i);
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._sp); // Performing the calculations here...
        }
        MPI_Barrier(MPI_COMM_WORLD); // Wait for the other processes to finish before moving on.
        printf("(%li,%li) calculated by root process\n", tmpObj._lkt, tmpObj._lkb);

        /* Gather the results from the child processes into the externally linked matrices meant for this purpose. */
        char chars_to_receive[30];
        for(int an_id = 1; an_id < world_size; an_id++) {
            
            ierr = MPI_Recv( chars_to_receive, 30, MPI_CHAR, MPI_ANY_SOURCE,
                  RETURN_DATA_TAG, MPI_COMM_WORLD, &status);
  
            sender = status.MPI_SOURCE;

            printf("Slave process %i returned\n", sender);
     
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
            threadObj(tmpObj._lkt,tmpObj._lkb,tmpObj._is_jj,tmpObj._sp);
            printf("vec_slave_processes el %d: %li, %li, %p\n", world_rank, tmpObj._lkt, tmpObj._lkb, (void*)vec_slave_processes);
        }
        char chars_to_send[30];
        sprintf(chars_to_send,"vec_slave_processes el %d completed", world_rank);
        /* Finally send integers to root process to notify the state of the calculations. */
        
        ierr = MPI_Send( chars_to_send, 30, MPI_CHAR, root_process, RETURN_DATA_TAG, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    delete vec_root_process;
    delete vec_slave_processes;
    ierr = MPI_Finalize();
    #elif DIM == 2
    HF::K_2D qq(0.0,0.0,std::complex<double>(0.0,0.0)); // photon 4-vector
    ThreadFunctor::ThreadWrapper threadObj(qq,splInline);
    while (it<totSize){
        if (totSize % NUM_THREADS != 0){
            if ( (totSize-it)<NUM_THREADS ){
                size_t last_it=totSize-it;
                std::vector<std::thread> tt(last_it);
                for (size_t l=0; l<last_it; l++){
                    ltot=it+l; // Have to make sure spans over the whole array of k-space.
                    lkt = static_cast<size_t>(floor(ltot/vecK.size())); // Samples the rows
                    lkb = (ltot % vecK.size()); // Samples the columns
                    std::thread t(std::ref(threadObj),sp,lkt,lkb,is_jj);
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
                    std::thread t(std::ref(threadObj),sp,lkt,lkb,is_jj);
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
                std::thread t(std::ref(threadObj),sp,lkt,lkb,is_jj);
                tt[l]=std::move(t);
                // tt[l]=thread(threadObj,lkt,lkb,beta);
            }
            threadObj.join_all(tt);
        }
        it+=NUM_THREADS;
    }
    #endif
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
#endif /* PARALLEL */

#endif /* end of Thread_Utils_H_ */