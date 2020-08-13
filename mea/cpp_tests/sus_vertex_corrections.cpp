#include "sus_vertex_corrections.hpp"

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    
    #ifdef NCA
    std::string inputFilename("../../../NCA_OCA/data_1D_test_NCA_damping_0.000000/1D_U_14.000000_beta_6.000000_n_0.500000_Ntau_4096/SE_1D_NCA_AFM_U_14.000000_beta_6.000000_N_tau_4096_h_0.000000_Nit_53.dat");
    #else
    std::string inputFilename("../../data/1D_U_8.000000_beta_7.000000_n_0.500000_N_tau_128/Self_energy_1D_U_8.000000_beta_7.000000_n_0.500000_N_tau_128_Nit_25.dat");
    std::string inputFilenameLoad("../../build4/data/1D_U_8.000000_beta_7.000000_n_0.500000_N_tau_512/Self_energy_1D_U_8.000000_beta_7.000000_n_0.500000_N_tau_512");
    #endif 
    // Current-current and spin-spin correlation functions are computed at same time.
    const bool is_single_ladder_precomputed = true;
    // Fetching results from string
    std::vector<std::string> results;
    std::vector<std::string> fetches = {"U", "beta", "N_tau"};
    
    results = get_info_from_filename(inputFilename,fetches);

    #ifndef NCA
    const unsigned int Ntau = 2*(unsigned int)atoi(results[2].c_str());
    #else
    const size_t NCA_Ntau = 2*(unsigned int)atoi(results[2].c_str()); // size of the full NCA calculation
    const size_t Ntau = 2*128; // One has to assume that the number of Matsubara frequencies defining the self-energy is sufficient.
    #endif
    // Has to be a power of two as well: this is no change from IPT.
    assert(Ntau%2==0);
    const unsigned int N_q = 31;
    const unsigned int N_k = 15;
    const double beta = atof(results[1].c_str());
    const double U = atof(results[0].c_str());
    const double mu = U/2.0; // Half-filling. Depending whether AFM-PM solution is loaded or not, mu=U/2 in PM only scenario and mu=0.0 in AFM-PM scenario.
    const double n = 0.5;

    // beta array constructed
    std::vector<double> beta_array;
    for (size_t j=0; j<=Ntau; j++){
        beta_array.push_back( j*beta/(Ntau) );
    }
    // q_tilde_array constructed
    std::vector<double> q_tilde_array;
    double k_tmp;
    for (size_t l=0; l<N_q; l++){
        k_tmp = l*2.0*M_PI/(double)(N_q-1);
        q_tilde_array.push_back(k_tmp);
    }
    // k_t_b_array constructed
    std::vector<double> k_t_b_array;
    for (size_t l=0; l<N_k; l++){
        k_tmp = -M_PI + l*2.0*M_PI/(double)(N_k-1);
        k_t_b_array.push_back(k_tmp);
    }

    // HDF5 business
    H5::H5File* file = nullptr;
    #ifdef INFINITE
    #ifdef NCA
    std::string filename("bb_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_NCA_infinite_ladder_sum.hdf5");
    #else
    std::string filename("bb_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_infinite_ladder_sum.hdf5");
    #endif
    #else
    #ifdef NCA
    std::string filename("bb_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_NCA_single_ladder_sum.hdf5");
    #else
    std::string filename("bb_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_single_ladder_sum.hdf5");
    #endif
    #endif
    const H5std_string FILE_NAME( filename );
    // The different processes cannot create more than once the file to be written in.
    if (world_rank==root_process){
        file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    }
    // Creating HDF5 file to contain the single ladder denominator
    H5::H5File* file_sl = nullptr;
    const std::string path_to_save_single_ladder_mat("./sl_data/");
    const std::string intermediate_path = "single_ladder_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n);
    #ifdef NCA
    const std::string filename_sl = intermediate_path+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_k)+"_Nq_"+std::to_string(N_q)+"_NCA";
    #else
    const std::string filename_sl = intermediate_path+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_k)+"_Nq_"+std::to_string(N_q);
    #endif
    const H5std_string FILE_NAME_SL( path_to_save_single_ladder_mat+intermediate_path+"/"+filename_sl+".hdf5" );
    std::cout << "FILE_NAME_SL: " << FILE_NAME_SL << std::endl;
    #ifndef INFINITE
    if ( (world_rank==root_process) && is_single_ladder_precomputed ){
        try{
            // Making the directory tree 
            mkdirTree(path_to_save_single_ladder_mat+intermediate_path,"");
            std::string pattern_to_look = path_to_save_single_ladder_mat+intermediate_path+"/"+filename_sl+".*";
            std::cout << pattern_to_look << std::endl;
            std::vector<std::string> glob_files_found;
            // Looking if the HDF5 file about to be created exists already
            glob_files_found = glob(pattern_to_look);
            char buf_chr[filename_sl.length()+30];
            sprintf(buf_chr,"File %s exists already!!",filename_sl.c_str());
            if (glob_files_found.size()>0){
                throw std::runtime_error( buf_chr );
            }
            file_sl = new H5::H5File( FILE_NAME_SL, H5F_ACC_TRUNC );
        } catch(std::runtime_error& err){
            std::cerr << err.what() << "\n";
            exit(1);
        }
    }
    #endif
    // Getting the data of the self-energy
    FileData dataFromFile;  
    #ifdef NCA
    dataFromFile = get_data(inputFilename,NCA_Ntau);
    #else
    int largest_Nit = get_largest_Nit(inputFilenameLoad); // fetching the last iteration only. Have to follow some standard notation
    dataFromFile = get_data(inputFilenameLoad+"_Nit_"+std::to_string(largest_Nit)+".dat",4*Ntau);
    #endif
    std::vector<double> wn = std::move(dataFromFile.iwn);
    std::vector<double> re = std::move(dataFromFile.re);
    std::vector<double> im = std::move(dataFromFile.im);
    #ifdef NCA
    std::vector< std::complex<double> > sigma_iwn(4*Ntau);
    std::vector< std::complex<double> > iwn(Ntau); // this size has got to be 1/4 the size of the self-energy loaded...
    try{
        if (wn.size()<4*Ntau)
            throw std::out_of_range("Problem in the truncation of the self-energy in loading process.");
        
        for (size_t i=0; i<4*Ntau; i++){
            sigma_iwn[i] = std::complex<double>(re[i+wn.size()/2-2*Ntau],im[i+wn.size()/2-2*Ntau]);
        }
    } catch(const std::out_of_range& err){
        std::cerr << err.what() << "\n";
        exit(1);
    }
    #else
    std::vector< std::complex<double> > sigma_iwn(4*Ntau);
    std::vector< std::complex<double> > iwn(Ntau);
    for (size_t i=0; i<wn.size(); i++){
        sigma_iwn[i]=std::complex<double>(re[i],im[i]);
    }
    #endif
    std::transform(wn.data()+wn.size()/2-Ntau/2,wn.data()+wn.size()/2+Ntau/2,iwn.data(),[](double d){ return std::complex<double>(0.0,d); });

    // Bosonic Matsubara array
    std::vector< std::complex<double> > iqn; // for the total susceptibility
    std::vector< std::complex<double> > iqn_tilde; // for the inner loop inside Gamma.
    for (size_t j=0; j<Ntau; j++){ // change Ntau for lower value to decrease time when testing...
        iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
    }
    const unsigned int DATA_SET_DIM = iqn.size(); // this corresponds to the length of the bosonic Matsubara array

    for (signed int j=(-static_cast<int>(Ntau/2))+1; j<(signed int)static_cast<int>(Ntau/2); j++){ // Bosonic frequencies.
        iqn_tilde.push_back( std::complex<double>(0.0 , (2.0*(double)j)*M_PI/beta ) );
    }

    #ifndef NCA
    std::vector<double> initVec(4*Ntau,0.0); // This data contains four times as much data to perform the interpolation
    // Spline object used to interpolate the self-energy
    IPT2::SplineInline< std::complex<double> > splInlineObj(2*Ntau,initVec,q_tilde_array,iwn);
    splInlineObj.loadFileSpline(inputFilenameLoad,IPT2::spline_type::linear);
    #else
    std::vector<double> initVec(NCA_Ntau,0.0); // This data contains twice as much data to perform the interpolation
    // Spline object used to interpolate the self-energy
    IPT2::SplineInline< std::complex<double> > splInlineObj(NCA_Ntau/2,initVec,q_tilde_array,iwn);
    splInlineObj.loadFileSpline(inputFilename,IPT2::spline_type::linear);
    #endif

    // /************************************************* TESTING ********************************************************/
    std::ofstream ofNCA("NCA_test_SE.dat", std::ios::out);
    std::ofstream ofSE1("NCA_test_SE_spl.dat",std::ios::out);
    std::ofstream ofNCAOG("NCA_test_SE_og.dat", std::ios::out);
    // std::cout << (2*Ntau-1)+iwn.size()-1-0 << ": " << sigma_iwn[(2*Ntau-1)+iwn.size()-1-0] << " and " << (2*Ntau-1)+iwn.size()-1-0+iqn.size()-1 << ": " << sigma_iwn[(2*Ntau-1)+iwn.size()-1-0+iqn.size()-1] << std::endl;
    // std::cout << (iwn[iwn.size()-1]-iqn_tilde[0]).imag() << ": " << wn[wn.size()/2-2*Ntau+(2*Ntau-1)+iwn.size()-1-0] << " and " << (iwn[iwn.size()-1]-iqn_tilde[0]+iqn[iqn.size()-1]).imag() << ": " << wn[wn.size()/2-2*Ntau+(2*Ntau-1)+iwn.size()-1-0+iqn.size()-1] << std::endl;
    // std::function<std::complex<double>(double)> int_k_1D;
    // const Integrals intObj;
    // auto ikn_tilde = iwn[5], ikn_bar = iwn[52];
    // double kk_tilde = M_PI, kk_bar = M_PI/3.2;
    // for (size_t j=0; j<iwn.size(); j++){
    //     int_k_1D = [&](double k){
    //         return ( 1.0 / ( ikn_bar-iwn[j]+iqn[12] + U/2.0 - epsilonk(kk_bar-k) - sigma_iwn[(iwn.size())+52-j+12] ) 
    //             )*( 1.0 / ( ikn_tilde-iwn[j]+iqn[12] + U/2.0 - epsilonk(kk_tilde-k) - sigma_iwn[(iwn.size())+5-j+12] ) 
    //             );
    //         // return ( 1.0 / ( ikn_tilde-iqn_tilde[j] + U/2.0 - epsilonk(kk_tilde-k) - sigma_iwn[(2*Ntau-1)+5-j] )// sigma_iwn[(2*Ntau-1)+15-j] ) 
    //         // )*( 1.0 / ( ikn_bar-iqn_tilde[j]+iqn[12] + U/2.0 - epsilonk(kk_bar-k) - sigma_iwn[(2*Ntau-1)+52-j+12] ) 
    //         // );
    //         // return ( 1.0/( ikn_tilde + U/2.0 - epsilonk(kk_tilde) - sigma_iwn[3*iwn.size()/2+11] )
    //         //     )*( 1.0/( ikn_tilde-iqn[12] + U/2.0 - epsilonk(kk_tilde) - sigma_iwn[3*iwn.size()/2+11-12] )
    //         //     )*( 1.0/( ikn_bar + U/2.0 - epsilonk(kk_bar) - sigma_iwn[3*iwn.size()/2+33] )
    //         //     )*( 1.0/( ikn_bar+iqn[12] + U/2.0 - epsilonk(kk_bar) - sigma_iwn[3*iwn.size()/2+33+12] )
    //             // );
    //     };
    //     ofNCA << iwn[j].imag() << "  " << 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI) << "\n";
    // }
    // ofNCA.close();
    // for (size_t j=0; j<iqn_tilde.size(); j++){
    //     int_k_1D = [&](double k){
    //         return ( 1.0 / ( ikn_bar-iwn[j]+iqn[12] + U/2.0 - epsilonk(kk_bar-k) - splInlineObj.calculateSpline( (ikn_bar-iwn[j]+iqn[12]).imag() ) ) 
    //             )*( 1.0 / ( ikn_tilde-iwn[j]+iqn[12] + U/2.0 - epsilonk(kk_tilde-k) - splInlineObj.calculateSpline( (ikn_tilde-iwn[j]+iqn[12]).imag() ) ) 
    //             );
    //     //     return ( 1.0 / ( ikn_tilde-iqn_tilde[j] + U/2.0 - epsilonk(kk_tilde-k) - splInlineObj.calculateSpline( (ikn_tilde-iqn_tilde[j]).imag() ) ) 
    //     //     )*( 1.0 / ( ikn_bar-iqn_tilde[j]+iqn[12] + U/2.0 - epsilonk(kk_bar-k) - splInlineObj.calculateSpline( (ikn_bar-iqn_tilde[j]+iqn[12]).imag() ) ) 
    //         // );
    //         // return ( 1.0/( ikn_tilde + U/2.0 - epsilonk(kk_tilde) - splInlineObj.calculateSpline( (ikn_tilde).imag() ) )
    //         //     )*( 1.0/( ikn_tilde-iqn[12] + U/2.0 - epsilonk(kk_tilde) - splInlineObj.calculateSpline( (ikn_tilde-iqn[12]).imag() ) )
    //         //     )*( 1.0/( ikn_bar + U/2.0 - epsilonk(kk_bar) - splInlineObj.calculateSpline( (ikn_bar).imag() ) )
    //         //     )*( 1.0/( ikn_bar+iqn[12] + U/2.0 - epsilonk(kk_bar) - splInlineObj.calculateSpline( (ikn_bar+iqn[12]).imag() ) )
    //         //     );
    //     };
    //     ofSE1 << iwn[j].imag() << "  " << 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI) << "\n";
    // }
    // ofSE1.close();
    size_t iii = 0;
    for (size_t j=0; j<iwn.size(); j++){
        // ofNCA << (iwn[iii]-iwn[j]+iqn[12]).imag() << "  " << sigma_iwn[(2*iwn.size()-1)+iii-j+12].real() << "  " << sigma_iwn[(2*iwn.size()-1)+iii-j+12].imag() << "\n";
        ofNCA << (iwn[j]+iqn[iii]).imag() << "  " << sigma_iwn[(3*Ntau/2)+j+iii].real() << "  " << sigma_iwn[(3*Ntau/2)+j+iii].imag() << "\n";
    }
    ofNCA.close();
    for (size_t j=0; j<iwn.size(); j++){
        // std::cout << splInlineObj.calculateSpline( (iwn[iii]-iqn_tilde[j]).imag() );
        // ofSE1 << (iwn[iii]-iwn[j]+iqn[12]).imag() << "  " << splInlineObj.calculateSpline( (iwn[iii]-iwn[j]+iqn[12]).imag() ).real() << "  " << splInlineObj.calculateSpline( (iwn[iii]-iwn[j]+iqn[12]).imag() ).imag() << "\n";
        ofSE1 << (iwn[j]+iqn[iii]).imag() << "  " << splInlineObj.calculateSpline( (iwn[j]+iqn[iii]).imag() ).imag() << "\n";
    }
    ofSE1.close();
    for (size_t i=0; i<sigma_iwn.size(); i++){
        ofNCAOG << sigma_iwn[i].real() << "  " << sigma_iwn[i].imag() << "\n";
    }
    ofNCAOG.close();
    // exit(0);
    /**************************************************** TESTING ********************************************************/
    
    // One-ladder calculations
    #ifndef INFINITE
    IPT2::OneLadder< std::complex<double> > one_ladder_obj(splInlineObj,sigma_iwn,iqn,k_t_b_array,iqn_tilde,mu,U,beta);
    #else
    IPT2::InfiniteLadders< std::complex<double> > inf_ladder_obj(splInlineObj,sigma_iwn,iqn,k_t_b_array,iqn_tilde,mu,U,beta);
    if (is_single_ladder_precomputed){
        IPT2::InfiniteLadders< std::complex<double> >::_FILE_NAME = FILE_NAME_SL;
    }
    #endif
    
    // Setting MPI stuff up
    constexpr size_t tot_k_size = N_k*N_k;
    const size_t num_elem_per_proc = (world_size != 1) ? static_cast<size_t>(tot_k_size/world_size) : tot_k_size-1;
    std::cout << "num_elem_per_proc: " << num_elem_per_proc << "\n";
    MPI_Status status;
    std::vector<MPIData> GG_iqn;
    
    std::vector<MPIData> vec_to_processes;
    int start, end, num_elem_to_send, num_elem_to_receive, ierr;
    // Committing custom data type
    MPI_Datatype MPI_Data_struct_t;
    IPT2::create_mpi_data_struct(MPI_Data_struct_t);
    //std::vector< std::vector<MPIData> >* gathered_MPI_data = new std::vector< std::vector<MPIData> >();
    auto_ptr< std::vector< std::vector<MPIData> > > gathered_MPI_data(new std::vector< std::vector<MPIData> >());

    // Dispatching the tasks across the processes called in
    if (world_rank==root_process){
        IPT2::set_vector_processes(&vec_to_processes,N_k);
        for (auto el : vec_to_processes){
            std::cout << el.k_bar << " " << el.k_tilde << std::endl;
        }
        std::cout << "world size: " << world_size << "\n";
        for (int an_id=1; an_id<world_size; an_id++){ // This loop is skipped if world_size=1
            start = an_id*num_elem_per_proc + 1;
            end = (an_id + 1)*num_elem_per_proc;
            if((tot_k_size - start) < num_elem_per_proc) // Taking care of the remaining data.
                end = tot_k_size - 1;
            num_elem_to_send = end - start + 1;
            std::cout << "num elem to send for id " << an_id << " is " << num_elem_to_send << "\n";
            ierr = MPI_Send( &num_elem_to_send, 1 , MPI_INT, an_id, SEND_NUM_TO_SLAVES, MPI_COMM_WORLD );
            ierr = MPI_Send( (void*)(vec_to_processes.data()+start), num_elem_to_send, MPI_Data_struct_t,
                    an_id, SEND_DATA_TAG, MPI_COMM_WORLD );
            
        }
        // Root process now deals with its share of the workload
        #ifndef INFINITE
        arma::Cube< std::complex<double> > container_sl(Ntau,Ntau,num_elem_per_proc+1);
        #endif
        for (size_t i=0; i<=num_elem_per_proc; i++){
            auto mpidataObj = vec_to_processes[i];
            clock_t begin = clock();
            try {
                #ifndef INFINITE
                arma::Mat< std::complex<double> >* mat_slice_ptr = &container_sl.slice(i);
                GG_iqn = one_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_single_ladder_precomputed,(void*)mat_slice_ptr);
                #else
                GG_iqn = inf_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_single_ladder_precomputed);
                #endif
            } catch(const std::invalid_argument& err){
                std::cerr << err.what() << "\n";
                exit(1);
            }
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "one_ladder_obj number " << i << " of world_rank " << world_rank << " lasted " << elapsed_secs << " secs to be computed" << "\n";
            gathered_MPI_data->push_back(std::move(GG_iqn));
            printf("(%li,%li) calculated by root process\n", mpidataObj.k_bar, mpidataObj.k_tilde);
            #ifndef INFINITE
            if (is_single_ladder_precomputed){ // maybe take this out of the loop
                try{
                    #if DIM == 1
                    H5std_string  DATASET_NAME( std::string("kbar_")+std::to_string(k_t_b_array[mpidataObj.k_bar])+std::string("ktilde_")+std::to_string(k_t_b_array[mpidataObj.k_tilde]) );
                    #else
                    H5std_string  DATASET_NAME( std::string("ktildex_")+std::to_string(k_t_b_array[mpidataObj.k_bar])+std::string("ktildey_")+std::to_string(k_t_b_array[mpidataObj.k_tilde]) );
                    #endif
                    save_matrix_in_HDF5(container_sl.slice(i),DATASET_NAME,file_sl);
                } catch(const std::runtime_error& err ){
                    std::cerr << err.what() << "\n";
                    exit(1);
                }
            }
            #endif
        }
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        // Fetch data from the slaves
        MPIDataReceive mpi_data_receive;
        int recv_root_num_elem;
        #ifndef INFINITE
        ArmaMPI< std::complex<double> > armaMatObj(Ntau,Ntau);
        arma::Mat< std::complex<double> > single_ladders_to_save;
        #endif
        int get_size;
        for (int an_id=1; an_id<world_size; an_id++){ // This loop is skipped if world_size=1
            ierr = MPI_Recv( &recv_root_num_elem, 1, MPI_INT, 
                an_id, RETURN_NUM_RECV_TO_ROOT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int l=0; l<recv_root_num_elem; l++){
                ierr = MPI_Probe(an_id,l+SHIFT_TO_DIFFERENTIATE_TAGS,MPI_COMM_WORLD,&status); // Peeking the data received and comparing to num_elem_to_receive
                ierr = MPI_Get_count(&status, MPI_Data_struct_t, &get_size);
                mpi_data_receive.size = (size_t)get_size;
                std::cout << "size mpi_data_receive: " << mpi_data_receive.size << std::endl;
                mpi_data_receive.data_struct = (MPIData*)malloc(mpi_data_receive.size*sizeof(MPIData));
                ierr = MPI_Recv((void*)mpi_data_receive.data_struct,mpi_data_receive.size,MPI_Data_struct_t,an_id,l+SHIFT_TO_DIFFERENTIATE_TAGS,MPI_COMM_WORLD,&status);
                gathered_MPI_data->push_back( std::vector<MPIData>(mpi_data_receive.data_struct,mpi_data_receive.data_struct+mpi_data_receive.size) );
                free(mpi_data_receive.data_struct);
            }
            #ifndef INFINITE
            // Precompute the single ladder denominator
            if (is_single_ladder_precomputed){
                // Gathering the tags from the slaves
                std::vector<int> returned_tags_from_slaves(recv_root_num_elem);
                MPI_Recv( (void*)(returned_tags_from_slaves.data()), recv_root_num_elem, MPI_INT, an_id, RETURN_TAGS_TO_ROOT, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
                for (auto tag : returned_tags_from_slaves){
                    single_ladders_to_save = armaMatObj.recv_Arma_mat_MPI(tag,an_id);
                    auto n_ks = inverse_Cantor_pairing(tag); // n_k_bar, n_k_tilde
                    std::cout << "n_bar: " << std::get<0>(n_ks) << " and n_tilde: " << std::get<1>(n_ks) << std::endl;
                    try{
                        #if DIM == 1
                        H5std_string  DATASET_NAME( std::string("kbar_")+std::to_string(k_t_b_array[std::get<0>(n_ks)])+std::string("ktilde_")+std::to_string(k_t_b_array[std::get<1>(n_ks)]) );
                        #else
                        H5std_string  DATASET_NAME( std::string("ktildex_")+std::to_string(k_t_b_array[std::get<0>(n_ks)])+std::string("ktildey_")+std::to_string(k_t_b_array[std::get<1>(n_ks)]) );
                        #endif
                        save_matrix_in_HDF5(single_ladders_to_save,DATASET_NAME,file_sl);
                    } catch( std::runtime_error& err ){
                        std::cerr << err.what() << "\n";
                        exit(1);
                    }
                }
            }
            #endif
        }

        std::ofstream test1("test_susceptibilities_not_flattened_1D.dat", std::ios::out);
        for (auto el : gathered_MPI_data->back()){
            test1 << el.cplx_data_jj.real() << "\t\t" << el.cplx_data_jj.imag() << "\t\t" << el.cplx_data_szsz.real() << "\t\t" << el.cplx_data_szsz.imag() << "\n";
        }
        test1.close();

        std::vector< std::complex<double> > mpi_data_to_transfer_hdf5_jj(iqn.size());
        std::vector< std::complex<double> > mpi_data_to_transfer_hdf5_szsz(iqn.size());
        for (size_t l=0; l<gathered_MPI_data->size(); l++){
            std::vector<MPIData> mpi_data_hdf5_tmp = gathered_MPI_data->at(l);
            #if DIM == 1
            std::string DATASET_NAME("kbar_"+std::to_string(k_t_b_array[mpi_data_hdf5_tmp[0].k_bar])+"ktilde_"+std::to_string(k_t_b_array[mpi_data_hdf5_tmp[0].k_tilde]));
            #else
            std::string DATASET_NAME("ktildex_"+std::to_string(k_t_b_array[mpi_data_hdf5_tmp[0].k_bar])+"ktildey_"+std::to_string(k_t_b_array[mpi_data_hdf5_tmp[0].k_tilde]));
            #endif
            std::cout << "DATASET_NAME: " << DATASET_NAME << std::endl;
            // extracting the std::complex<double> datasets from the MPI struct
            std::transform(mpi_data_hdf5_tmp.begin(),mpi_data_hdf5_tmp.end(),mpi_data_to_transfer_hdf5_jj.begin(),[](MPIData d){ return d.cplx_data_jj; });
            std::transform(mpi_data_hdf5_tmp.begin(),mpi_data_hdf5_tmp.end(),mpi_data_to_transfer_hdf5_szsz.begin(),[](MPIData d){ return d.cplx_data_szsz; });
            writeInHDF5File(mpi_data_to_transfer_hdf5_jj, mpi_data_to_transfer_hdf5_szsz, file, DATA_SET_DIM, DATASET_NAME);
        }

    } else{
        /* Slave processes receive their part of work from the root process. */
        ierr = MPI_Recv( &num_elem_to_receive, 1, MPI_INT, 
            root_process, SEND_NUM_TO_SLAVES, MPI_COMM_WORLD, &status);
        std::vector<MPIData> vec_for_slaves(num_elem_to_receive);
        ierr = MPI_Recv( (void*)(vec_for_slaves.data()), num_elem_to_receive, MPI_Data_struct_t, 
            root_process, SEND_DATA_TAG, MPI_COMM_WORLD, &status);
        /* Calculate the sum of the portion of the array */
        #ifndef INFINITE
        arma::Cube< std::complex<double> > container_sl(Ntau,Ntau,num_elem_to_receive); // n_rows, n_cols, n_slices
        #endif
        for(size_t i = 0; i < num_elem_to_receive; i++) {
            auto mpidataObj = vec_for_slaves[i];
            clock_t begin = clock();
            try{
                #ifndef INFINITE
                arma::Mat< std::complex<double> >* mat_slice_ptr = &container_sl.slice(i);
                GG_iqn = one_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_single_ladder_precomputed,(void*)mat_slice_ptr);
                #else
                GG_iqn = inf_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_single_ladder_precomputed);
                #endif
            } catch(const std::invalid_argument& err){
                std::cerr << err.what() << "\n";
                exit(1);
            }

            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "one_ladder_obj number " << i << " of world_rank " << world_rank << " lasted " << elapsed_secs << " secs to be computed" << "\n";
            gathered_MPI_data->push_back(std::move(GG_iqn));
            printf("(%li, %li) calculated by slave_process %d\n", mpidataObj.k_bar, mpidataObj.k_tilde, world_rank); //, (void*)&vec_for_slaves);
        }
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        ierr = MPI_Send( &num_elem_to_receive, 1 , MPI_INT, root_process, RETURN_NUM_RECV_TO_ROOT, MPI_COMM_WORLD );
        for (int l=0; l<num_elem_to_receive; l++){
            ierr = MPI_Send( (void*)(gathered_MPI_data->at(l).data()), gathered_MPI_data->at(l).size(), MPI_Data_struct_t, root_process, l+SHIFT_TO_DIFFERENTIATE_TAGS, MPI_COMM_WORLD);
        }
        #ifndef INFINITE
        if (is_single_ladder_precomputed){
            // Sending tags to root.
            ierr = MPI_Send( (void*)(one_ladder_obj.tag_vec.data()), num_elem_to_receive, MPI_INT, root_process, RETURN_TAGS_TO_ROOT, MPI_COMM_WORLD );
            for (size_t t=0; t<one_ladder_obj.tag_vec.size(); t++){
                MPI_Send(container_sl.slice(t).memptr(),Ntau*Ntau,MPI_CXX_DOUBLE_COMPLEX,root_process,one_ladder_obj.tag_vec[t],MPI_COMM_WORLD);
            }
        }
        #endif
    }

    std::vector<double> FFT_k_bar_q_tilde_tau;
    constexpr double k_bar = M_PI/2.0;
    #if DIM == 1
    // Should be a loop over the external momentum from this point on...
    arma::Mat< std::complex<double> > G_k_bar_q_tilde_iwn(q_tilde_array.size(),iwn.size());
    arma::Mat<double> G_k_bar_q_tilde_tau(q_tilde_array.size(),Ntau+1);

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            G_k_bar_q_tilde_iwn(l,n_bar) = 1.0/( ( iwn[n_bar] ) + mu - epsilonk(k_bar-q_tilde_array[l]) - sigma_iwn[3*iwn.size()/2+n_bar] );
        }
    }

    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        for (size_t l=0; l<q_tilde_array.size(); l++){
            // Substracting the tail of the Green's function
            G_k_bar_q_tilde_iwn(l,n_bar) -= 1.0/(iwn[n_bar]) + epsilonk(q_tilde_array[l])/iwn[n_bar]/iwn[n_bar]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }

    // FFT of G(iwn) --> G(tau)
    for (size_t l=0; l<q_tilde_array.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_bar_q_tilde_iwn(l,arma::span::all).begin(),G_k_bar_q_tilde_iwn(l,arma::span::all).end());
        FFT_k_bar_q_tilde_tau = IPT2::get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
        for (size_t i=0; i<beta_array.size(); i++){
            G_k_bar_q_tilde_tau(l,i) = FFT_k_bar_q_tilde_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(q_tilde_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]));
        }
    }
    #elif DIM == 2
    // Should be a loop over the external momentum from this point on...
    arma::Cube< std::complex<double> > G_k_bar_q_tilde_iwn(q_tilde_array.size(),q_tilde_array.size(),iwn.size());
    arma::Cube<double> G_k_bar_q_tilde_tau(q_tilde_array.size(),q_tilde_array.size(),Ntau+1);

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            for (size_t m=0; m<q_tilde_array.size(); m++){
                G_k_bar_q_tilde_iwn.at(l,m,n_bar) = 1.0/( ( iwn[n_bar] ) + mu - epsilonk(q_tilde_array[l]+k_bar,q_tilde_array[m]+k_bar) - splInlineObj.calculateSpline(iwn[n_bar].imag()) );
            }
        }
    }

    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        for (size_t l=0; l<q_tilde_array.size(); l++){
            for (size_t m=0; m<q_tilde_array.size(); m++){
                // Substracting the tail of the Green's function
                G_k_bar_q_tilde_iwn.at(l,m,n_bar) -= 1.0/(iwn[n_bar]) + epsilonk(q_tilde_array[l]+k_bar,q_tilde_array[m]+k_bar)/iwn[n_bar]/iwn[n_bar]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
            }
        }
    }

    // FFT of G(iwn) --> G(tau)
    for (size_t l=0; l<q_tilde_array.size(); l++){
        for (size_t m=0; m<q_tilde_array.size(); m++){
            std::vector< std::complex<double> > G_iwn_k_slice(G_k_bar_q_tilde_iwn(arma::span(l,l),arma::span(m,m),arma::span::all).begin(),G_k_bar_q_tilde_iwn(arma::span(l,l),arma::span(m,m),arma::span::all).end());
            FFT_k_bar_q_tilde_tau = IPT2::get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
            for (size_t i=0; i<beta_array.size(); i++){
                G_k_bar_q_tilde_tau(l,m,i) = FFT_k_bar_q_tilde_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(q_tilde_array[l]+k_bar,q_tilde_array[m]+k_bar); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]));
            }
        }
    }
    #endif
    /* TEST G(-tau) */
    std::ofstream test2("test_calculate_spline_Green_function_1D.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        #if DIM == 1
        test2 << beta_array[j] << "  " << -1.0*G_k_bar_q_tilde_tau(2,Ntau-j) << "\n";
        #elif DIM == 2
        test2 << beta_array[j] << "  " << -1.0*G_k_bar_q_tilde_tau(2,2,Ntau-j) << "\n";
        #endif
    }
    test2.close();

    if (world_rank==root_process){
        delete file;
        if (is_single_ladder_precomputed){
            delete file_sl;
        }
    }
    // delete gathered_MPI_data;
    MPI_Type_free(&MPI_Data_struct_t);

    ierr = MPI_Finalize();

    return 0;

}
