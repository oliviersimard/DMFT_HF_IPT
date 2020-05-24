#include "sus_vertex_corrections_1D.hpp"

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    
    std::string inputFilename("../data/Self_energy_1D_U_10.000000_beta_50.000000_n_0.500000_N_tau_128_Nit_32.dat");
    std::string inputFilenameLoad("../data/Self_energy_1D_U_10.000000_beta_50.000000_n_0.500000_N_tau_256");
    // Choose whether current-current or spin-spin correlation function is computed.
    const bool is_jj = true;
    const bool is_single_ladder_precomputed = true;
    // Fetching results from string
    std::vector<std::string> results;
    std::vector<std::string> fetches = {"U", "beta", "N_tau"};
    
    results = get_info_from_filename(inputFilename,fetches);

    const unsigned int Ntau = 2*(unsigned int)atoi(results[2].c_str());
    const unsigned int N_q = 3;
    const unsigned int N_k = 5;
    const double beta = atof(results[1].c_str());
    const double U = atof(results[0].c_str());
    const double mu = U/2.0; // Half-filling
    const double n = 0.5;

    // beta array constructed
    std::vector<double> beta_array;
    for (size_t j=0; j<=Ntau; j++){
        double beta_tmp = j*beta/(Ntau);
        beta_array.push_back(beta_tmp);
    }
    // q_tilde_array constructed
    std::vector<double> q_tilde_array;
    for (size_t l=0; l<N_q; l++){
        double q_tilde_tmp = l*2.0*M_PI/(double)(N_q-1);
        q_tilde_array.push_back(q_tilde_tmp);
    }
    // k_t_b_array constructed
    std::vector<double> k_t_b_array;
    for (size_t l=0; l<N_k; l++){
        double q_tilde_tmp = -M_PI + l*2.0*M_PI/(double)(N_k-1);
        k_t_b_array.push_back(q_tilde_tmp);
    }

    // HDF5 business
    H5::H5File* file = nullptr;
    #ifdef INFINITE
    std::string filename("bb_1D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_isjj_"+std::to_string(is_jj)+"_infinite_ladder_sum.hdf5");
    #else
    std::string filename("bb_1D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_isjj_"+std::to_string(is_jj)+"_single_ladder_sum.hdf5");
    #endif
    const H5std_string FILE_NAME( filename );
    const unsigned int DATA_SET_DIM = Ntau;
    // The different processes cannot create more than once the file to be written in.
    if (world_rank==root_process){
        file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    }
    // Creating HDF5 file to contain the single ladder denominator
    H5::H5File* file_sl = nullptr;
    const std::string path_to_save_single_ladder_mat("./sl_data/");
    const std::string intermediate_path = "single_ladder_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n);
    const std::string filename_sl = intermediate_path+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_q);
    const H5std_string FILE_NAME_SL( path_to_save_single_ladder_mat+intermediate_path+"/"+filename_sl+".hdf5" );
    std::cout << "FILE_NAME_SL: " << FILE_NAME_SL << std::endl;
    #ifndef INFINITE
    if ( (world_rank==root_process) && is_single_ladder_precomputed ){
        try{
            // Making the directory tree 
            mkdirTree(path_to_save_single_ladder_mat+intermediate_path,"");
            std::string pattern_to_look = path_to_save_single_ladder_mat+intermediate_path+"/"+filename_sl+"*";
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
    // Getting the data
    FileData dataFromFile = get_data(inputFilename,Ntau);
    std::vector<double> wn = dataFromFile.iwn;
    std::vector<double> re = dataFromFile.re;
    std::vector<double> im = dataFromFile.im;
    std::vector< std::complex<double> > sigma_iwn;
    for (size_t i=0; i<wn.size(); i++){
        sigma_iwn.push_back(std::complex<double>(re[i],im[i]));
    }
    std::vector< std::complex<double> > iwn(wn.size());
    std::transform(wn.begin(),wn.end(),iwn.begin(),[](double d){ return std::complex<double>(0.0,d); }); // casting into array of double for cubic spline.

    // Bosonic Matsubara array
    std::vector< std::complex<double> > iqn; // for the total susceptibility
    std::vector< std::complex<double> > iqn_tilde; // for the inner loop inside Gamma.
    for (size_t j=0; j<Ntau; j++){
        iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
    }
    for (signed int j=(-static_cast<int>(Ntau/2))+1; j<(signed int)static_cast<int>(Ntau/2); j++){ // Bosonic frequencies.
        std::complex<double> matFreq(0.0 , (2.0*(double)j)*M_PI/beta );
        iqn_tilde.push_back( matFreq );
    }

    spline<double> splObj;
    std::vector<double> initVec(2*Ntau,0.0); // This data contains twice as much data to perform the interpolation
    // Spline object used to interpolate the self-energy
    IPT2::SplineInline< std::complex<double> > splInlineObj(Ntau,initVec,q_tilde_array,iwn);
    splInlineObj.loadFileSpline(inputFilenameLoad,IPT2::spline_type::linear);
    
    // One-ladder calculations
    #ifndef INFINITE
    IPT2::OneLadder< std::complex<double> > one_ladder_obj(splInlineObj,iqn,k_t_b_array,iqn_tilde,mu,U,beta);
    #else
    IPT2::InfiniteLadders< std::complex<double> > inf_ladder_obj(splInlineObj,iqn,k_t_b_array,iqn_tilde,mu,U,beta);
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
    std::vector< std::vector<MPIData> >* gathered_MPI_data = new std::vector< std::vector<MPIData> >();

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
            #ifndef INFINITE
            arma::Mat< std::complex<double> >* mat_slice_ptr = &container_sl.slice(i);
            GG_iqn = one_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_jj,is_single_ladder_precomputed,(void*)mat_slice_ptr);
            #else
            GG_iqn = inf_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_jj,is_single_ladder_precomputed);
            #endif
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "one_ladder_obj number " << i << " of world_rank " << world_rank << " lasted " << elapsed_secs << " secs to be computed" << "\n";
            gathered_MPI_data->push_back(GG_iqn);
            printf("(%li,%li) calculated by root process\n", mpidataObj.k_bar, mpidataObj.k_tilde);
            #ifndef INFINITE
            if (is_single_ladder_precomputed){ // maybe take this out of the loop
                try{
                    save_matrix_in_HDF5(container_sl.slice(i),k_t_b_array[mpidataObj.k_bar],k_t_b_array[mpidataObj.k_tilde],file_sl);
                } catch( std::runtime_error& err ){
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
                        save_matrix_in_HDF5(single_ladders_to_save,k_t_b_array[std::get<0>(n_ks)],k_t_b_array[std::get<1>(n_ks)],file_sl);
                    } catch( std::runtime_error& err ){
                        std::cerr << err.what() << "\n";
                        exit(1);
                    }
                }
            }
            #endif
        }

        std::ofstream test1("test_1.dat", std::ios::out);
        for (auto el : gathered_MPI_data->back()){
            test1 << el.cplx_data.real() << "\t\t" << el.cplx_data.imag() << "\n";
        }
        test1.close();

        std::vector< std::complex<double> > mpi_data_to_transfer_hdf5(Ntau);
        for (size_t l=0; l<gathered_MPI_data->size(); l++){
            std::vector<MPIData> mpi_data_hdf5_tmp = gathered_MPI_data->at(l);
            std::string DATASET_NAME("kbar_"+std::to_string(k_t_b_array[mpi_data_hdf5_tmp[0].k_bar])+"ktilde_"+std::to_string(k_t_b_array[mpi_data_hdf5_tmp[0].k_tilde]));
            std::cout << "DATASET_NAME: " << DATASET_NAME << std::endl;
            // extracting the std::complex<double> data from the MPI struct
            std::transform(mpi_data_hdf5_tmp.begin(),mpi_data_hdf5_tmp.end(),mpi_data_to_transfer_hdf5.begin(),[](MPIData d){ return d.cplx_data; });
            writeInHDF5File(mpi_data_to_transfer_hdf5, file, DATA_SET_DIM, DATASET_NAME);
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
            #ifndef INFINITE
            arma::Mat< std::complex<double> >* mat_slice_ptr = &container_sl.slice(i);
            GG_iqn = one_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_jj,is_single_ladder_precomputed,(void*)mat_slice_ptr);
            #else
            GG_iqn = inf_ladder_obj(mpidataObj.k_bar,mpidataObj.k_tilde,is_jj,is_single_ladder_precomputed);
            #endif
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "one_ladder_obj number " << i << " of world_rank " << world_rank << " lasted " << elapsed_secs << " secs to be computed" << "\n";
            gathered_MPI_data->push_back(GG_iqn);
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

    
    // Should be a loop over the external momentum from this point on...
    arma::Mat< std::complex<double> > G_k_bar_q_tilde_iwn(q_tilde_array.size(),iwn.size());
    arma::Mat<double> G_k_bar_q_tilde_tau(q_tilde_array.size(),Ntau+1);
    std::vector<double> FFT_k_bar_q_tilde_tau;
    constexpr double k_bar = M_PI/2.0;

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            G_k_bar_q_tilde_iwn(l,n_bar) = 1.0/( ( iwn[n_bar] ) + mu - epsilonk(k_bar-q_tilde_array[l]) - splInlineObj.calculateSpline(iwn[n_bar].imag()) );
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

    /* TEST G(-tau) */
    std::ofstream test2("test_1_corr_no_cubic_spline.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test2 << beta_array[j] << "  " << -1.0*G_k_bar_q_tilde_tau(2,Ntau-j) << "\n";
    }
    test2.close();

    if (world_rank==root_process){
        delete file;
        if (is_single_ladder_precomputed){
            delete file_sl;
        }
    }
    delete gathered_MPI_data;
    MPI_Type_free(&MPI_Data_struct_t);

    ierr = MPI_Finalize();

    return 0;

}
