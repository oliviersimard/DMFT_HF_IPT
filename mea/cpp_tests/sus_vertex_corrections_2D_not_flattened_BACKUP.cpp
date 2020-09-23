#include "sus_vertex_corrections_2D_not_flattened.hpp"

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    
    std::string inputFilename("../../data/2D_U_10.000000_beta_8.000000_n_0.500000_N_tau_128/Self_energy_2D_U_10.000000_beta_8.000000_n_0.500000_N_tau_128_Nit_18.dat");
    std::string inputFilenameLoad("../../data/2D_U_10.000000_beta_8.000000_n_0.500000_N_tau_256/Self_energy_2D_U_10.000000_beta_8.000000_n_0.500000_N_tau_256");
    // Fetching results from string
    std::vector<std::string> results;
    std::vector<std::string> fetches = {"U", "beta", "N_tau"};
    
    results = get_info_from_filename(inputFilename,fetches);

    const unsigned int Ntau = 2*(unsigned int)atoi(results[2].c_str());
    const unsigned int N_q = 3;
    const unsigned int N_k = 3;
    const double beta = atof(results[1].c_str());
    const double U = atof(results[0].c_str());
    const double mu = U/2.0; // Half-filling
    // const double n = 0.5;

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
        #ifdef TRIGTABLES
        // building finer cosine table
        sine_table_finer.push_back(std::sin(q_tilde_tmp));
        cosine_table_finer.push_back(std::cos(q_tilde_tmp));
        #endif
    }
    // k_t_b_array constructed
    std::vector<double> k_tilde_m_k_bar;
    for (size_t l=0; l<N_k; l++){
        double k_tmp = -M_PI + l*2.0*M_PI/(double)(N_k-1);
        k_tilde_m_k_bar.push_back(k_tmp);
        sine_table.push_back(std::sin(k_tmp));
        cosine_table.push_back(std::cos(k_tmp));
        #ifdef TRIGTABLES
        // building sine and cosine tables
        sine_table.push_back(std::sin(k_tmp));
        cosine_table.push_back(std::cos(k_tmp));
        #endif
    }

    // HDF5 business
    H5::H5File* file = nullptr;
    std::string filename("bb_2D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nq_"+std::to_string(N_q)+"_Nk_"+std::to_string(N_k)+"_single_ladder_sum_third.hdf5");
    const H5std_string FILE_NAME( filename );
    // The different processes cannot create more than once the file to be written in.
    if (world_rank==root_process){
        file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    }
    // Getting the data
    FileData dataFromFile;
    dataFromFile = get_data(inputFilename,Ntau);
    std::vector<double> wn = dataFromFile.iwn;
    std::vector< std::complex<double> > iwn(wn.size());
    std::transform(wn.begin(),wn.end(),iwn.begin(),[](double d){ return std::complex<double>(0.0,d); }); // casting into array of double for cubic spline.
    /************************* testing **********************************/
    dataFromFile = get_data(inputFilenameLoad+"_Nit_18.dat",2*Ntau);
    std::vector<double> re = dataFromFile.re;
    std::vector<double> im = dataFromFile.im;
    std::vector< std::complex<double> > sigma_iwn;
    for (size_t i=0; i<2*Ntau; i++){
        sigma_iwn.push_back(std::complex<double>(re[i],im[i]));
    }
    /************************* testing **********************************/
    // Bosonic Matsubara array
    std::vector< std::complex<double> > iqn; // for the total susceptibility
    std::vector< std::complex<double> > iqn_tilde; // for the inner loop inside Gamma.
    for (size_t j=0; j<3; j++){
        iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
    }
    const unsigned int DATA_SET_DIM = iqn.size();
    for (signed int j=(-static_cast<int>(Ntau/2))+1; j<(signed int)static_cast<int>(Ntau/2); j++){ // Bosonic frequencies.
        std::complex<double> matFreq(0.0 , (2.0*(double)j)*M_PI/beta );
        iqn_tilde.push_back( matFreq );
    }
    spline<double> splObj;
    std::vector<double> initVec(2*Ntau,0.0); // This data contains twice as much data to perform the interpolation
    // Spline object used to interpolate the self-energy
    IPT2::SplineInline< std::complex<double> > splInlineObj(Ntau,initVec,q_tilde_array,iwn);
    splInlineObj.loadFileSpline(inputFilenameLoad,IPT2::spline_type::linear);
    /************************* testing **********************************/
    std::cout << "spline stuff" << "\n\n";
    std::ofstream ofSE1("test_one_SE.dat",std::ios::out);
    std::ofstream ofSE2("test_two_SE.dat",std::ios::out);
    size_t iii=12;
    for (size_t j=0; j<iqn_tilde.size(); j++){
        // std::cout << splInlineObj.calculateSpline( (iwn[iii]-iqn_tilde[j]).imag() );
        ofSE1 << (iwn[iii]-iqn_tilde[j]).imag() << "  " << splInlineObj.calculateSpline( (iwn[iii]-iqn_tilde[j]).imag() ).imag() << "\n";
    }
    ofSE1.close();
    std::cout << "simple size_t subtraction" << "\n\n";
    std::cout << "size SE " << sigma_iwn.size() << std::endl;
    for (size_t j=0; j<iqn_tilde.size(); j++){
        ofSE2 << (iwn[iii]-iqn_tilde[j]).imag() << "  " << sigma_iwn[(Ntau-1)+iii-j].imag() << "\n";
    }
    ofSE2.close();
    exit(0);
    /************************* testing **********************************/
    // One-ladder calculations
    #ifdef TRIGTABLES
    TrigP<double,int> trigObj(sine_table,cosine_table);
    TrigP<double,int> trigObj_finer(sine_table_finer,cosine_table_finer);
    IPT2::OneLadder< std::complex<double> > one_ladder_obj(trigObj,trigObj_finer,splInlineObj,iqn,k_tilde_m_k_bar,iqn_tilde,mu,U,beta);
    #else
    IPT2::OneLadder< std::complex<double> > one_ladder_obj(splInlineObj,iqn,k_tilde_m_k_bar,sine_table,cosine_table,iqn_tilde,mu,U,beta);
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
    auto_ptr< std::vector< std::vector<MPIData> > > gathered_MPI_data(new std::vector< std::vector<MPIData> >());
    // std::vector< std::vector<MPIData> >* gathered_MPI_data = new std::vector< std::vector<MPIData> >();

    // Dispatching the tasks across the processes called in
    if (world_rank==root_process){
        IPT2::set_vector_processes(&vec_to_processes,N_k);
        for (auto el : vec_to_processes){
            std::cout << el.k_tilde_x << " " << el.k_tilde_y << std::endl;
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
        for (size_t i=0; i<=num_elem_per_proc; i++){
            auto mpidataObj = vec_to_processes[i];
            clock_t begin = clock();
            try{
                GG_iqn = one_ladder_obj(mpidataObj.k_tilde_x,mpidataObj.k_tilde_y);
            } catch (const std::invalid_argument& err){
                std::cerr << err.what()  << "\n";
                exit(1);
            }
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "one_ladder_obj number " << i << " of world_rank " << world_rank << " lasted " << elapsed_secs << " secs to be computed" << "\n";
            gathered_MPI_data->push_back(GG_iqn);
            #ifndef TRIGTABLES
            printf("(%li,%li) calculated by root process\n", mpidataObj.k_tilde_x, mpidataObj.k_tilde_y);
            #else
            printf("(%i,%i) calculated by root process\n", mpidataObj.k_tilde_x, mpidataObj.k_tilde_y);
            #endif
        }
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        // Fetch data from the slaves
        MPIDataReceive mpi_data_receive;
        int recv_root_num_elem;
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
        }

        std::ofstream test1("test_calculate_spline_Green_function_2D.dat", std::ios::out);
        for (auto el : gathered_MPI_data->back()){
            test1 << el.cplx_data_jj.real() << "\t\t" << el.cplx_data_jj.imag() << "\t\t" << el.cplx_data_szsz.real() << "\t\t" << el.cplx_data_szsz.imag() << "\n";
        }
        test1.close();

        std::vector< std::complex<double> > mpi_data_to_transfer_hdf5_jj(iqn.size());
        std::vector< std::complex<double> > mpi_data_to_transfer_hdf5_szsz(iqn.size());
        for (size_t l=0; l<gathered_MPI_data->size(); l++){
            std::vector<MPIData> mpi_data_hdf5_tmp = gathered_MPI_data->at(l);
            std::string DATASET_NAME("kbar_"+std::to_string(k_tilde_m_k_bar[mpi_data_hdf5_tmp[0].k_tilde_x])+"ktilde_"+std::to_string(k_tilde_m_k_bar[mpi_data_hdf5_tmp[0].k_tilde_y]));
            std::cout << "DATASET_NAME: " << DATASET_NAME << std::endl;
            // extracting the std::complex<double> data from the MPI struct
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
        for(size_t i = 0; i < num_elem_to_receive; i++) {
            auto mpidataObj = vec_for_slaves[i];
            clock_t begin = clock();
            try{
                GG_iqn = one_ladder_obj(mpidataObj.k_tilde_x,mpidataObj.k_tilde_y);
            } catch (const std::invalid_argument& err){ // error might be thrown from integral
                std::cerr << err.what() << "\n";
                exit(1);
            }
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "one_ladder_obj number " << i << " of world_rank " << world_rank << " lasted " << elapsed_secs << " secs to be computed" << "\n";
            gathered_MPI_data->push_back(GG_iqn);
            #ifndef TRIGTABLES
            printf("(%li, %li) calculated by slave_process %d\n", mpidataObj.k_tilde_x, mpidataObj.k_tilde_y, world_rank); //, (void*)&vec_for_slaves);
            #else
            printf("(%i, %i) calculated by slave_process %d\n", mpidataObj.k_tilde_x, mpidataObj.k_tilde_y, world_rank); //, (void*)&vec_for_slaves);
            #endif
        }
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        ierr = MPI_Send( &num_elem_to_receive, 1 , MPI_INT, root_process, RETURN_NUM_RECV_TO_ROOT, MPI_COMM_WORLD );
        for (int l=0; l<num_elem_to_receive; l++){
            ierr = MPI_Send( (void*)(gathered_MPI_data->at(l).data()), gathered_MPI_data->at(l).size(), MPI_Data_struct_t, root_process, l+SHIFT_TO_DIFFERENTIATE_TAGS, MPI_COMM_WORLD);
        }
    }

    
    // Should be a loop over the external momentum from this point on...
    arma::Cube< std::complex<double> > G_k_bar_q_tilde_iwn(q_tilde_array.size(),q_tilde_array.size(),iwn.size());
    arma::Cube<double> G_k_bar_q_tilde_tau(q_tilde_array.size(),q_tilde_array.size(),Ntau+1);
    std::vector<double> FFT_k_bar_q_tilde_tau;
    constexpr double k_bar = M_PI/2.0;

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            for (size_t m=0; m<q_tilde_array.size(); m++){
                G_k_bar_q_tilde_iwn(l,m,n_bar) = 1.0/( ( iwn[n_bar] ) + mu - epsilonk(q_tilde_array[l]+k_bar,q_tilde_array[m]+k_bar) - splInlineObj.calculateSpline(iwn[n_bar].imag()) );
            }
        }
    }

    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        for (size_t l=0; l<q_tilde_array.size(); l++){
            for (size_t m=0; m<q_tilde_array.size(); m++){
                // Substracting the tail of the Green's function
                G_k_bar_q_tilde_iwn(l,m,n_bar) -= 1.0/(iwn[n_bar]) + epsilonk(q_tilde_array[l]+k_bar,q_tilde_array[m]+k_bar)/iwn[n_bar]/iwn[n_bar]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
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

    /* TEST G(-tau) */
    std::ofstream test2("test_1_corr_no_cubic_spline.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test2 << beta_array[j] << "  " << -1.0*G_k_bar_q_tilde_tau(2,2,Ntau-j) << "\n";
    }
    test2.close();

    if (world_rank==root_process){
        delete file;
    }
    // delete gathered_MPI_data;
    MPI_Type_free(&MPI_Data_struct_t);

    ierr = MPI_Finalize();

    return 0;

}