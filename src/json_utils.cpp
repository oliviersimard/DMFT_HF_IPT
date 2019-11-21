#include "json_utils.hpp"
//#include <typeinfo> # Useful to evaluate type that is considered

const std::streampos Json_utils::getSize(const std::string& filename) const{

    std::streampos begin, end;
    std::ifstream myfile(filename, std::ifstream::in);
    begin = myfile.tellg();
    myfile.seekg(0, std::ifstream::end);
    end = myfile.tellg();
    myfile.close();

    return end-begin;
}

MembCarrier::MembCarrier(double dp[], double dp2[], int ip[], bool bp[], char* chp[]){
    size_t i;
    for (i=0; i<MAX_DOUBLE_SIZE; i++) this->db_arr[i]=*(dp+i);
    for (i=0; i<2; i++) this->db_arr2[i]=*(dp2+i); 
    for (i=0; i<MAX_INT_SIZE; i++) this->int_arr[i]=*(ip+i);
    for (i=0; i<MAX_BOOL_SIZE; i++) this->boo_arr[i]=*(bp+i);
    for (i=0; i<MAX_CHR_SIZE; i++) this->char_arr[i]=*(chp+i);
}

const MembCarrier Json_utils::JSONLoading(const std::string& filename) const noexcept(false){
    std::ifstream JSONText(filename, std::ifstream::in);
    std::stringstream buffer;
    buffer << JSONText.rdbuf();
    json_spirit::mValue value;
    json_spirit::read(buffer,value); // Setting mValue object from buffer content.
    const std::string fileText(buffer.str()); // needs c_str() if printed using stdout.
    JSONText.close();
    if (VERBOSE > 0) std::cout << fileText.c_str() << std::endl;

    // Reading from json file
    //double precision
    const auto& n_t_spin_val = get_object_item(value, "n_t_spin");
    const auto& Umax_val = get_object_item(value, "Umax");
    const auto& Ustep_val = get_object_item(value, "Ustep");
    const auto& Umin_val = get_object_item(value, "Umin");
    const auto& betamax_val = get_object_item(value, "betamax");
    const auto& betastep_val = get_object_item(value, "betastep");
    const auto& betamin_val = get_object_item(value, "betamin");
    const auto& q_1D_val = get_object_item(value, "q_1D");
    //integers
    const auto& Ntau_val = get_object_item(value, "Ntau");
    const auto& N_it_val = get_object_item(value, "N_it");
    const auto& gridK_val = get_object_item(value, "gridK");
    //bool
    const auto& full_dG_dphi = get_object_item(value, "dGdPhi");
    const auto& load_self_val = get_object_item(value, "load_self");
    const auto& is_jj_val = get_object_item(value, "is_jj");
    //char
    const auto& solver_type_val = get_object_item(value, "solver_type");
    //array
    const auto& array_val = get_object_item(value, "q_2D");
    double container[2]; // Way to extract array inputs in json file!
    for (size_t i=0; i < array_val.get_array().size(); i++){
        container[i] = array_val.get_array().at(i).get_real();
    }

    if (VERBOSE > 0){
        std::cout << "Size of input file: " << getSize(filename) << "\n";
    }
    
    //Collecting the json variables to instantiate param object
    double Umax = Umax_val.get_real(); double Ustep = Ustep_val.get_real();
    double Umin = Umin_val.get_real(); double n_t_spin = n_t_spin_val.get_real();
    double betamax = betamax_val.get_real(); double betastep = betastep_val.get_real();
    double betamin = betamin_val.get_real(); double q_1D = q_1D_val.get_real();
    int N_it = N_it_val.get_int(); int Ntau = Ntau_val.get_int(); int gridK = gridK_val.get_int();
    bool dGdPhi = full_dG_dphi.get_bool(); bool load_self = load_self_val.get_bool(); bool is_jj = is_jj_val.get_bool();
    char* solver_type = const_cast<char*>(const_cast<std::string*>(&solver_type_val.get_str())->c_str());

    //std::cout << dGdPhi << " has type: " << typeid(dGdPhi).name() << std::endl;

    //Creating the arrays
    double dub[MAX_DOUBLE_SIZE] = { n_t_spin, Umax, Ustep, Umin, betamax, betastep, betamin, q_1D };
    // for (size_t i=0; i<=sizeof(dub)/sizeof(double); i++){
    //     std::cout << dub[i] << std::endl;
    // }
    int integ[MAX_INT_SIZE] = { Ntau, N_it, gridK };
    // for (size_t i=0; i<=sizeof(integ)/sizeof(int); i++){
    //     std::cout << integ[i] << std::endl;
    // }
    bool boo[MAX_BOOL_SIZE] = { dGdPhi, load_self, is_jj };

    char* chr[MAX_CHR_SIZE] = { solver_type };

    MembCarrier membCarrObj(dub,container,integ,boo,chr);

    return membCarrObj;
}