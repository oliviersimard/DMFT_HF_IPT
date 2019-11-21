#ifndef Json_Utils_H_
#define Json_Utils_H_

#include <json_spirit.h> // Must have c++ linkage
#include <sstream>
#include <fstream>
#include <string>


#define VERBOSE 0
#define MAX_DOUBLE_SIZE 10
#define MAX_INT_SIZE 5
#define MAX_BOOL_SIZE 4
#define MAX_CHR_SIZE 3

struct MembCarrier;
class Json_utils{
    public:
        const std::streampos getSize(const std::string&) const;
        const json_spirit::mValue& get_object_item(const json_spirit::mValue&, const std::string&) const;
        const json_spirit::mValue& get_array_item(const json_spirit::mValue&, size_t) const;
        const MembCarrier JSONLoading(const std::string& filename) const noexcept(false);
        Json_utils()=default;
        Json_utils(const Json_utils&)=delete;
        Json_utils& operator=(const Json_utils&)=delete;
};

inline const json_spirit::mValue& Json_utils::get_object_item(const json_spirit::mValue& element, const std::string& name) const{

    return element.get_obj().at(name);
}

inline const json_spirit::mValue& Json_utils::get_array_item(const json_spirit::mValue& element, size_t index) const{

    return element.get_array().at(index);
}


struct MembCarrier{
    double db_arr[MAX_DOUBLE_SIZE];
    double db_arr2[2]; // db_ptr2 is meant to contain the k-space vector for 2D case!
    int int_arr[MAX_INT_SIZE];
    bool boo_arr[MAX_BOOL_SIZE];
    char* char_arr[MAX_CHR_SIZE];
    explicit MembCarrier(double[], double[], int[], bool[], char*[]);
    MembCarrier& operator=(const MembCarrier&)=delete;
};

#endif /* json_utils_H_ */