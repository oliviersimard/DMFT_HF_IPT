#include<fstream>
#include<iostream>
#include<string>
#include "../src/tridiagonal.hpp"

using namespace std;

vector<double> exam1(1024);// array that can hold 100 numbers for 1st column 
vector<double> exam2(1024);// array that can hold 100 numbers for 2nd column 
vector<double> exam3(1024);// array that can hold 100 numbers for 3rd column  
vector<double> exam4(1024);
vector<double> exam5(1024);

int main(){ // int main NOT void main
  ifstream infile;   
  string firstline("");
  unsigned int num = 0; // num must start at 0
  const unsigned int N_tau=512;
  const double beta=10;
  arma::Cube< complex<double> > inputFunct(2,2,1024);
  infile.open("../data/Self_energy_1D_U_3.000000_beta_10.000000_n_0.500000_N_tau_512_Nit_3.dat");// file containing numbers in 3 columns 
  if(infile.fail()){ // checks to see if file opended 
    cout << "error" << endl; 
    return 1; // no point continuing if the file didn't open...
  } 

  for (signed int j=(-(signed int)N_tau); j<(signed int)N_tau; j++){
    std::complex<double> matFreq(0.0 , (2.0*(double)j+1.0)*M_PI/beta );
    iwnArr_l.push_back( matFreq );
  }
  vector<double> realIwn(iwnArr_l.size());
  transform(iwnArr_l.begin(),iwnArr_l.end(),realIwn.begin(),[](complex<double> cd){ return cd.imag(); }); // casting into array of double for cubic spline.
  spline< complex<double> > spl;
  while(!infile.eof()){ // reads file to end of *file*, not line
    if (num==0 && firstline==""){ 
      getline(infile,firstline);
      cout << firstline << endl;
      if (firstline[0]!='/'){
        cout << "Gotta remove first line" << endl;
        exit(1);
      }
    }else{
	    infile >> exam1[num]; // read first column number
	    infile >> exam2[num]; // read second column number
	    infile >> exam3[num]; // read third column number
      infile >> exam4[num];
      infile >> exam5[num];
      //}
      ++num; // go to the next number
      // you can also do it on the same line like this:
      // infile >> exam1[num] >> exam2[num] >> exam3[num]; ++num;
    }
  } 
  infile.close();
  for (size_t i=0; i<1024; i++){
    inputFunct.slice(i)(0,0)=complex<double>(exam4[i],exam5[i]);
    //cout << inputFunct.slice(i)(0,0) << endl;
  }
  spl.set_points(realIwn,inputFunct);
  cout << spl.operator()(-100.) << endl;
  return 0; // everything went right.
} 
