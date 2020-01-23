#include "gmock/gmock.h" // To enable array comparisons, strings, etc..
#include "gtest/gtest.h" // To enable basic check-ups.
#include "src/test_utils.hpp"

#define EXPECT_FLOATS_NEARLY_EQ(expected, actual, thresh) \
        EXPECT_EQ(expected.size(), actual.size()) << "Array sizes differ.";\
        for (size_t idx = 0; idx < std::min(expected.size(), actual.size()); ++idx) \
        { \
            EXPECT_NEAR(expected[idx], actual[idx], thresh) << "at index: " << idx;\
        }

#define EXPECT_CMPLX_NEARLY_EQ(expected, actual, thresh) \
        for (size_t idx = 0; idx < std::min(expected.size(), actual.size()); ++idx) \
        { \
            EXPECT_NEAR(expected[idx+static_cast<size_t>(expected.size()/2)].real(), actual[idx].real(), thresh) << "RE at index: " << idx;\
            EXPECT_NEAR(expected[idx+static_cast<size_t>(expected.size()/2)].imag(), actual[idx].imag(), thresh) << "IM at index: " << idx;\
        }

#define EXPECT_COMPLEX_DOUBLE_EQ(a, b, thresh) \
  EXPECT_NEAR(a.real(), b.real(), thresh); \
  EXPECT_NEAR(a.imag(), b.imag(), thresh); 

namespace {

class TestHello : public ::testing::Test {
};

TEST_F(TestHello, Message) {
    EXPECT_EQ("Hello World!", hello_world()) << "Your function should return: Hello, World!";
}

TEST_F(TestHello, MessageStdOut) {
    std::stringstream buffer;
    cout_redirect redirectObj(buffer.rdbuf());
    std::cout << "Hello, World!";
    std::string stdoutval = buffer.str();
    redirectObj.~cout_redirect(); // redirect to standard output buffer.
    EXPECT_EQ("Hello, World!", stdoutval) << "Your function should return: Hello, World!";
}

} /* end of namespace */

namespace { // To ensure that the variables are local to the TU.
  struct TestFFT : Data, ::testing::Test{
    public:
      FFTtools FFTObj;
      static const double beta;
      static const double U;
      static const unsigned int N_k;
      static const unsigned int Ntau;
      static const double hyb_c;
      static std::vector< std::complex<double> > functFilling(){
        std::vector< std::complex<double> > retArr;
        for (signed int j=(-(signed int)Ntau); j<(signed int)Ntau; j++){
          std::complex<double> matFreq(0.0 , (2.0*(double)j+1.0)*M_PI/beta );
          retArr.push_back( 1./matFreq );
        }
        return retArr;
      };
      static std::vector<double> functFillingTau(){
        std::vector<double> retArr;
        for (unsigned int j=0; j<=10; j++){
          retArr.push_back(-0.5);
        }
        return retArr;
      }
      static std::vector< std::complex<double> > iomeganArr;
      static std::vector<double> tauArr, actualArrUp;//, actualArrDown;
      //static std::vector< std::complex<double> > actualArrUpC, actualArrDownC;
      static arma::Cube< std::complex<double> > GreenTestwn;
      static arma::Cube<double> GreenTesttau_pos, GreenTesttau_neg;
      static GreenStuff GreenTest;
  };
  // Static members' initializations.
  const double TestFFT::beta = 100;
  const double TestFFT::U = 0.5;
  const unsigned int TestFFT::N_k = 200;
  const unsigned int TestFFT::Ntau = 256; // Needs to be a power of 2
  const double TestFFT::hyb_c=4;
  std::vector< std::complex<double> > TestFFT::iomeganArr = TestFFT::functFilling();
  std::vector<double> TestFFT::tauArr = TestFFT::functFillingTau();
  std::vector<double> TestFFT::actualArrUp(2*TestFFT::Ntau+1,0.0);
  arma::Cube< std::complex<double> > TestFFT::GreenTestwn(2,2,2*TestFFT::Ntau,arma::fill::zeros);
  arma::Cube<double> TestFFT::GreenTesttau_pos(2,2,2*TestFFT::Ntau+1,arma::fill::zeros), TestFFT::GreenTesttau_neg(2,2,2*TestFFT::Ntau+1,arma::fill::zeros);
  GreenStuff TestFFT::GreenTest(TestFFT::Ntau,TestFFT::N_k,TestFFT::beta,TestFFT::U,TestFFT::hyb_c,TestFFT::iomeganArr,TestFFT::GreenTesttau_pos,TestFFT::GreenTesttau_neg,TestFFT::GreenTestwn);

  TEST_F(TestFFT, iwnTotau){
    const unsigned int timeGrid = 2*Ntau;
    const std::complex<double> im(0.0,1.0);
    std::complex<double>* inUp=new std::complex<double> [timeGrid];
    std::complex<double>* outUp=new std::complex<double> [timeGrid];
    fftw_plan pUp; //fftw_plan pDown;
    for(size_t k=0;k<timeGrid;k++){
        inUp[k]=iomeganArr[k];
    }
    pUp=fftw_plan_dft_1d(timeGrid, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(pUp); //fftw_execute(pDown);

    for(size_t i=0; i<timeGrid; i++){
      actualArrUp[i] = ( (1./beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * outUp[i] ).real();
    }
    actualArrUp[timeGrid]=-1.0-actualArrUp[0]; 

    // for (unsigned int i=0; i<=10; i++){ // Converting into vector objects to compare!
    //   actualArrUp[i]=GreenTest.matsubara_t_pos(arma::span(0,0),arma::span(0,0),arma::span(300,310))[i]; // Includes 310.
    // }
    std::vector<double> actualArrUpCompare(actualArrUp.begin()+static_cast<size_t>(Ntau/2), actualArrUp.begin()+static_cast<size_t>(Ntau/2)+11); // Excludes 11.
    // for (auto el : actualArrUp){
    //   std::cout << el << std::endl;
    // }
    delete [] inUp;
    delete [] outUp;
    fftw_destroy_plan(pUp);

    EXPECT_FLOATS_NEARLY_EQ(tauArr, actualArrUpCompare, 0.001);
  }

  TEST_F(TestFFT, staticnessMemberVariables){ // GreenTest was instantiated first, therefore the static members of GreenStuff should be the  
  //same input in the first call to constructor
    const unsigned int Ntau2Compare = Ntau-2;
    arma::Cube< std::complex<double> > GreenTestwn2Compare(2,2,2*Ntau2Compare,arma::fill::zeros);
    arma::Cube<double> GreenTesttau2Compare_pos(2,2,2*Ntau2Compare+1,arma::fill::zeros), GreenTesttau2Compare_neg(2,2,2*Ntau2Compare+1,arma::fill::zeros);
    GreenStuff GreenTest2Compare(Ntau2Compare,TestFFT::N_k-10,TestFFT::beta/2.0,TestFFT::U/3.0,TestFFT::hyb_c,TestFFT::iomeganArr,GreenTesttau2Compare_pos,GreenTesttau2Compare_neg,GreenTestwn2Compare);
    EXPECT_EQ(GreenStuff::N_tau,Ntau) << "N_tau failed to be static within GreenStuff upon first call to constructor.";
    EXPECT_EQ(GreenStuff::N_k,N_k) << "N_k failed to be static within GreenStuff upon first call to constructor.";
    EXPECT_DOUBLE_EQ(GreenStuff::beta,beta) << "beta failed to be static within GreenStuff upon first call to constructor.";
    EXPECT_DOUBLE_EQ(GreenStuff::U,U) << "U failed to be static within GreenStuff upon first call to constructor.";
  }

} /* end of namespace */

namespace{
  struct TestUtilities : ::testing::Test {
    Integrals integralsObj;
    static const std::complex<double> im;
  };
  const std::complex<double> TestUtilities::im(0.0,1.0);
  
  TEST_F(TestUtilities, FalsePostionMethod){
    std::function<double(double)> testFunct = [](double x){return x*x*x-x*x+2.0;};
    double rootResult = integralsObj.falsePosMethod(testFunct,-20.0,20.0);
    EXPECT_NEAR(-1.0,rootResult,0.0001);
  }

  TEST_F(TestUtilities, twoDIntegral){
    // double TotalTime=0.0;
    // clock_t beginLambda = clock();
    std::function<double(double,double)> easyFunct = [](double x, double y){ return x*y*y; };
    double resultI2D = integralsObj.I2D(easyFunct,0.,1.,0.,1.);
    // clock_t endLambda = clock();
    // TotalTime+=(double)(endLambda - beginLambda)/CLOCKS_PER_SEC;
    EXPECT_NEAR(0.166667,resultI2D,0.001);
  }

  TEST_F(TestUtilities, oneDIntegral){ /* this integral method is recursive */
    std::function<double(double)> testFunct = [](double x){ return x*x; };
    double resultI1D = integralsObj.integrate_simps(testFunct,0.0,2.0,1e-5);
    EXPECT_NEAR(resultI1D,2.66666,1e-5);
  }

  TEST_F(TestUtilities, LUDecomposition){
    LUtools< std::complex<double> > LUObj;
    arma::Mat< std::complex<double> > matArmaD = { { 2.+1.0*im, -1.-0.3*im, 0.0+0.0*im },
                                                   { -4.-2.1*im, 6.-0.1*im, 3.+0.0*im },
                                                   { 0.0+0.0*im, -2.-0.3*im, 8.+0.5*im } };

    std::vector< std::complex<double> > subdiagonalArma, diagonalArma, superdiagonalArma;

    std::vector< std::complex<double> > BArma{ 2.+0.1*im , 0.5-2.3*im , 7.+0.2*im }, xArma(matArmaD.n_rows,0.0);

    // Building main parts of the routine.
    for (unsigned int k=0; k<matArmaD.n_cols-1; k++){
      subdiagonalArma.push_back(matArmaD(k+1,k));
      superdiagonalArma.push_back(matArmaD(k,k+1));
    }
    for (unsigned int k=0; k<matArmaD.n_cols; k++){
      diagonalArma.push_back(matArmaD(k,k));
    }

    try{
      LUObj.tridiagonal_LU_decomposition(subdiagonalArma,diagonalArma,superdiagonalArma);
      LUObj.tridiagonal_LU_solve(subdiagonalArma,diagonalArma,superdiagonalArma,BArma,xArma);
    } 
    catch (const std::exception& err){
      std::cout << err.what() << std::endl;
    }

    EXPECT_COMPLEX_DOUBLE_EQ(std::complex<double>(1.00096,-0.553344), xArma[0], 0.0001);
    EXPECT_COMPLEX_DOUBLE_EQ(std::complex<double>(0.452799,-0.341565), xArma[1], 0.0001);
    EXPECT_COMPLEX_DOUBLE_EQ(std::complex<double>(0.994411,-0.105562), xArma[2], 0.0001);
  }

  TEST_F(TestUtilities, CubicSpline){
    spline<double> spl;
    std::vector<double> X(5);
    arma::Cube<double> Y(2,2,5);
    X[0]=0.1; X[1]=0.4; X[2]=1.2; X[3]=1.8; X[4]=2.0;
    Y.slice(0)(0,0)=0.1; Y.slice(1)(0,0)=0.7; Y.slice(2)(0,0)=0.6; Y.slice(3)(0,0)=1.1; Y.slice(4)(0,0)=0.9;

    spl.set_points(X,Y);    // currently it is required that X is already sorted

    double x=1.5;
    double result=spl(x);
    EXPECT_NEAR(result,0.915345,1e-6);
  }

  TEST_F(TestUtilities, GlobSearchFiles){
    std::string strTest("*duGrandNimporteQuoi*");
    std::stringstream buffer;
    cerr_redirect redirectObj(buffer.rdbuf());
    try{
      glob(strTest);
    } catch(const std::exception& err){
      std::cerr << err.what() << std::endl;
    }
    std::string stdoutval = buffer.str();
    redirectObj.~cerr_redirect(); // redirect to standard output buffer.
    EXPECT_EQ("-3\n", stdoutval) << "Glob should return -3 to std output. Check if Verbose > 0.";
  }

  TEST_F(TestUtilities, LoadingSplineInline){
    const unsigned int N_tau = 128;
    const double beta = 50.0;
    // std::vector<double> generated_random_numbers; // simulates random iwn array to plot the spline data
    // Saving the interpolated data from the computed self-energy
    std::ofstream oSelf("../test/data/self_energy_as_interpolated_from_linear_spline.dat", std::ofstream::out | std::ofstream::app);

    for (signed int j=(-(signed int)N_tau); j<(signed int)N_tau; j++){ // Fermionic frequencies.
      std::complex<double> matFreq(0.0 , (2.0*(double)j+1.0)*M_PI/beta );
      iwnArr_l.push_back( matFreq );
    }
    for (size_t j=0; j<N_tau; j++){ // Bosonic frequencies.
      std::complex<double> matFreq(0.0 , (2.0*(double)j)*M_PI/beta );
      iqnArr_l.push_back( matFreq );
    }
    // Loading the test file in ./data
    std::string fileToLoad("../test/data/Self_energy_1D_U_11.000000_beta_50.000000_n_0.500000_N_tau_128");
    std::vector<double> initVec(2*N_tau,0.0); // Important that it is 2*MULT_N_TAU, but changed it to be used in this test.
    IPT2::SplineInline< std::complex<double> > splInlineObj(N_tau,initVec,vecK,iwnArr_l,iqnArr_l);
    splInlineObj.loadFileSpline(fileToLoad,IPT2::spline_type::linear);
    std::vector< std::complex<double> > result_before_spline = splInlineObj.get_loaded_data_interpolated();
    std::vector< std::complex<double> > result_after_spline;
    // Generating the random iwn numbers properly bracketed (positive Matsubara frequencies)
    // const double max_val = (2.0*N_tau-1.0)*M_PI/beta, min_val = M_PI/beta;
    // generated_random_numbers = generate_random_numbers(10*N_tau,min_val,max_val);

    std::complex<double> self_energy_spline;
    for (size_t i=static_cast<size_t>(iwnArr_l.size()/2); i<iwnArr_l.size(); i++){
      self_energy_spline = splInlineObj.calculateSpline(iwnArr_l[i].imag());
      result_after_spline.push_back(self_energy_spline);
      if (i==0)
        oSelf << "/" << "\n";
      oSelf << iwnArr_l[i].imag() << "\t\t" << self_energy_spline.real() << "\t\t" << self_energy_spline.imag() << "\n";
    }
    oSelf.close();
    EXPECT_CMPLX_NEARLY_EQ(result_before_spline,result_after_spline,1e-3);
  }

  // TEST_F(TestGreenStuff, copyAssignment){
  //   // defining new instance of GreenStuff.
  //   unsigned int Ntau2Compare = Ntau-2;
  //   arma::Cube< std::complex<double> > GreenTestwn2Compare(2,2,Ntau2Compare,arma::fill::zeros);
  //   arma::Cube<double> GreenTesttau2Compare(2,2,Ntau2Compare+1,arma::fill::zeros);
  //   GreenStuff GreenTest2Compare(Ntau2Compare,N_k-10,beta/2.0,U/3.0,GreenTesttau2Compare,GreenTestwn2Compare);

  //   std::stringstream buffer;
  //   cerr_redirect redirectObj(buffer.rdbuf());
  //   GreenTest = GreenTest;
  //   std::string stdoutval = buffer.str();
  //   redirectObj.~cerr_redirect();
  //   buffer.str(std::string()); // clearing stringstream.
  //   std::cout << buffer.str();
  //   EXPECT_EQ("Call to identical copy assignment function.\n", stdoutval) << "Copy assignment should return to std output.";
  //   cerr_redirect redirectObj2(buffer.rdbuf());
  //   GreenTest2Compare = GreenTest;
  //   std::string stdoutval2 = buffer.str();
  //   redirectObj2.~cerr_redirect(); // redirect to standard output buffer.
  //   EXPECT_EQ("assignment operator used.\nCall to copy assignment function.\n", stdoutval2) << "Copy assignment should return to std output.";
  // }

} /* end of namespace */


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
