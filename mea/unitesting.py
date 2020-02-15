import unittest
#import numpy.testing as npt
import numpy as np
import LU_Doolittle_alg as spl

class TestSplineUtilsMethods(unittest.TestCase):

    def test_m_b(self):
        #test_splineObj = spl.SplineUtils()
        resp = np.array([3.24867725,-0.44973545,1.00529101,-0.01058201],dtype=float)
        sup = [5,9,7]
        dia = [1,4,-5,1]
        sub = [-1,2,2]
        rhs_test = [1,4,-6,2]
        spl.SplineUtils.tdma(sup,dia,sub,rhs_test)
        bool_result = (np.round(resp,6)==np.round(spl.SplineUtils.m_b,6)).all()
        self.assertTrue(bool_result)

    def test_Fourier_transform_normalisation(self):
        test_arr = [1.2,1.4,1.6,1.0,1.5]
        Fourier_arr = np.fft.fft(test_arr)
        inv_Fourier_arr = np.fft.ifft(Fourier_arr)
        bool_result = (np.round(inv_Fourier_arr,8)==np.round(test_arr,8)).all()
        self.assertTrue(bool_result)

    def test_Fourier_transformation_custom(self):
        #x = np.random.rand(128)+1.j*np.random.rand(128)
        x = np.arange(0,128)
        bool_result = np.allclose(spl.FFT_Cooley_Tukey.FFT(x), np.fft.fft(x),rtol=1e-5)
        self.assertTrue(bool_result)

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()