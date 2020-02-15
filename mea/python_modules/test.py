import numpy as np
import matplotlib.pyplot as plt

def tridiagonal_LU_decomposition(subdiagonal,diagonal,superdiagonal):
    assert ( len(subdiagonal)==len(superdiagonal) ) and ( len(subdiagonal)==(len(diagonal)-1) ), "Error in sizes of the tridiagonal matrices."

    size = len(subdiagonal)

    for i in range(size):
        if diagonal[i]==0.0:
            raise Exception("Cannot have zeros along the diagonal...")
        else:
            subdiagonal[i] /=  diagonal[i]
            diagonal[i+1] -= subdiagonal[i]*superdiagonal[i]
        if diagonal[size]==0.0:
            raise Exception("Cannot have zeros along the diagonal...see last element.")
    
    return subdiagonal, diagonal, superdiagonal

def tridiagonal_LU_solve(subdiagonal,diagonal,superdiagonal,rhs):
    size = len(diagonal)
    mb = np.zeros((size,),dtype=float)
    mb[0] = rhs[0]

    for i in range(1,size):
        mb[i] = rhs[i] - subdiagonal[i-1]*mb[i-1]

    mb[size-1] /= diagonal[size-1]
    for i in range(size-2,-1,-1):
        mb[i] = superdiagonal[i]*mb[i+1]
        mb[i] /= diagonal[i]

    # deleting the no-longer necessary stuff
    del subdiagonal
    del diagonal
    del superdiagonal
    del rhs

    return mb


if __name__=="__main__":

    test_mat = np.array([
        [1.0,2.0,0.0,0.0],
        [8.0,3.0,5.0,0.0],
        [0.0,7.0,6.0,9.0],
        [0.0,0.0,1.0,4.0]
    ],dtype=float)
    
    rhs = np.array([4.0,3.0,2.0,0.0],dtype=float)

    subdiagonal=np.zeros((len(rhs)-1,),dtype=float)
    diagonal=np.zeros((len(rhs),),dtype=float)
    superdiagonal=np.zeros((len(rhs)-1,),dtype=float)

    for i in range(1,len(rhs)):
        subdiagonal[i-1] = test_mat[i,i-1]
        diagonal[i-1] = test_mat[i-1,i-1]
        superdiagonal[i-1] = test_mat[i-1,i]
    
    diagonal[len(rhs)-1] = test_mat[len(rhs)-1,len(rhs)-1]

    subdiagonal, diagonal, superdiagonal = tridiagonal_LU_decomposition(subdiagonal,diagonal,superdiagonal)

    mb = tridiagonal_LU_solve(subdiagonal,diagonal,superdiagonal,rhs)

    print(mb)
