from PyGOFMM import *
import unittest
import os
import numpy as np
from math import exp
from mpi4py import MPI

class PyDataTesting(unittest.TestCase):
    # test get/set
    def test_getset(self):
        pd = PyData(7,8)

        # set value 
        pd.setvalue(2,4,2.5) # choosing numbers representable in base 2
        pd.setvalue(4,2,4.25)

        # test get value
        self.assertEqual(pd.getvalue(2,4), 2.5)
        self.assertEqual(pd.getvalue(4,2), 4.25)

    # test initialization
    def test_initialization(self):
        pd = PyData(10,15)
        self.assertEqual(pd.row(),10)
        self.assertEqual(pd.col(),15)

    # test null init
    def test_null(self):
        pd = PyData()
        self.assertEqual(pd.row(),0)
        self.assertEqual(pd.col(),0)


    # test randomizer
    def test_rand(self):
        # initialize
        m = 100
        pd = PyData(m,2)

        # call rand
        pd.rand(0.5,1.5)


        # sum a column and check falls within bounds
        col_sum = 0.0
        for i in range(m):
            col_sum += pd.getvalue(i,0)

        # divide by m
        col_sum = col_sum/m;

        # should be within 0.25,0.75
        self.assertTrue(col_sum > 0.9 and col_sum < 1.1,msg='Randomize failed, sum is {}'.format(col_sum) )


    # test save/load
    def test_saveload(self):
        m = 5
        # initialize
        pd = PyData(m,m)

        # set value 
        pd.setvalue(2,4,2.5) # choosing numbers representable in base 2
        pd.setvalue(4,2,4.25)

        # write to file
        filenm = './tmp_test_data.bin'
        pd.write(filenm)

        # load
        pd2 = PyData(m,m)
        pd2.read(m,m,filenm)
        os.remove(filenm)


        # assert equal at every position
        for i in range(m):
            for j in range(m):
                self.assertEqual( pd.getvalue(i,j), pd2.getvalue(i,j) )


    # test submatrix functionality of data class
    def test_submatrix(self):
        # large matrix
        m = 10
        pd = PyData(m,m)
        pd.rand(0.5,1.5)

        # sub matrix indices
        s = 5
        I = np.arange(5,dtype=np.intp) + 3

        # get actual submatrix
        sub_pd = pd.submatrix(I,I)

        # loop over indices and check
        for i in range(s):
            for j in range(s):
                sub_val = sub_pd.getvalue(i,j)
                tru_val = pd.getvalue(I[i], I[j])
                self.assertEqual(sub_val,tru_val)
        



class PyConfigTesting(unittest.TestCase):
    def test_initialization(self):
        metric = "USER_DISTANCE"
        #metric = 0;
        problem_size = 10000
        leaf_node_size = 64
        neighbor_size = 32
        maximum_rank = 20
        tolerance = 1e-5
        budget = 2e-1
        secure_accuracy = True
        tc = PyConfig(metric, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)
        self.assertEqual(metric, tc.getMetricType())
        
        self.assertEqual(problem_size, tc.getProblemSize())
        self.assertEqual(leaf_node_size, tc.getLeafNodeSize())

class MatrixRoutineTesting(unittest.TestCase):
    def test_SPDMatrix(self):
        m = 2000;
        n = 2000;
        precision = "FLOAT"; #not used
        testM = PySPDMatrix(m, n);
        self.assertEqual(m, testM.row())
        self.assertEqual(n, testM.col())
        self.assertEqual(m*n, testM.size())
        self.assertEqual(0.0, testM.getvalue(10, 10))
        self.assertEqual(0.0, testM.getvalue(m-1, n-2))
        
        testM.randspd(0.0, 1.0)
        self.assertEqual(m, testM.row())
        self.assertEqual(n, testM.col())
        
        #check symmetry
        self.assertEqual(testM.getvalue(40, 50), testM.getvalue(50, 40))

        #check diagonally dominant
        #row_sum = 0.0
        #for i in range(m):
        #   for j in range(n):
        #       if(i!=j):
        #           row_sum += testM.getvalue(i, j)
        #   self.assertTrue(testM.getvalue(i, i) >= row_sum, msg="Diagonal Dominance Failed. [A(i, i) RowSum(i)] = {}".format([testM.getvalue(i, i), row_sum]))

    def test_KernelMatrix(self):
        m = 200
        n = 150
        d = 20
        trn_idx = 10
        tst_idx = 15


        # make data
        Xtr = PyData(d,m)
        Xtr.rand(0.0,1.0)
        Xte = PyData(d,n)
        Xte.rand(0.0,1.0)

        # make kernel object
        ks = PyKernel("GAUSSIAN")
        ks.SetBandwidth(1.0) # results in scaling of -1/2


        # Init symmetric kernel
        KK = PyKernelMatrix(Xtr,ks)
        Kt = PyKernelMatrix(Xtr,ks,Xte)
        self.assertEqual(1.0, KK.getvalue(2,2))
        self.assertEqual(1.0, KK.getvalue(10,10))

        # Test kernels
        diff_sym = 0.0
        diff_asym = 0.0
        for di in range(d):
            xtr = Xtr.getvalue(di,trn_idx)
            xtr1 = Xtr.getvalue(di,tst_idx)
            xte = Xte.getvalue(di,tst_idx)
            
            # differences
            diff_sym += (xtr - xtr1) * (xtr - xtr1)
            diff_asym += (xtr - xte) * (xtr - xte)

        pyexp_asym = exp(-0.5 * diff_asym)
        pyexp_sym = exp(-0.5 * diff_sym)
        self.assertAlmostEqual(pyexp_asym,KK.getvalue(trn_idx,tst_idx))
        self.assertAlmostEqual(pyexp_sym,Kt.getvalue(trn_idx,tst_idx))
    
    def test_compress(self):
        m = 1000
        n = 1000
        prt = PyRuntime()
        prt.init()
        print("Initialized\n")
        #testM = PySPDMatrix(m, n)
        #testM.randspd(0.0, 1.0)
        stol = 0.001
        budget = 0.1
        #Ctree = compress(testM, stol, budget)
        prt.finalize()  
        

if __name__=='__main__':
    unittest.main()

