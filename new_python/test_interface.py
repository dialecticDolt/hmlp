import PyData
import unittest

class PyDataTesting(unittest.TestCase):
    # test get/set

    # test initialization
    def test_initialization(self):
        pd = PyData.PyData(10,15);
        self.assertEqual(pd.row(),10);
        self.assertEqual(pd.col(),15);

    # test null init
    def test_null(self):
        pd = PyData.PyData();
        self.assertEqual(pd.row(),0);
        self.assertEqual(pd.col(),0);


    # test randomizer


    # test save/load

if __name__=='__main__':
    unittest.main()

