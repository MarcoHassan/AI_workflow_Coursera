#!/usr/bin/env python
"""
model tests
"""


import unittest
## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train()
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## load the model
        model = model_load()
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def test_03_predict(self):
        """
        test the predict functionality
        """

        ## load model first
        model = model_load()
        
        ## example predict
        for query in [np.array([[3.89, 5.78,
                                7.42086181, 9.42086181,
                                2.1904, 6.1966,
                                1.7743]]),
                      np.array([[3.89, 5.78,
                                7.42086181, 9.42086181,
                                2.1904, 6.1966,
                                1.7743]]),
                      np.array([[3.89, 4.78,
                                3.42086181, 12.42086181,
                                2.1904, 3.1966,
                                2.123123]])]:
            result = model_predict(query,model)

            y_pred = result[0]
            self.assertTrue(y_pred > 0)

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
