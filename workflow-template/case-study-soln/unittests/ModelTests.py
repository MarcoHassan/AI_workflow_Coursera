#!/usr/bin/env python
"""
model tests
"""

import warnings
warnings.filterwarnings("ignore")

import sys

paths = ['/Users/marcohassan/Desktop/Learning/AI_workflow_Coursera/workflow-template/case-study-soln/']
paths.extend(sys.path)
sys.path = paths

import unittest

## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    # def test_01_train(self):
    #     """
    #     test the train functionality
    #     """

    #     ## train the model
    #     model_train(test=True)
    #     self.assertTrue(os.path.exists(SAVED_MODEL))

    # def test_02_load(self):
    #     """
    #     test the train functionality
    #     """
                        
    #     ## train the model
    #     model = model_load()
        
    #     self.assertTrue('predict' in dir(model))
    #     self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load()
    
        ## ensure that a list can be passed
        query_data = np.array([[6.1,2.8]])
        query_data = query_data.tolist()

        result = model_predict(query_data,model,test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] in [0,1])

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
