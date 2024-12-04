import numpy as np
from utils import *

logger = get_logger('gen_data')

class TrainData:    
    def __init__(self, domain: list = [], size: int = 100, ftrain: float = 0.7, sigmaN: float = 1, seed: int = None) -> None:
        """
        Class to generate data for training and testing linear regression models.\n

        Attributes:
         domain (list) : Domain of the function.
         size (int)    : Size of the dataset.
         ftrain (float): Fraction of the dataset to use for training.
         sigmaN (float): Standard deviation of the Gaussian noise.
         seed (int)    : Seed for the random number generator.
        """
        self.domain = domain
        self.size   = size
        self.ftrain = ftrain
        self.sigmaN = sigmaN
        self.seed   = seed


    def create_ground_curve(self,ground_func,numpoints: int = 100,**params: dict) -> None:
        """
        Create the pair (X,h) for the ground curve.\n

        Args:
            ground_func (function): Function to generate the ground curve (must be a callable object).
            numpoints (int)       : Number of points to generate.
            **params (dict)       : Parameters for the ground curve.

        Returns:
            None.
        """
        X  = np.linspace(self.domain[0],self.domain[1],numpoints)
        h  = ground_func(X,**params)
        self.X_ref = X
        self.h_ref = h


    def create_dataset(self,ground_func,**params: dict) -> None:
        """
        Create the dataset (X,t) with Gaussian noise.\n

        Args:
            ground_func (function): Function to generate the ground curve (must be a callable object).
            **params (dict)       : Parameters for the ground curve.

        Returns:
            None.
        """
        logger.info('Creating dataset...')
        X       = sample_domain(lims=self.domain,N=self.size,seed=self.seed)
        h       = ground_func(X,**params)
        NX      = add_gaussian_noise(size=self.size,sigma=self.sigmaN,seed=self.seed)
        self.X  = X        
        self.t  = h + NX    


    def split_data(self) -> tuple:
        """
        Split the dataset into training and testing sets.\n

        Args:
            None.
        
        Returns:
            tuple: (X_train,X_test,t_train,t_test) for the training and testing sets.
        """
        ftrain = int(self.ftrain*self.size)
        
        self.X_train = self.X[:ftrain]
        self.t_train = self.t[:ftrain]
        #sort the data in the increasing order of X
        self.X_train,self.t_train = zip(*sorted(zip(self.X_train,self.t_train)))

        self.X_test  = self.X[ftrain:]
        self.t_test  = self.t[ftrain:]
        self.X_test,self.t_test = zip(*sorted(zip(self.X_test,self.t_test)))

        return self.X_train,self.X_test,self.t_train,self.t_test
