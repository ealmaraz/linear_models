import numpy as np
from numpy.linalg import inv
from basis import *
from utils import get_logger

logger = get_logger('regression')

class LinearRegressor:
    def __init__(self,basis,wML: np.ndarray = np.array([]),betaML: float = None) -> None:
        """
        Class to perform linear regression with a given basis.\n

        Attributes:
            basis (object)      : Basis for the linear regression.
            wML (np.ndarray)    : Maximum likelihood weights.
            betaML (float)      : Maximum likelihood precision.
            Phi (np.ndarray)    : Design matrix.
        """
        logger.info('Initializing regression model...')
        self.basis  = basis
        self.wML    = wML
        self.betaML = betaML


    def build_design_matrix(self,X: np.ndarray) -> None:
        """
        Build the design matrix (Phi) for the linear regression.\n
        
        Args:
            X (np.ndarray): Array with the input data.
        
        Returns:
            None.
        """
        logger.info('Building Design Matrix...')
        Phi      = np.vstack([self.basis.eval_at_point(x) for x in X])
        self.Phi = Phi


    def get_wML(self,t: np.ndarray) -> None:
        """
        Calculate the maximum likelihood weights (wML).\n

        Args:
            t (np.ndarray): Array with the target values.

        Returns:
            None.
        """
        logger.info('Calculating Maximum Likelihood Weights...')
        PhiT_Phi     = np.matmul(self.Phi.T,self.Phi)
        inv_PhiT_Phi = inv(PhiT_Phi)
        RH           = np.matmul(inv_PhiT_Phi,self.Phi.T)
        wML          = np.matmul(RH,t)
        self.wML     = wML

    def get_betaML(self,X: np.ndarray,t: np.ndarray) -> None:
        """
        Calculate the maximum likelihood precision (betaML).\n

        Args:
            X (np.ndarray)   : Array with the input data.
            t (np.ndarray)   : Array with the target values.

        Returns:
            None.
        """
        logger.info('Calculating Maximum Likelihood precision...')
        N           = len(t)
        y_at_N      = self.eval_on_dataset(X)
        delta2      = (t-y_at_N)**2
        betaML      = N/np.sum(delta2)
        self.betaML = betaML    


    def fit(self,X: np.ndarray,t: np.ndarray) -> None:
        """
        Run the linear regression algorithm.\n

        Args:
            X (np.ndarray): Array with the input data.
            t (np.ndarray): Array with the target values.

        Returns:
            None.
        """
        logger.info('Training regression model...')
        self.build_design_matrix(X)
        self.get_wML(t)
        self.get_betaML(X,t)
        logger.info('Training completed.')


    def eval_at_point(self,xval: float) -> float:
        """
        Evaluate the linear regression at a given point.\n

        Args:
            xval (float): Point to evaluate the linear regression.

        Returns:
            float: Value of the linear regression at xval.
        """
        phi_x     = self.basis.eval_at_point(xval).reshape(-1,1)
        wML_T     = self.wML.reshape(-1,1).T
        y_at_xval = np.matmul(wML_T,phi_x)
        y_at_xval = y_at_xval.flatten()[0]
        return y_at_xval


    def eval_on_dataset(self,X: np.ndarray) -> np.ndarray:
        """
        Evaluate the linear regression on a dataset.\n

        Args:
            X (np.ndarray): Array with the input data.

        Returns:
            np.ndarray: Array with the values of the linear regression evaluated
        """
        y_at_X = np.array([self.eval_at_point(x) for x in X])
        return y_at_X


    def get_mse(self,X: np.ndarray,t: np.ndarray) -> None:
        """
        Calculate the mean squared error (MSE) of the linear regression.\n

        Args:
            X (np.ndarray): Array with the input data.
            t (np.ndarray): Array with the target values.

        Returns:
            None.
        """
        y_pred = self.eval_on_dataset(X)
        N      = len(t)
        mse    = np.sum((t-y_pred)**2)/N
        self.mse = mse