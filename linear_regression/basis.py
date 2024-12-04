import numpy as np
from utils import get_logger

logger = get_logger('basis')

class PolynomialBasis:
    def __init__(self, degree: int = 1) -> None:
        """
        Class to generate a polynomial basis for linear regression.\n
        
        Attributes:
            degree (int): Maximum degree of the polynomial basis.
        """
        self.degree = degree
        logger.info('{} created.'.format(self.__repr__()))


    def eval_at_point(self,xval: float) -> np.ndarray:
        """
        Evaluate the polynomial basis at a given point.\n
        
        Args:
            xval (float): Point to evaluate the basis.\n

        Returns:
            np.ndarray: Array with the basis functions evaluated at xval.\n
                        [1, x^1 ... *x^n]\n
        """
        return np.array([np.power(xval,i) for i in range(self.degree+1)])


    def __repr__(self) -> str:
        """
        Print the representation of the polynomial basis.\n
        """
        cls   = self.__class__.__name__
        funcs = [1]+[f'x^{i}' for i in range(1,self.degree+1)]
        repr = f'{cls}(degree={self.degree},basis_funcs={funcs})'
        return repr
    

class GaussianBasis:
    def __init__(self,centers: list = [],std: float = 1) -> None:
        """
        Class to generate a Gaussian basis for linear regression.\n
    
        Attributes:
            centers (list): List of centers for the Gaussian basis.
            std (float)   : Standard deviation of the Gaussian basis.
        """
        self.centers = centers
        self.std     = std
        logger.info('{} created.'.format(self.__repr__()))
        

    def eval_at_point(self,xval: float) -> np.ndarray:
        """
        Evaluate the Gaussian basis at a given point.\n
            [exp(-0.5*((x-c)/std)^2)\

        Args:
            xval (float): Point to evaluate the basis.

        Returns:
            np.ndarray: Array with the basis functions evaluated
                [1, exp(-0.5*((x-c1)/std)^2), ... exp(-0.5*((x-cn)/std)^2)]
        """
        one_arr = np.array([1])
        exp_arr = np.exp(-0.5*((xval-self.centers)/self.std)**2)
        
        return np.concatenate((one_arr,exp_arr)) 


    def __repr__(self)-> str:
        """
        Print the representation of the Gaussian basis.\n
        """
        cls   = self.__class__.__name__
        repr = f'{cls}(num_centers={len(self.centers)},spread={self.std})'
        return repr
        