import logging
import numpy as np
import numpy.typing as npt
import random

def sample_domain(lims: list, N: int, seed: int = None) -> npt.ArrayLike:
    """
    Sample the domain of the function.\n

    Args:
        lims (list)         : List of two elements, the lower and upper limits of the domain.
        N (int)             : Number of samples to generate.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: Array of samples from the domain.
    """

    if seed != None:
        random.seed(seed)

    X = np.random.uniform(low=lims[0], high=lims[1], size=N)

    return X


def add_gaussian_noise(size: int, sigma: float, seed: int = None) -> npt.ArrayLike:
    """
    Add Gaussian noise to the data.\n

    Args:
        size (int)          : Size of the data.
        sigma (float)       : Standard deviation of the noise.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: Array of Gaussian noise.
    """
    
    if seed != None:
        random.seed(seed)
    
    NX = np.random.normal(loc=0,scale=sigma,size=size)
    
    return NX


def get_logger(name) -> logging.Logger:
    """
    Create a logger object.\n

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Logger object.
    """
    #logger = logging.getLogger(name)
    #logger.setLevel(logging.INFO)

    log_format = '%(asctime)s: %(name)s [%(levelname)s] %(message)s'
    logging.basicConfig(filename='info.log',format=log_format, filemode='a',datefmt='%Y-%d-%m %I:%M:%S',level=logging.DEBUG)
    # logger       = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler(export_dict['log_file_path'])
    # fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s: [%(levelname)s] %(message)s',datefmt='%Y-%d-%m %I:%M:%S')
    # fh.setFormatter(formatter)
    ch.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(ch)
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    return logging.getLogger(name)
