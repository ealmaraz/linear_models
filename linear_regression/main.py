import datetime
import logging
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys
import shutil

from gen_data import *
from basis import *
from linear_regression import *
from plottling import *
from utils import get_logger

logger = get_logger('main') 

def ground_func(X,**params):
    """
    Reference function for the dataset.\n
    Args:
        X (np.ndarray): Array with the input data.
        **params (dict): Parameters for the reference function.

    Returns:
        np.ndarray: Array with the reference function evaluated at X.
            h = Ae*exp(-decay*X)+offset)*As*sin(freq*X+phase) 
    """
    params = {'Ae':1,'decay':1,'offset':0,'As':1,'freq':1,'phase':0,**params}
    exp_part  = params['Ae']*np.exp(-params['decay']*X)+params['offset']
    sin_part  = params['As']*np.sin(params['freq']*X+params['phase'])
    return exp_part*sin_part


def main(export_dict: dict) -> None:

    #parameters of the experiment
    domain     = [0.8,1.9]
    size       = 100
    ftrain     = 0.7
    sigmaN     = 0.05
    seed       = None
    func_desc  = "h = Ae*exp(-decay*X)+offset)*As*sin(freq*X+phase)"
    exp_params = {'Ae':3,'decay':2.1}
    sin_params = {'As':1,'freq':2*np.pi,'phase':0.3}

    logger.info('Starting analysis...')
    #creation of dataset
    params      = {**exp_params,**sin_params}
    dataset     = TrainData(domain=domain,size=size,ftrain=ftrain,sigmaN=sigmaN,seed=seed)
    dataset.create_ground_curve(ground_func=ground_func,numpoints=100,**params)
    dataset.create_dataset(ground_func,**params)
    X_train,X_test,t_train,t_test = dataset.split_data()

    #training linear regressor models: polynomial basis & Gaussian basis
    poly_basis  = PolynomialBasis(degree=3)
    linear_poly = LinearRegressor(poly_basis)
    linear_poly.fit(X_train,t_train)
    linear_poly.get_mse(X_test,t_test)

    #centers      = X_train[1:-1:4]
    centers      = np.linspace(domain[0],domain[1],10)
    gauss_basis  = GaussianBasis(centers=centers,std=1)
    linear_gauss = LinearRegressor(gauss_basis)
    linear_gauss.fit(X_train,t_train)
    linear_gauss.get_mse(X_test,t_test)

    #logging results
    record_in_log(dataset=dataset,list_models=[linear_poly,linear_gauss],export_dict=export_dict,func_info=[func_desc,params])

    #plot creation
    if export_dict['save_plot']:
        plot_fitting_results(dataset=dataset,list_models=[linear_poly,linear_gauss],export_dict=export_dict)
    else:
        fig = plot_fitting_results(dataset=dataset,list_models=[linear_poly,linear_gauss],export=False,export_dict=export_dict)
        fig.show()

    logger.info('Analysis completed.')
    #Needed to close the logging handlers and release info_log file to be moved to the export folder
    logging.shutdown()
    
    #store log & figure (if produced) in a separate folder
    export_dir = os.path.join('./logs/',export_dict['export_prefix'])
    os.makedirs(export_dir)
    shutil.move('./info.log',os.path.join(export_dir,'info.log'))
    if export_dict['save_plot']:
        shutil.move('./plot.html',os.path.join(export_dir,'plot.html'))



if __name__ == "__main__":    

    log_dir = './logs/'
    save_plot  = True

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    export_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_dict   = {'export_prefix':export_prefix,'save_plot':save_plot}

    main(export_dict=export_dict)


