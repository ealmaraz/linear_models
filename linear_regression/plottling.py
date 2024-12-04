import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from gen_data import *
from basis import *
from linear_regression import *
from utils import get_logger

logger = get_logger('plotting')

def plot_fitting_results(dataset: TrainData,list_models: list = [],export_dict={}) -> None:
    """
    Plot the linear regression results for all models in the list.\n

    Args:
        dataset (TrainData) : Object with the dataset.
        list_models (list)  : List with the models to plot.
        export (bool)       : Flag to export the plot.

    Returns:
        figure if export is False, otherwise the plot is saved.
    """
    logger.info('Creating plot with the results...')

    fig = go.Figure()

    #reference curve
    fig.add_trace(go.Scatter(x=dataset.X_ref,
                             y=dataset.h_ref,
                             mode='lines',
                             line=dict(width=4),
                             name='Ground Curve'))
    
    #training data
    fig.add_trace(go.Scatter(x=dataset.X_train,
                             y=dataset.t_train,
                             mode='markers',
                             marker=dict(size=10),
                             name='Training Data (N = {})'.format(len(dataset.X_train))))
    
    #test data
    fig.add_trace(go.Scatter(x=dataset.X_test,
                             y=dataset.t_test,
                             mode='markers',
                             marker=dict(size=10,symbol='square'),
                             name='Test Data (N = {})'.format(len(dataset.X_test))))
    #fitting curves    
    for model in list_models:
        t_fit = model.eval_on_dataset(dataset.X_ref)
        fig.add_trace(go.Scatter(x=dataset.X_ref,
                                 y=t_fit,
                                 mode='lines',
                                 line=dict(width=2,dash='dash'),
                                 name=model.basis.__class__.__name__+f' (MSE = {model.mse:.4f})'))

    #layout
    fig.update_layout(xaxis=dict(title=dict(text='X')),
                      yaxis=dict(title=dict(text='t')),
                      title='Linear Regression Fitting results',
                      template='plotly_dark')
    
    if export_dict['save_plot']:
        logger.info('Exporting plot to file...')
        fig.write_html(r'./plot.html')
        return
    else:
        return fig 



def record_in_log(dataset: TrainData,list_models: list = [],export_dict={},func_info=[]) -> None:
    """
    Create a log file with the results of the linear regression.\n

    Args:
        dataset (TrainData) : Object with the dataset.
        list_models (list)  : List with the models to log.
        export_path (str)   : Path to export the log file.

    Returns:
        None.
    """
    logger.info('Saving logs with analysis info...')

    log_path = './info.log'

    with open(log_path,"a+") as file:
        file.write(f'Dataset Parameters         :\n')
        file.write(f'Domain                     : {dataset.domain}\n')
        file.write(f'Size                       : {dataset.size}\n')
        file.write(f'Fraction of training data  : {dataset.ftrain}\n')
        file.write(f'Standard deviation of noise: {dataset.sigmaN}\n')
        file.write(f'Random seed                : {dataset.seed}\n')
        file.write(f'Ground function            : {func_info[0]}\n')
        file.write(f'Ground function parameters : {func_info[1]}\n')

        for model in list_models:
            file.write(f'Model       : {model.basis.__class__.__name__}\n')
            file.write(f'MSE         : {model.mse}\n')
            file.write(f'ML Weights  : {model.wML}\n')
            file.write(f'ML Precision: {model.betaML}\n')
