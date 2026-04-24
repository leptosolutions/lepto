import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_distribution(variable, sample_weights=None, var_name="", nbins=20):
    
    """
    Plot Observed vs Predicted by category of `variable`, with Exposure on a secondary y-axis,
    using Plotly Express traces.

    Parameters
    ----------
    variable : array-like
    y : array-like
        Target on the response scale (e.g., frequency, severity, probability).
    sample_weights : array-like or None, optional
    var_name : str
    nbins : int

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure with Observed/Predicted lines and Exposure bars (secondary y).
    """

    if sample_weights is None:
        sample_weights = np.ones_like(len(variable), dtype=float)
    else:
        sample_weights = np.array(sample_weights, dtype=float)

    # Cut numeric into bins
    if (pd.api.types.is_numeric_dtype(variable)) & (len(np.unique(variable))>=nbins):
         new_var = pd.cut(variable, nbins)
         categories = new_var.cat.categories.astype(str)
         mapping = {cat: f"{i:03} : {cat}" for i, cat in enumerate(categories, start=1)}
         variable = new_var.astype(str).map(mapping)

    # --- Inputs normalization ---
    variable = np.array(variable)
    # --- Build working frame ---
    df = pd.DataFrame(
        {
            "var": variable,
            "w": sample_weights}
    )

    # --- Aggregations by category ---
    # exposure = sum of weights
    gb_expo = df.groupby("var", observed=False)["w"].sum().rename("exposure")
    df_gb = pd.concat([gb_expo], axis=1).reset_index()


    # Create subplot with secondary y-axis
    fig = make_subplots()

    # Exposure bars
    fig.add_trace(go.Bar(
        x=df_gb["var"], y=df_gb["exposure"],
        name="Exposure",
        opacity=0.5
    ))

   
    # Layout
    fig.update_layout(
        title=f"Distribution: {var_name}" if var_name else "Distribution",
        xaxis=dict(title=var_name or "Variable", type="category"),
        yaxis=dict(title="Value"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    return fig

def weighted_qcut(values, weights, quantile, **kwargs):
    """ Quantile cuts weighted

    Paramaters
    -----------
    values : 1d array-like
        Value to cut

    weights : 1d array-like
        weights used for qcut

    quantile : int
        Number of quantiles

    **kwargs : Dict
        Options for pd.cut() function

    Returns
    -------
    bins : 1d array-like
        values binned

    """
    # Create pandas frame
    df = pd.DataFrame({'values': np.array(values),
                       'weights': np.array(weights)})
    # Define quantile
    quantiles = np.linspace(0, 1, quantile + 1)
    # Cutting
    df_sort = df.sort_values('values')
    order = df_sort['weights'].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    # Reorder bins and output values
    bins.sort_index(axis=0, inplace=True)
    bins = bins.values
    return bins

def lift_chart(y, pred, weights=None, bins=10):
    """ Compute and plot lift chart

    Parameters
    ----------
    y : 1d array-like
        Ground truth (correct) label

    pred : 1d array-like
        Predicted label

    weights : 1d array-like
        weights

    bins : int
        Number of bins
    Return
    --------
    fig : plotly FigureWidget

    """
    if weights is None:
        weights = np.ones(len(y))
    # Data
    lift_data = pd.DataFrame({'y': y,
                              'pred': pred,
                              'weights': weights})
    # Compute y_pred_weighted
    lift_data['pred_w'] = lift_data['pred'] * lift_data['weights']
    lift_data['y_w'] = lift_data['y'] * lift_data['weights']

    # Compute lift chart
    lift_data = lift_data.sort_values('pred', ascending=[1])
    lift_data['bin_lift'] = weighted_qcut(lift_data['pred'],
                                          lift_data['weights'],
                                          bins,
                                          duplicates='drop')

    lift_data_gb = lift_data.groupby('bin_lift', observed=False).agg({
        'y_w': 'sum',
        'weights': 'sum',
        'pred_w': 'sum'
    })
    lift_data_gb['y_w'] = lift_data_gb['y_w'] / lift_data_gb['weights']
    lift_data_gb['pred_w'] = lift_data_gb['pred_w'] / lift_data_gb['weights']

    # Bin ranges
    y_pred_min = lift_data.groupby('bin_lift', observed=False)['pred'].min()
    y_pred_max = lift_data.groupby('bin_lift', observed=False)['pred'].max()
    lift_data_gb['pred_range'] = '(' + y_pred_min.round(5).astype('str') + ',' + y_pred_max.round(5).astype('str') + ']'


    ###################################
    # Plot lift
    trace1 = go.Scatter(
        x = list(lift_data_gb['pred_range']),
        y = list(lift_data_gb['y_w']),
        name = 'Actual',
        mode = 'lines+markers',
        yaxis = 'y1'
    )

    trace2 = go.Scatter(
        x=list(lift_data_gb['pred_range']),
        y=list(lift_data_gb['pred_w']),
        name='Prediction',
        mode='lines+markers',
        yaxis='y1'
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title='Lift Chart',
        xaxis=dict(title='Prediction bins',
                   tickvals=lift_data_gb['pred_range'],
                   ticktext=lift_data_gb['pred_range']),
        yaxis=dict(title='Average response/weight'),
        showlegend=True)
    fig=go.Figure(data=data, layout=layout)

    return fig
