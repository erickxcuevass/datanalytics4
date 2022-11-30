from dash import Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors, ensemble
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

path = "WORKFILE Supsa Energy Audit Information - Marzo 2022 - Actualizada.xlsx"

df = pd.read_excel(path)

df.head()

df.columns = ['ID','PROD_LINE','PLATAFORM','FAMILIA','TEST_COMPLETION_DATE',
               'REFRIGERANT','MODEL_NUM_TESTED','SERIAL_NUM','TERMOSTATO/TERMISTOR','POSITION',
               'TARGET','ENERGY_CONSUMED', 'BELOW_RATING_POINT','RC_TEMP_AVG_DF1', 'RC1_TEMP_DF1', 
               'RC2_TEMP_DF1', 'RC3_TEMP_DF1','FC_TEMP_AVG_DF1', 'FC1_TEMP_DF1', 'FC2_TEMP_DF1', 
               'FC3_TEMP_DF1','ENERGY_USAGE_DF1', 'RUN_TIME_DF1', 'AVG_AMBIENT_TEMP_DF1','SECOND_POINT_TEMP_SETTING_DF2',
               'RC_TEMP_AVG_DF2', 'RC1_TEMP_DF2', 'RC2_TEMP_DF2', 'RC3_TEMP_DF2', 'FC_TEMP_AVG_DF2', 'FC1_TEMP_DF2', 
               'FC2_TEMP_DF2', 'FC3_TEMP_DF2', 'ENERGY_USAGE_DF2', 'RUN_TIME_DF2','AVG_AMBIENT_TEMP_DF2','ABILITY', 'COMPRESSOR','SUPPLIER', 'E-STAR/STD']
df.columns
df.dtypes


newdf = df[['RC_TEMP_AVG_DF1','FC_TEMP_AVG_DF1','RC_TEMP_AVG_DF2','FC_TEMP_AVG_DF2','ENERGY_USAGE_DF1','RUN_TIME_DF1','ENERGY_USAGE_DF2','RUN_TIME_DF2']].copy()


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server


models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor,'Random Forest': ensemble.RandomForestRegressor}

df_ref1 = df.groupby('REFRIGERANT').get_group('R134')
df_ref2 = df.groupby('REFRIGERANT').get_group('R600')
df_fam1 = df.groupby('FAMILIA').get_group('4W3G80')
df_fam2 = df.groupby('FAMILIA').get_group('4W3G12')
df_fam3 = df.groupby('FAMILIA').get_group('6w3n80')


control1 = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x_variable",
                    options=[
                        {"label":col,"value":col} for col in newdf.columns
                    ],
                    value='RC_TEMP_AVG_DF1',
                ),
            ]
        ),
    ],
    color="warning", 
    body=True,
)

control2 = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y_variable",
                    options=[
                        {"label":col,"value":col} for col in newdf.columns
                    ],
                    value='FC_TEMP_AVG_DF1',
                ),
            ]
        ),
    ],
    color="warning", 
    body=True,
)

control3 = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Select model:"),
                dcc.Dropdown(
                    id="regmodel",
                    options=["Regression", "Decision Tree", "k-NN", "Random Forest"],
                    value='Decision Tree',
                    clearable=False

                ),
            ]
        ),
    ],
    color="warning", 
    body=True,
)
        
       
app.layout = dbc.Container(
    [
        html.H1("Dashboard Tendencias"),
        html.Hr(),
        dbc.Row(
            [   
                dbc.Col(control2, md=4),
                dbc.Col(control1, md=4),
                dbc.Col(control3, md=4),
            ],
            align="center",
        ),
        dbc.Row(
            [   
                dbc.Col(dcc.Graph(id="reg_general"), md=4),
                dbc.Col(dcc.Graph(id="reg_ref1"), md=4),
                dbc.Col(dcc.Graph(id="reg_ref2"), md=4),
            ],
            align="center",
        ),
        dbc.Row(
            [   
                dbc.Col(dcc.Graph(id="reg_fam1"), md=4),
                dbc.Col(dcc.Graph(id="reg_fam2"), md=4),
                dbc.Col(dcc.Graph(id="reg_fam3"), md=4),
            ],
            align="center",
        ),
    ],
    fluid=True,
)
                
#GENERAL

@app.callback(
    Output("reg_general", "figure"),
    [
        Input("x_variable", "value"),
        Input("y_variable", "value"),
        Input("regmodel", "value"),
    ],
) 
        
def regresion_graph(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        newdf[X], newdf[y], test_size = 0.35, random_state=42)

    model = models[name]()
    model.fit(X_train.values.reshape(-1, 1), y_train)

    x_range = np.linspace(newdf[X].min(), newdf[X].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')

    ],     layout = {"xaxis": {"title": X}, "yaxis": {"title": y}, "title":"GENERAL"})
    

    return fig

    
#refrigerante1 

@app.callback(
    Output("reg_ref1", "figure"),
    [
        Input("x_variable", "value"),
        Input("y_variable", "value"),
        Input("regmodel", "value"),
    ],
) 

def regresion_graph(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        df_ref1[X], df_ref1[y], test_size = 0.35, random_state=42)

    model = models[name]()
    model.fit(X_train.values.reshape(-1, 1), y_train)

    x_range = np.linspace(df_ref1[X].min(), df_ref1[X].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
        
    ],     layout = {"xaxis": {"title": X}, "yaxis": {"title": y}, "title":"REF R134"})


    return fig


#refrigerante2

@app.callback(
    Output("reg_ref2", "figure"),
    [
        Input("x_variable", "value"),
        Input("y_variable", "value"),
        Input("regmodel", "value"),
    ],
) 
   
def regresion_graph(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        df_ref2[X], df_ref2[y], test_size = 0.35, random_state=42)

    model = models[name]()
    model.fit(X_train.values.reshape(-1, 1), y_train)

    x_range = np.linspace(df_ref2[X].min(), df_ref2[X].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
    
    ],     layout = {"xaxis": {"title": X}, "yaxis": {"title": y}, "title":"REF R600"}
)

    return fig


#familia1

@app.callback(
    Output("reg_fam1", "figure"),
    [
        Input("x_variable", "value"),
        Input("y_variable", "value"),
        Input("regmodel", "value"),
    ],
) 
       
def regresion_graph(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        df_fam1[X], df_fam1[y], test_size = 0.35, random_state=42)

    model = models[name]()
    model.fit(X_train.values.reshape(-1, 1), y_train)

    x_range = np.linspace(df_fam1[X].min(), df_fam1[X].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')

    ],     layout = {"xaxis": {"title": X}, "yaxis": {"title": y}, "title":"FAM 4W3G80"})

    
    return fig


    
#familia2

@app.callback(
    Output("reg_fam2", "figure"),
    [
        Input("x_variable", "value"),
        Input("y_variable", "value"),
        Input("regmodel", "value"),
    ],
) 
     
def regresion_graph(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        df_fam2[X], df_fam2[y], test_size = 0.35, random_state=42)

    model = models[name]()
    model.fit(X_train.values.reshape(-1, 1), y_train)

    x_range = np.linspace(df_fam2[X].min(), df_fam2[X].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
        
    ],    layout = {"xaxis": {"title": X}, "yaxis": {"title": y}, "title":"FAM 4W3G12"})


    return fig



#familia3

@app.callback(
    Output("reg_fam3", "figure"),
    [
        Input("x_variable", "value"),
        Input("y_variable", "value"),
        Input("regmodel", "value"),
    ],
) 
        
def regresion_graph(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        df_fam3[X], df_fam3[y], test_size = 0.35, random_state=42)

    model = models[name]()
    model.fit(X_train.values.reshape(-1, 1), y_train)

    x_range = np.linspace(df_fam3[X].min(), df_fam3[X].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')

    ],    
        layout = {"xaxis": {"title": X}, "yaxis": {"title": y}, "title":"FAM 6w3n80"} )

    return fig

def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in newdf.columns
    ]

# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x_variable", "options"), [Input("y_variable", "value")])(
    filter_options
)
app.callback(Output("y_variable", "options"), [Input("x_variable", "value")])(
    filter_options
)

app.run_server()
