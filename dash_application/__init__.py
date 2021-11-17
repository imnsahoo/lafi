from os import name
import dash
#import dash_core_components as dcc
from dash import dcc
from dash import html
#import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd

df = px.data.gapminder()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app =''
def create_dash_application(flask_app,final_xgb_pred_w_date,final_dtree_pred_w_date,final_rf_pred_w_date,final_actual_hour_w_date):
    #print(final_xgb_pred_w_date)
    dash_app = dash.Dash(server =flask_app, url_base_pathname ="/role/",)
    dash_app.layout = html.Div([
    dcc.Dropdown(id='dpdn2', value=final_xgb_pred_w_date, multi=True,
                 options=[{'label': x, 'value': x} for x in
                          final_xgb_pred_w_date.Date.unique()]),
    html.Div([
        dcc.Graph(id='pie-graph', figure={}, className='six columns'),
        dcc.Graph(id='my-graph', figure={}, clickData=None, hoverData=None, 
                  config={
                      'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                    
                        },
                  className='six columns'
                  )
        ])
    ])
    @dash_app.callback(
        Output(component_id='my-graph', component_property='figure'),
        Input(component_id='dpdn2', component_property='value'),
    )
    def update_graph(Date_Choosen):
        dff = df[df.Date.isin(Date_Choosen)]
        fig = px.line(data_frame=dff, x='Date', y='Working hour', color='',
                    custom_data=['Date', 'Working hour', ])
        fig.update_traces(mode='lines+markers')
        return fig
    
    @dash_app.callback(
    Output(component_id='pie-graph', component_property='figure'),
    Input(component_id='my-graph', component_property='hoverData'),
    Input(component_id='my-graph', component_property='clickData'),
    Input(component_id='my-graph', component_property='selectedData'),
    Input(component_id='dpdn2', component_property='value')
    )
    def update_side_graph(hov_data, clk_data, slct_data, Date_chosen):
        if hov_data is None:
            dff2 = df[df.final_xgb_pred_w_date.isin('Date')]
            #dff2 = dff2[dff2.Date == ]
            #print(dff2)
            fig2 = px.pie(data_frame=dff2, values='Date', names='Sales',
                        title='')
            return fig2
        else:
            #print(f'hover data: {hov_data}')
            # print(hov_data['points'][0]['customdata'][0])
            # print(f'click data: {clk_data}')
            # print(f'selected data: {slct_data}')
            dff2 = df[df.Date.isin(country_chosen)]
            hov_year = hov_data['points'][0]['x']
            dff2 = dff2[dff2.Date == hov_year]
            fig2 = px.pie(data_frame=dff2, values='pop', names='', title=f' {hov_year}')
            return fig2
    return dash_app

