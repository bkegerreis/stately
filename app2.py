import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from umap import UMAP
import json

def make_graphs(datafile, algorithm, k, overlay):
    no_overlay = False
    if overlay=='None':
        no_overlay = True
        overlay = 'Medicaid Expansion (2017)' #placeholder, not actually used
    df = pd.read_csv(datafile, index_col=0)
    df_tmp = df.drop(['abbrev', 'Medicaid Expansion (2017)', '2016 Election', 'Median Household Income (2018)'], axis=1)
    umap = UMAP(n_neighbors=5, random_state=1035411).fit_transform(df_tmp)
    df['X'] = umap[:,0]
    df['Y'] = umap[:,1]
    clusters = None
    if algorithm=='gmm':
        clusters = GaussianMixture(k, random_state=1035411).fit_predict(df_tmp) + 1
    elif algorithm=='hc':
        clusters = AgglomerativeClustering(k).fit_predict(df_tmp) + 1
    else:
        clusters = KMeans(k, random_state=1035411).fit_predict(df_tmp) + 1
    df['cluster'] = clusters
    df['cluster'] = df['cluster'].astype('category')
    df['cluster'].cat.set_categories([x + 1 for x in range(max(df['cluster']))])
    df_for_bar = df.drop(['X', 'Y', 'abbrev', 'Medicaid Expansion (2017)', '2016 Election', 'Median Household Income (2018)'], axis=1)
    df_for_bar = df_for_bar.groupby('cluster').mean()
    idx_col = df_for_bar.index.name
    color_name_dict = {
        'age_for_app.csv': 'Age Group',
        'health_coverage_for_app.csv': 'Health Coverage',
        'industries_for_app.csv': 'Industry Group',
        'occupations_for_app.csv': 'Occupation Group'
    }
    color_name = color_name_dict[datafile]
    y_name = 'Percentage of Population'
    melted_for_bar = pd.melt(df_for_bar.reset_index(), id_vars=idx_col, var_name=color_name, value_name=y_name)
    overlay_dict = {
        'Medicaid Expansion (2017)': {'yes':'rgb(128,128,255)', 'no':'rgb(255,128,128)'},
        '2016 Election': {'Clinton':'rgb(128,128,255)', 'Trump':'rgb(255,128,128)'},
        'Median Household Income (2018)': {'foo':'bar'}
    }
    orders_dict = {
        'Medicaid Expansion (2017)': ['yes', 'no'],
        '2016 Election': ['Clinton', 'Trump'],
        'Median Household Income (2018)': ['foo']
    }
    hover_dict = {k:':.2f' for k in df.columns.drop(['cluster', 'X', 'Y', 'abbrev', 'Medicaid Expansion (2017)', '2016 Election', 'Median Household Income (2018)'])}
    hover_dict['X'] = False
    hover_dict['Y'] = False
    if no_overlay:
        hover_dict[overlay] = False
    region_dict = {
        'AL': 'South', 'AK': 'Other', 'AZ': 'West', 'AR': 'South', 'CA': 'West',
        'CO': 'West', 'CT': 'Northeast', 'DE': 'South', 'FL': 'South', 'GA': 'South',
        'HI': 'Other', 'ID': 'West', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest',
        'KS': 'Midwest', 'KY': 'South', 'LA': 'South', 'ME': 'Northeast', 'MD': 'South',
        'MA': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest', 'MS': 'South', 'MO': 'Midwest',
        'MT': 'West', 'NE': 'Midwest', 'NV': 'West', 'NH': 'Northeast', 'NJ': 'Northeast',
        'NM': 'West', 'NY': 'Northeast', 'NC': 'South', 'ND': 'Midwest', 'OH': 'Midwest',
        'OK': 'South', 'OR': 'West', 'PA': 'Northeast', 'RI': 'Northeast', 'SC': 'South',
        'SD': 'Midwest', 'TN': 'South', 'TX': 'South', 'UT': 'West', 'VT': 'Northeast',
        'VA': 'South', 'WA': 'West', 'WV': 'South', 'WI': 'Midwest', 'WY': 'West'
    }
    region_color_dict = {
        'Northeast': 'blue', 'South': 'red', 'Midwest': 'yellow', 'West': 'green', 'Other': 'black'
    }
    regions = [v for k,v in region_dict.items()]
    ari = adjusted_rand_score(clusters, regions)
    ari_output = 'Adjusted Rand Index between clusters and regions: ' + str(round(ari, 2))
    fig1 = px.scatter(
        df, x='X', y='Y',
        symbol='cluster',
        color=overlay,
        color_discrete_map=overlay_dict[overlay],
        color_continuous_scale='Oryel',
        category_orders={
            'cluster': [x+1 for x in range(max(df['cluster']))],
            overlay: orders_dict[overlay]
        },
        hover_name=df.index.values,
        hover_data=hover_dict,
        text=df['abbrev'],
        custom_data = ['abbrev'],
        template='simple_white'
    )
    fig1.update_traces(marker=dict(size=25, line=dict(width=2)))
    for t in fig1.data:
        t.marker.line.color = [region_color_dict[region_dict[abbrev]] for abbrev in t.text]
    if no_overlay:
        fig1.update_traces(marker=dict(color='rgb(128,128,255)'))
    fig1.update_xaxes(title_text='UMAP-1', showticklabels=False, ticks='', showline=False)
    fig1.update_yaxes(title_text='UMAP-2', showticklabels=False, ticks='', showline=False)
    fig1.update_layout(plot_bgcolor='rgb(240,240,250)')
    if overlay == 'Median Household Income (2018)':
        fig1.update_layout(coloraxis_colorbar=dict(x=0, xanchor='right'))
    
    fig2 = px.bar(
        melted_for_bar, x=idx_col, y=y_name,
        color=color_name, color_discrete_sequence=px.colors.qualitative.Safe,
        template='simple_white'
    )
    fig2.update_xaxes(tickvals=[x+1 for x in range(max(melted_for_bar['cluster']))], ticks='', showline=False)
    fig2.update_yaxes(showticklabels=False, ticks='', showline=False)
    fig2.update_layout(plot_bgcolor='rgb(240,240,250)')
    return fig1, fig2, ari_output

# US Census regions, but AK and HI have been moved from West to Other
region_dict = {
    'AL': 'South', 'AK': 'Other', 'AZ': 'West', 'AR': 'South', 'CA': 'West',
    'CO': 'West', 'CT': 'Northeast', 'DE': 'South', 'FL': 'South', 'GA': 'South',
    'HI': 'Other', 'ID': 'West', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest',
    'KS': 'Midwest', 'KY': 'South', 'LA': 'South', 'ME': 'Northeast', 'MD': 'South',
    'MA': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest', 'MS': 'South', 'MO': 'Midwest',
    'MT': 'West', 'NE': 'Midwest', 'NV': 'West', 'NH': 'Northeast', 'NJ': 'Northeast',
    'NM': 'West', 'NY': 'Northeast', 'NC': 'South', 'ND': 'Midwest', 'OH': 'Midwest',
    'OK': 'South', 'OR': 'West', 'PA': 'Northeast', 'RI': 'Northeast', 'SC': 'South',
    'SD': 'Midwest', 'TN': 'South', 'TX': 'South', 'UT': 'West', 'VT': 'Northeast',
    'VA': 'South', 'WA': 'West', 'WV': 'South', 'WI': 'Midwest', 'WY': 'West'
}
region_color_dict = {
    'Northeast': 'blue', 'South': 'red', 'Midwest': 'yellow', 'West': 'green', 'Other': 'black'
}
usa_df = pd.DataFrame({
    'abbrev': [k for k in region_dict]
})
usa_df['region'] = [region_dict[k] for k in usa_df['abbrev']]
region_map = px.choropleth(
    usa_df, locations=[k for k in region_dict], locationmode='USA-states', scope='usa',
    color='region', color_discrete_map=region_color_dict,
    category_orders={'region':['Northeast', 'South', 'Midwest', 'West', 'Other']},
    title='Symbol outlines denote region'
)

# here's where the magic happens
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='stately.io'),
    html.Div(children="""
        Disrupting tomorrow, today!
    """),
    html.Div(children="""
        Figures may take a few seconds to display properly
    """),
    # options div
    html.Div([
        html.Label('Choose your data'),
        dcc.Dropdown(
            id='data-choice',
            options=[
                {'label': 'Age', 'value':'age_for_app.csv'},
                {'label': 'Health Coverage', 'value':'health_coverage_for_app.csv'},
                {'label': 'Industries', 'value':'industries_for_app.csv'},
                {'label': 'Occupations', 'value':'occupations_for_app.csv'}
            ],
            value='health_coverage_for_app.csv'
        ),
        html.Label('Choose a clustering algorithm'),
        dcc.Dropdown(
            id='algo-choice',
            options=[
                {'label':'K Means', 'value':'kmeans'},
                {'label':'Gaussian Mixture', 'value':'gmm'},
                {'label':'Hierarchical Clustering', 'value':'hc'},
            ],
            value='kmeans'
            ),
        html.Label('Choose a number of clusters'),
        dcc.RadioItems(
            id='cluster-choice',
            options=[
                {'label':'2', 'value':2},
                {'label':'3', 'value':3},
                {'label':'4', 'value':4},
                {'label':'5', 'value':5}
            ],
            value=4
        ),
        html.Label('Choose a color overlay'),
        dcc.Dropdown(
            id='overlay-choice',
            options=[
                {'label': 'Medicaid Expansion (2017)', 'value':'Medicaid Expansion (2017)'},
                {'label': '2016 Election', 'value':'2016 Election'},
                {'label': 'Median Household Income (2018)', 'value':'Median Household Income (2018)'},
                {'label': 'None', 'value':'None'}
            ],
            value='Medicaid Expansion (2017)'
        ),
    ]),
    # adjusted rand div
    html.Div(
        id='ari-output'
    ),
    # scatter and region map div
    html.Div([
        html.Div([
            dcc.Graph(
                id='cluster-umap',
                hoverData={'points': [{'customdata': ['AL']}]}
            )
        ], style={'display':'inline-block', 'width':'75%'}),
        html.Div([
            dcc.Graph(
                id='region-map',
                figure=region_map
            ),
        ], style={'display':'inline-block', 'float':'right', 'width':'24%'})
    ]),
    # stacked bar div
    html.Div([
        html.Div([
            dcc.Graph(
                id='state-bar'
            )
        ], style={'display':'inline-block', 'width':'24%'}),
        html.Div([
            dcc.Graph(
                id='cluster-stacked-bar'
            )
        ], style={'display':'inline-block', 'float':'right', 'width':'75%'})
    ])
])

@app.callback(
    [Output('cluster-umap', 'figure'),
    Output('cluster-stacked-bar', 'figure'),
    Output('ari-output', 'children')],
    [Input('data-choice', 'value'),
    Input('algo-choice', 'value'),
    Input('cluster-choice', 'value'),
    Input('overlay-choice', 'value')]
)
def update_graphs(data_source, algorithm, cluster_number, overlay):
    fig1, fig2, ari = make_graphs(data_source, algorithm, cluster_number, overlay)
    return fig1, fig2, ari

@app.callback(
    Output('state-bar', 'figure'),
    [Input('data-choice', 'value'),
    Input('cluster-umap', 'hoverData')]
)
def make_state_bar(data_source, hoverData):
    state_name = hoverData['points'][0]['customdata'][0]
    d = pd.read_csv(data_source, index_col=0)
    df = d[d['abbrev']==state_name]
    dff = df.drop(['Medicaid Expansion (2017)', '2016 Election', 'Median Household Income (2018)'], axis=1)
    m = pd.melt(dff, id_vars='abbrev', var_name='Category', value_name='Percentage of Population')
    fig = px.bar(m, x='abbrev', y='Percentage of Population', color='Category', color_discrete_sequence=px.colors.qualitative.Safe, template='simple_white', )
    fig.update_xaxes(title_text=state_name, showticklabels=False, ticks='', showline=False)
    fig.update_yaxes(showticklabels=False, ticks='', showline=False)
    fig.update_layout(plot_bgcolor='rgb(240,240,250)', showlegend=False)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)