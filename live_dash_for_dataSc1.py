from dash import Dash, html, dcc, dash_table, Output, Input, State
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from ml_model_api import query_sentiment_model
from newsapi import NewsApiClient

# --- NEWS API ---
NEWS_API_KEY = ""
newsapi = NewsApiClient(api_key=NEWS_API_KEY)



app = Dash()

app.layout = html.Div([
    html.H1('My Financial Dashboard'),
    dcc.Input(id='ticker-input',
              placeholder='Search for symbols from Yahoo Finance',
              style={'width': '50%'}),
    html.Button(id='submit-button', children='Submit', n_clicks=0),
    html.Br(),
    html.Br(),
    # The Tabs component starts here
    dcc.Tabs([
        dcc.Tab(label='Charts',
                children=[
                    html.H3('Price Candlestick Chart (1-Day)'),
                    dcc.Graph(id='stock-graph'),
                    html.Div(id='ml-sentiment-output',
                             style={'fontSize': 20, 'textAlign': 'center', 'padding': '10px'}),
                    html.Hr(),
                    html.H3('30-Day Rolling Volatility (1-Year)'),
                    dcc.Graph(id='volatility-graph')
                ]),
        dcc.Tab(label='Recent Data',
                children=[
                    html.Div(id='latest-price-div'),
                    dash_table.DataTable(id='stock-table')
                ])]),
   dcc.Interval(id='chart-interval', interval=1000 * 60 * 15, n_intervals=0),
   dcc.Interval(id='table-interval', interval=1000 * 60, n_intervals=0)
    ])



# Callback for the main candlestick chart
@app.callback(
    Output('stock-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    Input('chart-interval', 'n_intervals'),
    State('ticker-input', 'value')
)
def update_chart(button_click, chart_interval, ticker):
    if not ticker:
        return go.Figure()  # Return an empty figure if no ticker

    price = yf.Ticker(ticker).history(period='1d', interval='15m').reset_index()
    if not price.empty:
        fig = go.Figure(data=go.Candlestick(
            x=price['Datetime'],
            open=price['Open'],
            high=price['High'],
            low=price['Low'],
            close=price['Close']
        ))
        fig.update_layout(title=f'{ticker.upper()} Intraday Price')
        return fig
    return go.Figure()


# --- CALLBACK FOR VOLATILITY ---
@app.callback(
    Output('volatility-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    State('ticker-input', 'value')
)
def update_volatility_chart(n_clicks, ticker):
    if n_clicks == 0 or not ticker:
        return go.Figure()  # Return an empty figure if button not clicked or no ticker

    # 1. Data Acquisition: Get 1 year of daily data
    price_data = yf.Ticker(ticker).history(period='1y')['Close']

    if price_data.empty:
        return go.Figure()

    # 2. Data Transformation: Calculate daily returns
    daily_returns = price_data.pct_change()

    # 3. Apply the Rolling Calculation
    window_size = 30
    # Calculate rolling standard deviation and annualize it
    volatility = daily_returns.rolling(window=window_size).std() * (252 ** 0.5)

    # 4. Create the Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=volatility.index,
        y=volatility.values,
        mode='lines',
        name='30-Day Volatility'
    ))
    fig.update_layout(
        title=f'{ticker.upper()} Annualized 30-Day Rolling Volatility',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility'
    )
    return fig


# Callback for the data table (no changes needed here)
@app.callback(
    Output('latest-price-div', 'children'),
    Output('stock-table', 'data'),
    Input('submit-button', 'n_clicks'),
    Input('table-interval', 'n_intervals'),
    State('ticker-input', 'value')
)
def update_table(button_click, table_interval, ticker):
    if not ticker:
        return '', []

    price = yf.Ticker(ticker).history(period='1d', interval='1m').reset_index().tail(10)
    if not price.empty:
        latest_price = price['Close'].iloc[-1]
        latest_time = price['Datetime'].max().strftime('%b %d %Y %I:%M:%S %p')
        return (f'The latest price is ${latest_price:,.2f} at {latest_time} UTC',
                price.to_dict('records'))
    else:
        return f'No data for ticker {ticker} on Yahoo Finance', []


# --- NEW CALLBACK FOR HUGGING FACE MODEL ---
@app.callback(
    Output('ml-sentiment-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('ticker-input', 'value')
)
def update_ml_sentiment(n_clicks, ticker):
    if n_clicks == 0 or not ticker:
        return "Enter a ticker to get ML sentiment on the latest news."

    # 1. Get a recent headline to analyze
    try:
        headlines = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=1)
        if not headlines['articles']:
            return f"No recent news found for {ticker.upper()}."
        headline_text = headlines['articles'][0]['title']
    except Exception as e:
        return "Could not fetch news."

    # 2. Call the ML model API with the headline
    result = query_sentiment_model(headline_text)

    if result is None:
        return "ML sentiment analysis failed."

    # 3. Format and display the result
    label = result.get('label', 'UNKNOWN').capitalize()
    score = result.get('score', 0)
    emoji = "😊" if label == 'Positive' else "😞"

    return f"Latest Headline: '{headline_text}' | ML Sentiment: {label} {emoji} (Confidence: {score:.0%})"


if __name__ == '__main__':
    app.run(debug=True)