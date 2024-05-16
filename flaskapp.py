import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load ARIMA model results
with open('arima_model.pkl', 'rb') as f:
    arima_result = pickle.load(f)

# Load SARIMA model results
with open('sarima_results.pkl', 'rb') as f:
    sarima_results = pickle.load(f)

# Load ETS model results
with open('ets_results.pkl', 'rb') as f:
    ets_results = pickle.load(f)

# Load data from CSV file
data = pd.read_csv('Daily_atmoshphere_data.csv', parse_dates={'date': ['year', 'month', 'day']}, index_col='date')

# Forecasting function for ARIMA model
def forecast_arima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# Forecasting function for SARIMA model
def forecast_sarima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# Forecasting function for ETS model
def forecast_ets(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# Function to generate ACF and PACF plots


def generate_arima_graphs(data):
    st.subheader('ARIMA GRAPHS')
    plt.figure(figsize=(12, 6))
    plot_acf(data['cycle'], lags=50)
    plt.title('Autocorrelation Function (ACF) - ARIMA')
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    plot_pacf(data['cycle'], lags=50)
    plt.title('Partial Autocorrelation Function (PACF) - ARIMA')
    st.pyplot(plt)

    st.subheader('SARIMA GRAPHS')
    # Plot ACF and PACF for SARIMA
    if 'residuals' in sarima_results:
        plt.figure(figsize=(12, 6))
        plot_acf(sarima_results['residuals'], lags=50)
        plt.title('Autocorrelation Function (ACF) - SARIMA')
        st.pyplot(plt)

        plt.figure(figsize=(12, 6))
        plot_pacf(sarima_results['residuals'], lags=50)
        plt.title('Partial Autocorrelation Function (PACF) - SARIMA')
        st.pyplot(plt)
    else:
        st.error("Error: SARIMA results do not contain residuals.")

# Create a Streamlit app
def main():
    st.title('Time Series Forecasting')
    st.sidebar.title('Options')
    model_selection = st.sidebar.radio('Select Model', ('ARIMA', 'SARIMA', 'ETS'))

    st.sidebar.header('Forecast Settings')
    forecast_steps = st.sidebar.slider('Number of Days for Forecasting:', min_value=1, max_value=30, value=7)

    if model_selection == 'ARIMA':
        st.header('ARIMA Model Results')
        st.text(arima_result.summary().as_text())

        # Generate ARIMA graphs
        generate_arima_graphs(data)

        # Forecasting
        st.header('Forecasting with ARIMA')
        forecast_button_arima = st.button('Generate Forecast')

        if forecast_button_arima:
            forecast = forecast_arima(arima_result, forecast_steps)
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1)[1:]
            if len(forecast) == len(future_dates):
                st.subheader("Forecasted CO2 Levels (cycle) with ARIMA:")
                # Create a DataFrame for easier manipulation
                forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)
                st.dataframe(forecast_df.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black', 'border': '1px solid #ddd'}))
            else:
                st.error("Error: Length of forecast does not match length of future dates.")

    elif model_selection == 'SARIMA':
        st.header('SARIMA Model Results')
        st.text(sarima_results['result'].summary().as_text())

        # Generate ARIMA graphs
        generate_arima_graphs(data)

        # Forecasting
        st.header('Forecasting with SARIMA')
        forecast_button_sarima = st.button('Generate Forecast')

        if forecast_button_sarima:
            forecast = forecast_sarima(sarima_results['result'], forecast_steps)
            if len(forecast) == forecast_steps:
                st.subheader("Forecasted CO2 Levels (cycle) with SARIMA:")
                # Create a DataFrame for easier manipulation
                
                forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)
                st.dataframe(forecast_df.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black', 'border': '1px solid #ddd'}))
            else:
                st.error("Error: Length of forecast does not match the selected number of forecast steps.")

    elif model_selection == 'ETS':
        st.header('ETS Model Results')
        st.text(ets_results['result'].summary().as_text())

        # Forecasting
        st.header('Forecasting with ETS')
        forecast_button_ets = st.button('Generate Forecast')

        if forecast_button_ets:
            forecast = forecast_ets(ets_results['result'], forecast_steps)
            if len(forecast) == forecast_steps:
                st.subheader("Forecasted CO2 Levels (cycle) with ETS:")
                # Create a DataFrame for easier manipulation
                forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
                forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)
                st.dataframe(forecast_df.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black', 'border': '1px solid #ddd'}))
            else:
                st.error("Error: Length of forecast does not match the selected number of forecast steps.")

if __name__ == '__main__':
    main()
