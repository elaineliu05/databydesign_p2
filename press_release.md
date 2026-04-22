# Fortune Telling or Forecasting: Predicting Stock Prices Over Time

## Hook

Has someone ever told you to buy or sell stocks? Being able to accurately predict stocks can lower your financial risk and maximize profit, but it isnt exactly an easy task. Predicting exact future prices requires a lot of data, expertise, and background, but you can start by making smarter, data driven predictions with time series data.

## Problem Statement

Trying to forecast stock prices is a colossal task, especially since there is a ton of data, news, and opinions about how stocks will change. It can be hard to narrow down what actually matters - but using time series analysis, all we need to know is what has happened in the past. The stock market is also a vast space - to make our task more measurable, it can be helpful to focus on a specific index, such as the S&P 500, to get started.

## Solution Description

In this research, we aim to predict the future closing price of the S&P 500 Index using historical time series open/high/low/close data from Alpha Vantage. Instead of overwhelming information on economic signals and societal factors, we just use past S&P 500 data to forecast future S&P 500 prices. To do this, we will analyze json files using a document database model for more flexibility. As for the actual forecasts, we will create and evaluate a machine learning model to learn patterns and trends in the data. With this information, we hope you will be able to make more informed decisions and gain valuable insights about your stocks.

## Chart
<img width="805" height="599" alt="image" src="https://github.com/user-attachments/assets/dabdc713-04a0-4bae-b5e9-5478e2d6bf70" />

The chart above shows the close prices of S&P 500 over time. We can see that there is definitely an incresing trend, but there are peaks and lows throughout the years. It will be interesting to see if there is a robust enough pattern/trend to determine whether we can accurately predict stock prices or not.
