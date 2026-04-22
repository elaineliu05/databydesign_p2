# DS 4320 Project 2: Forecasting S&P 500 Stock Prices Using Time-Series Analysis

Executive summary: The goal of this github repository is to forecast the future closing price of the S&P 500 Index using historical time series data. This was done using a data analysis pipeline and includes jupyter notebooks for cleaning and analyzing stock market data, which can be found in the pipeline folder. Also included in the repository is a press release for potential end users and license.

Name: Elaine Liu

Net ID: bpa2hu

DOI:

Press Release: 

Data:

Pipeline:

License: MIT License. 

## Problem Definition:

**General problem**: Forecasting stock prices

**Refined specific problem statement**:

Forecasting the future closing price of the S&P 500 Index using historical time series data. This will be done using Alpha Vantage stock market data, since they have open/high/low/close prices of the S&P 500 Index from 1997 to the present day. 

**Motivation**:

Being able to accurately forecast stock prices can be valuable information on when to buy and sell stocks. Looking into broad indices like S&P 500 can provide valuable insights into market trends and risk. Especially since there is a growing amount of financial data, we can apply time series analysis and data-driven methods to improve predictive performance.

**Rationale**:

The general problem of forecasting stock prices is broad and difficult to tackle, since it can be influenced by many external factors. To make it more approachable, I decided to focus on specifically the S&P 500 index, which features the top 500 US companies. In particular, we want to predict the closing price using historical time series data, which is much more measureable and realistic to implement. With this narrowed down scope, the broader problem of "forecasting stock prices" is much easier to tackle and more measurable.

**Headline**:

[Fortune Telling or Forecasting: Predicting Stock Prices Over Time]


## Domain Exposition:

| Term | Definition |
|----------|----------|
| Time Series | Data points collected in chronological order  |
| Forecasting | Predicting future values based on historical data   |
| Volatility | Degree of variation in price (e.g., high - low)   |
| Stationarity   | Property where statistical characteristics of a time series do not change over time   |
| Open Price   | Price at the start of the trading period   |
| High Price   | Highest price reached during the period   |
| Low Price   | Lowest price during the period   |
| Close Price   | Final price at the end of the period (target variable)   |

**Domain**:

This question revolves around the field of finance and machine learning, using time series analysis to analyze financial markets. We could categorize it under quantitative finance, wince we are using math and statsitical techniques to inform decision making in finance. The data we are using is stock market data in a document based database, which is stored in json format.

**Background reading**:

Link to UVA OneDrive [folder](https://myuva-my.sharepoint.com/:f:/g/personal/bpa2hu_virginia_edu/IgASqTjDrVcsTJPEdQM0KblxAeQdIeJmnHlEuYblQkY0_wo?e=Rdgyuo)

**Summary Table**:
| Title | Brief Description | Link |
|----------|----------|-------|
| Forecasting S&P 500 stocks    | Applies random forest, SVM, and LSTM to classify if a stock can return 2% more than its index     |[here](https://myuva-my.sharepoint.com/:u:/r/personal/bpa2hu_virginia_edu/Documents/databydesign_p2/Forecasting%20relative%20returns%20for%20S%26P%20500%20stocks%20using%20machine%20learning.url?csf=1&web=1&e=E1ERWN)|
| Stock Price trend Prediction in Vietnam    | Uses an LSTM model to analyze and forecast stock price movements in an emerging economy     |[here](https://myuva-my.sharepoint.com/:u:/r/personal/bpa2hu_virginia_edu/Documents/databydesign_p2/Applying%20machine%20learning%20algorithms%20to%20predict%20the%20stock%20price%20trend%20in%20the%20stock%20market%20%E2%80%93%20The%20case%20of%20Vietnam.url?csf=1&web=1&e=OgvPV9)|
| Predicting Closing Prices    | Uses a random forest and ANN to predict the next day closing price for 5 different companies    |[here](https://myuva-my.sharepoint.com/:u:/r/personal/bpa2hu_virginia_edu/Documents/databydesign_p2/Stock%20Closing%20Price%20Prediction%20using%20Machine%20Learning%20Techniques.url?csf=1&web=1&e=OhxYHh)|
| Comparing ML Methods    | Compares decision trees, random forest, SVM, and k means clustering on Tesla stock transactions     |[here](https://myuva-my.sharepoint.com/:u:/r/personal/bpa2hu_virginia_edu/Documents/databydesign_p2/A%20Comparative%20Study%20of%20Machine%20Learning%20Algorithms%20for%20Stock%20Price%20Prediction%20Using%20Insider%20Trading%20Data.url?csf=1&web=1&e=7PlFoF)|
| Overview and Tips| Provides background overview and tips for ML stock market predictions, as well as potential use cases    |[here](https://myuva-my.sharepoint.com/:u:/r/personal/bpa2hu_virginia_edu/Documents/databydesign_p2/www.itransition.com.url?csf=1&web=1&e=vmYdhH)|


## Data Creation

To get the raw data, I went to the Alpha Vantage documentation page, which is linked here https://www.alphavantage.co/documentation/. Scrolling down, we can see a section for the S&P 500 data, which has the weekly open, high, low, and close time series data for the S&P 500 index.

They have a link to the JSON output, which is linked here https://www.alphavantage.co/query?function=INDEX_DATA&symbol=SPX&interval=weekly&apikey=demo.  This directs the user to a page with all the data in JSON format. I copy pasted this data into a JSON file locally so that I could process it into documents to put into MongoDB.

Data Creation Table (with links to code): 

| Brief Description | Link |
| -------- | -------- |
| Flattened json file then loaded into MongoDB as individual documents | [link](https://github.com/elaineliu05/databydesign_p2/blob/main/json_to_mongo.py) |

**Bias Identification**:

There are multiple ways bias could have been introduced throughout the data collection process. First, this data is from Alpha Vantage, which aggregates market data and could include inconsistencies because of data vendor differences, delays in updates, etc. In addition, there is selection bias since this dataset only focuses on the S&P 500 index, which only contains large capital US companies. This dataset only has open, high, low, and close data, which is an example of feature bias, since we are ignoring other important factors in closing price, such as trading volume, economic environment, etc. Since the data is in weekly intervals, there is also a temporal bias since market conditions can be volatile over time, and using a weekly interval can potentially smooth out short-term volatility.

**Bias Mitigation**:

Although there are many sources of bias, there are a lot of methods we can use to mitigate it as well. For instance, to address the selection bias, we can specify that we are only predicting trends within large capital companies, not the entire market. We can mitigate feature bias by engineering variables from the data we have, such as moving averages, volatility by subtracting high low values, etc. Temporal bias can be addressed by using a time-based train-test split to better simulate actual forecasting. Some data preprocessing will also be necessary, such as normalization and scaling to reduce distortions caused by large price changes over time. We can reduce uncertainty by comparing multiple machine learning models and analyzing how sensitive predictions are to different training windows or hyperparameter combinations.

**Rationale**:

There were many important and critical decisions that needed to be made when compiling this dataset together. One major decision is to focus only on open/high/low/close data rather than incorporating external variables like economic indicators or sentiment data. This decision was made to simplify the problem and make it more measurable. However, this decision also limits the model’s ability to capture real-world drivers of market movement. Another important choice was the time granularity (monthly vs. weekly data). I chose to keep the intervals at a weekly resolution, rather than aggregating by month. This was to balance the trade-off between higher frequency data being able to capture more detail but also introducing more noise. In addition, the decision to use the closing price as the target variable is standard in financial modeling, but it assumes that end-of-period prices fully represent market behavior. 

## Metadata

**Implicit Schema Guidelines**:

Each document represents a single time step for the S&P 500 index. The required fields are symbol, name, interval, date, open, high, low, and close. Price fields are all stored as floats, not strings. All documents use the same field names and structures, and there are no nested arrays for the time series data. There are no records wth missing values - all incomplete records are excluded. These guidelines ensure consistency throughout the database while also preserving the flexibility of the document structure.

**Data Summary**

| Database Feature | Details |
|----------|----------|
| Data Source | Alpha Vantage |
| Data type | Time-series financial market data |
| Market Entity | S&P 500 |
| # of Documents | 1519 |
| Time interval | Weekly |
| Time Range | 1997 to 2026 |
| Features per Document | 8 |
| Target variable | closing price |

**Data Dictionary**

| Name | Data Type | Description | Example |
|----------|----------|----------|----------|
| symbol | string | Stock/index ticker symbol | SPX |
| name | string | Full name of the index | "S&P 500 INDEX" |
| interval | string | time frequency | weekly |
| date | datetime | Observation date | 2026-04-17 |
| open | float | Price at the start of the period| 6587.66 |
| high | float | Highest price during the period| 6811.18 |
| low | float | lowest price during the period | 6579.20 |
| close | float | Price at the end of the period| 6643.11 |

**Quantification of Uncertainty:**

There is limited uncertainty in open/high/low/close stock market data, since there is rarely missing data, and it is relatively straightforward to measure. For instance, financial data is recorded with finite precision and may vary slightly across data providers. However, the uncertainty present is mostly driven by inherent market volatility and external economic factors. The differences between data providers, such as Alpha Vantage, will likely have small measurement inconsistencies, which we can likely quantify as plus or minus 0.05. For instance, if a closing price is 6519.60, the uncertainty can be captured by an interval between 6519.55 and 6519.65.

