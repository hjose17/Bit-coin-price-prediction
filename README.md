# Bit-coin-price-prediction
Here's an expanded `README.md` that includes detailed descriptions of major modules, including the integration of Twitter sentiment analysis for Bitcoin price prediction.

---

# Bitcoin Price Prediction with Twitter Sentiment Analysis

This project focuses on predicting Bitcoin prices by leveraging machine learning techniques combined with Twitter sentiment analysis. The goal is to analyze the impact of public sentiment on Bitcoin price movements and create a predictive model using this additional feature. The notebook (`bitcoin.ipynb`) contains the entire workflow, including data collection, processing, model training, and evaluation.

## Project Overview

Cryptocurrency prices, especially Bitcoin, are often influenced by public sentiment, news, and other social signals. By combining historical price data with sentiment analysis from Twitter, this project aims to enhance the predictive power of the model, potentially improving its accuracy in forecasting future Bitcoin prices.

## Modules

The notebook consists of the following key modules:

1. **Data Collection**:
   - **Price Data**: Historical Bitcoin price data is collected from a reliable financial data source. This data includes open, high, low, close prices, and trading volumes for each time interval.
   - **Twitter Sentiment Data**: Tweets mentioning "Bitcoin" are collected using the Twitter API. These tweets are then processed to analyze sentiment, providing an additional feature (positive, negative, or neutral sentiment) that could impact Bitcoin prices.

2. **Data Preprocessing**:
   - **Cleaning and Filtering**: This step involves cleaning the price and sentiment data. Missing values are handled, duplicate records are removed, and unnecessary data points are filtered out.
   - **Sentiment Analysis**: Text preprocessing is performed on tweets to remove unnecessary characters, stop words, and other noise. A sentiment analysis tool (e.g., VADER or TextBlob) is used to quantify the sentiment of each tweet.
   - **Feature Engineering**: The sentiment scores are aggregated over time intervals matching the Bitcoin price data to create a new feature, `sentiment_score`, representing the average sentiment within each period.

3. **Exploratory Data Analysis (EDA)**:
   - Visualizations are created to explore relationships between Bitcoin price and sentiment data.
   - Trends in historical prices, tweet volumes, and sentiment scores over time are examined to check for correlations and potential predictive indicators.

4. **Model Building**:
   - The dataset is split into training and test sets.
   - Different machine learning models are trained to predict Bitcoin prices, using features such as historical price data, trading volume, and the new sentiment feature.
   - Models include:
      - **Linear Regression**: To capture simple linear relationships.
      - **Random Forest Regressor**: For a more flexible, non-linear approach.
      - **LSTM (Long Short-Term Memory)**: A deep learning model tailored for sequential data, suitable for time series forecasting.

5. **Model Training and Hyperparameter Tuning**:
   - Each model’s hyperparameters are tuned using techniques like cross-validation to identify the best configuration for predicting Bitcoin prices.
   - The training process involves experimenting with different combinations of features, including and excluding the `sentiment_score`, to evaluate the impact of sentiment on model performance.

6. **Model Evaluation**:
   - Each model is evaluated using metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to quantify the accuracy of predictions.
   - A comparison is made between models with and without sentiment data to demonstrate the effectiveness of using Twitter sentiment as an input feature.

7. **Results and Visualization**:
   - Graphs and plots are used to visualize the model’s predictions versus actual Bitcoin prices.
   - The impact of sentiment on prediction accuracy is highlighted, showcasing how Twitter sentiment influences price trends.

## Dataset

The project uses two primary datasets:
1. **Bitcoin Price Data**: Contains historical Bitcoin price and trading volume data.
2. **Twitter Sentiment Data**: Includes tweets mentioning Bitcoin, processed for sentiment analysis.

## Getting Started

### Prerequisites

To run this project, you'll need to have Python installed along with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` (for LSTM model)
- `vaderSentiment` or `TextBlob` (for sentiment analysis)
- `tweepy` (for Twitter API access)

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow vaderSentiment tweepy
```

### Setting Up Twitter API

To collect Twitter data, you’ll need a Twitter Developer account. Create a Twitter API key and add it to the notebook to access and collect tweets mentioning Bitcoin.

### Running the Notebook

1. Clone this repository.
2. Open `bitcoin.ipynb` in Jupyter Notebook or Jupyter Lab.
3. Run each cell in order to load the data, preprocess it, and train the model.

## Project Structure

- **bitcoin.ipynb**: Main notebook containing the code for data collection, preprocessing, modeling, and evaluation.
- **data/**: (Optional) Folder for storing raw and processed datasets.
- **models/**: (Optional) Folder for saving trained models for future use.
- **utils/**: (Optional) Folder for utility functions like Twitter data collection and sentiment analysis.

## Results

The model outputs predictions for Bitcoin prices, and evaluation metrics assess its accuracy. Plots are generated to visualize predictions, showing the effect of Twitter sentiment on price trends.

## Conclusion

This project demonstrates the influence of public sentiment on Bitcoin prices and highlights how incorporating social media sentiment can enhance price prediction models.

