# Hyderabad House Price Predictor

## Overview

This project is a machine learning-powered web application that predicts house rental prices in Hyderabad, India. Built with Streamlit, it provides an interactive interface for users to input property characteristics (bedrooms, bathrooms, area, locality, furnishing status) and receive price predictions based on a Random Forest regression model. The application includes data visualization capabilities, model performance metrics, and detailed explanations of predictions to help users understand the factors affecting rental prices.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

**Frontend Framework**: Streamlit-based web application providing an interactive user interface with sidebar controls for property inputs and main area for predictions and visualizations.

**Data Processing Pipeline**: Modular data processing system with dedicated `DataProcessor` class that handles CSV data loading, cleaning, and preprocessing. The processor handles price formatting, area normalization, categorical variable encoding, and data validation with robust error handling for missing or malformed data.

**Machine Learning Model**: Random Forest Regression model implemented through the `ModelTrainer` class, featuring automated feature encoding, train-test splitting, cross-validation, and comprehensive model evaluation metrics (MAE, RMSE, R²). The model uses property characteristics as features to predict rental prices.

**State Management**: Streamlit session state management to persist model training status, trained model instance, processed data, and performance metrics across user interactions, ensuring efficient resource utilization.

**Utility Layer**: Helper functions for currency formatting in Indian Rupees, prediction explanations with market context, and input validation to ensure data quality and user experience.

**File Structure**: Organized into separate modules (app.py for main interface, data_processor.py for data handling, model_trainer.py for ML operations, utils.py for utilities) promoting code maintainability and separation of concerns.

**Data Storage**: CSV-based data storage with the dataset located in `attached_assets/` directory, containing Hyderabad housing data with features like price, area, bedrooms, bathrooms, locality, and furnishing status.

## External Dependencies

**Core Libraries**: 
- Streamlit for web interface and user interaction
- Pandas and NumPy for data manipulation and numerical operations
- Scikit-learn for machine learning algorithms, preprocessing, and model evaluation

**Visualization**: 
- Plotly Express and Plotly Graph Objects for interactive data visualizations and charts

**Model Persistence**: 
- Joblib for saving and loading trained machine learning models

**Data Requirements**: 
- CSV dataset (`Hyderabad_House_Data_1755501086395.csv`) containing property listings with price, location, and property characteristic data

The application operates as a self-contained system without external API dependencies, using local data processing and model training capabilities.