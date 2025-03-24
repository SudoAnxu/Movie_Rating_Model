# Movie Rating Prediction

This repository contains a project focused on predicting movie ratings based on various attributes such as genre, director, actors, and more. The goal is to develop a robust model that can accurately forecast how well a movie will be received by audiences.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project involves data preprocessing, feature engineering, model selection, and evaluation. It uses a Random Forest Regressor to predict movie ratings.

## Features
- **Genre Popularity**: Calculates the average rating for each genre.
- **Actor Popularity**: Calculates the average rating for each actor.
- **Yearly Trend**: Categorizes movies into different year groups.
- **Duration Impact**: Categorizes movies by duration.
- **Vote Count Impact**: Categorizes movies by vote count.
- **Director Popularity**: Calculates the average rating for each director.

## Requirements
- **Python 3.x**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Jupyter Notebook**

To install the required packages, run:
```
pip install pandas numpy scikit-learn jupyter
```

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/SudoAnxu/Movie_Rating_Model.git
   ```
2. Navigate into the repository:
   ```
   cd Movie_Rating_Model
   ```
3. Open the Jupyter Notebook:
   ```
   jupyter notebook
   ```
4. Run the notebook to train and evaluate the model.

## Model Performance
The model's performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

## Contributing
Contributions are welcome. Please submit a pull request with your changes.

## License
This project is licensed under the apache 2.0 License. See [LICENSE](LICENSE) for details.
