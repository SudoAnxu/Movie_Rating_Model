
# Movie Rating Prediction
## Introduction
This project aims to predict movie ratings based on various attributes such as genre, director, actors, and more. The goal is to develop a model that can accurately forecast how well a movie will be received by audiences.

## Approach
Our approach involves several key steps:
1. **Data Preprocessing**: Clean and preprocess the dataset to ensure it's ready for analysis.
2. **Feature Engineering**: Create new features that capture important aspects of movies, such as genre popularity and actor popularity.
3. **Model Selection**: Choose a suitable model for prediction, considering factors like complexity and interpretability.
4. **Model Evaluation**: Assess the performance of the model using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

## Data Preprocessing
### Ensure Rating is Numeric
To perform calculations, we need to ensure that the 'Rating' column is numeric.

```
# Ensure 'Rating' is numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
```

### Thinking:
- **Why Numeric?**: Ratings must be numeric to be used in mathematical operations and statistical analyses.
- **Error Handling**: Using `errors='coerce'` converts non-numeric values to NaN, which can be handled later.

## Feature Engineering
Feature engineering is crucial for capturing meaningful relationships between variables.

### Genre Popularity
Calculate the average rating for each genre to understand its impact on movie ratings.

```
#### Step 1: Split Genres into Individual Rows
Split the 'Genre' column into individual rows to calculate the average rating for each genre.

```python
genres_df = df.assign(Genre=df['Genre'].str.split(',')).explode('Genre')
```

#### Step 2: Calculate Average Rating for Each Genre
Calculate the average rating for each genre.

```python
genre_popularity = genres_df.groupby('Genre')['Rating'].mean().reset_index()
genre_popularity.columns = ['Genre', 'GenrePopularity']
```

#### Step 3: Map Genre Popularity Back to the Original Dataset
Map the genre popularity back to the original dataset by calculating the average popularity for each movie's genres.

```python
def map_genre_popularity(genres):
    # Split genres and calculate average popularity
    genres_list = genres.split(',')
    # Remove leading/trailing spaces from each genre
    genres_list = [genre.strip() for genre in genres_list]
    # Filter genre popularity for these genres
    relevant_popularity = genre_popularity.loc[genre_popularity['Genre'].isin(genres_list)]

    # If no genres match, return NaN
    if relevant_popularity.empty:
        return None

    # Calculate average popularity
    avg_popularity = relevant_popularity['GenrePopularity'].mean()
    return avg_popularity

# Apply the function to each row
df['GenrePopularity'] = df['Genre'].apply(map_genre_popularity)
```

### Thinking:
- **Genre Impact**: Different genres appeal to different audiences, affecting ratings.
- **Average Popularity**: Captures the overall appeal of a genre across movies.

### Actor Popularity
Calculate the average rating for each actor to understand their impact on movie ratings.


#### Step 1: Create a DataFrame with Actor-Rating Pairs
Create a DataFrame with actor-rating pairs to calculate the average rating for each actor.

```
actor_ratings = pd.concat([
    df[['Actor 1', 'Rating']].rename(columns={'Actor 1': 'Actor'}),
    df[['Actor 2', 'Rating']].rename(columns={'Actor 2': 'Actor'}),
    df[['Actor 3', 'Rating']].rename(columns={'Actor 3': 'Actor'})
], ignore_index=True)

# Remove rows with missing actor names
actor_ratings = actor_ratings.dropna(subset=['Actor'])
```

#### Step 2: Calculate Actor Popularity
Calculate the average rating for each actor.

```
actor_popularity = actor_ratings.groupby('Actor')['Rating'].mean().reset_index()
actor_popularity.columns = ['Actor', 'ActorPopularity']
```

#### Step 3: Merge with Original Dataset
Merge the actor popularity back into the original dataset for each actor column.

```
for actor_col in ['Actor 1', 'Actor 2', 'Actor 3']:
    actor_popularity_map = actor_popularity.set_index('Actor')['ActorPopularity']
    df[f'{actor_col}Popularity'] = df[actor_col].map(actor_popularity_map)
```

### Thinking:
- **Actor Impact**: Actors can significantly influence a movie's success.
- **Average Rating**: Captures an actor's overall performance across films.

### Yearly Trend
Categorize movies into different year groups to capture trends over time.


#### Step 1: Convert Year to Numeric
Ensure the 'Year' column is numeric.

```python
if df['Year'].dtype != 'int64':
    df['Year'] = df['Year'].str.replace('(', '').str.replace(')', '').astype(int)
```

#### Step 2: Create Year Groups
Create year groups using `pd.cut`.

```python
df['YearGroup'] = pd.cut(df['Year'], bins=[1950, 2000, 2010, 2020], labels=['Pre-2000', '2000-2010', 'Post-2010'], right=False)
```

#### Step 3: One-Hot Encode Year Groups
One-hot encode the year groups.

```python
year_groups = pd.get_dummies(df['YearGroup'], prefix='YearGroup').astype(int)
```

#### Step 4: Merge with Original Dataset
Merge the one-hot encoded year groups into the original dataset.

```python
df = pd.concat([df, year_groups], axis=1)
df = df.drop(columns=['YearGroup'])
```

### Thinking:
- **Time Trends**: Movie preferences and production quality can change over time.
- **Categorization**: Helps capture broad trends without overfitting to specific years.

### Duration Impact
Categorize movies by duration to capture its impact on ratings.

#### Step 1: Convert Duration to Numeric
Convert the 'Duration' column to numeric.

```
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(int)
```

#### Step 2: Create Duration Groups
Create duration groups using `pd.cut`.

```
df['DurationGroup'] = pd.cut(df['Duration'], bins=[0, 90, 120, float('inf')], labels=['Short', 'Medium', 'Long'])
```

#### Step 3: One-Hot Encode Duration Groups
One-hot encode the duration groups.

```
duration_groups = pd.get_dummies(df['DurationGroup']).astype(int)
```

#### Step 4: Merge with Original Dataset
Merge the one-hot encoded duration groups into the original dataset.

```
df = pd.concat([df, duration_groups], axis=1)
df = df.drop(columns=['DurationGroup'])
```

### Thinking:
- **Duration Effect**: Movie length can influence audience engagement and ratings.
- **Categorization**: Simplifies the impact of duration into manageable categories.

### Vote Count Impact
Categorize movies by vote count to capture its impact on ratings.


#### Step 1: Convert Votes to Numeric
Convert the 'Votes' column to numeric.

```python
df['Votes'] = df['Votes'].str.replace(',', '').str.extract('(\d+)').astype(float)
```

#### Step 2: Create Vote Groups
Create vote groups using `pd.cut`.

```python
df['VoteGroup'] = pd.cut(df['Votes'], bins=[0, 100, 1000, float('inf')], labels=['Low', 'Medium', 'High'])
```

#### Step 3: One-Hot Encode Vote Groups
One-hot encode the vote groups.

```python
vote_groups = pd.get_dummies(df['VoteGroup']).astype(int)
```

#### Step 4: Merge with Original Dataset
Merge the one-hot encoded vote groups into the original dataset.

```python
df = pd.concat([df, vote_groups], axis=1)
df = df.drop(columns=['VoteGroup'])
```

### Thinking:
- **Vote Impact**: Higher vote counts often correlate with more popular movies.
- **Categorization**: Helps distinguish between low, medium, and high engagement levels.

### Director Popularity
Calculate the average rating for each director to capture their impact on movie ratings.

#### Step 1: Calculate Director Popularity
Calculate the average rating for each director.

```
director_popularity = df.groupby('Director')['Rating'].mean().reset_index()
director_popularity.columns = ['Director', 'DirectorPopularity']
```

#### Step 2: Merge with Original Dataset
Merge the director popularity into the original dataset.

```
df = pd.merge(df, director_popularity, on='Director')
```

### Thinking:
- **Director Impact**: Directors can significantly influence a movie's quality and reception.
- **Average Rating**: Captures a director's overall performance across films.

## Model Creation and Evaluation
### Model Selection and Training
Train a Random Forest Regressor model to predict movie ratings.


#### Step 1: Split Data into Training and Testing Sets
Split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X = df.drop(['Name', 'Rating'], axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Step 2: Train the Model
Train the Random Forest Regressor model.

```python
from sklearn.ensemble import RandomForestRegressor

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

### Thinking:
- **Model Choice**: Random Forest is robust and handles complex interactions well.
- **Training**: Uses a subset of the data to learn patterns.

### Model Evaluation
Evaluate the model using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).


#### Step 1: Make Predictions on the Test Set
Make predictions on the test set.

```
# Make predictions on the test set
```
y_pred = model.predict(X_test)
```

#### Step 2: Evaluate Model Performance
Evaluate the model's performance using MSE, MAE, and R².

```
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
# Evaluate model performance
```
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}, MAE: {mae}, R²: {r2}")
```

### Thinking:
- **Evaluation Metrics**: MSE and MAE measure error, while R² assesses goodness of fit.
- **Performance**: Helps determine if the model is accurate and reliable.

