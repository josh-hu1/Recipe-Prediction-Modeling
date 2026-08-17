# Food.com Recipe Rating Prediction

A data science project analyzing **731,927 Food.com user-recipe interactions** to investigate whether ratings are driven more by the recipe itself or by the individual user giving the rating.

## Key Findings

- **User identity was significantly more predictive of ratings than recipe identity**, suggesting that users have persistent rating tendencies.
- A tuned **Ridge regression model** achieved a test MAE of **0.4415**.
- The model exhibited a substantial **cold-start problem**:
  - Cold-start users: **0.6348 MAE**
  - Active users: **0.3206 MAE**
- Prediction error for users with little historical data was therefore nearly **twice as high** as for highly active users.

## Research Question

**Are recipe ratings more predictably determined by who is rating or by what is being rated?**

The analysis compares user-based and recipe-based predictive performance to determine whether observed ratings are influenced more strongly by persistent user behavior or by differences between individual recipes.

The results provide strong evidence that **user identity carries more predictive information than recipe identity**.

## Dataset

The project uses two Food.com datasets:

- `RAW_recipes.csv` — recipe-level information including preparation time, ingredients, number of steps, and descriptions
- `interactions.csv` — user-recipe interactions including ratings, review dates, and user IDs

The datasets contain:

- **83,782 recipes**
- **731,927 user-recipe interactions**

After merging on recipe ID, each row represents a user's interaction with a specific recipe.

## Methods

The project includes:

- Data cleaning and preprocessing
- Exploratory data analysis
- Missing-data analysis
- Permutation testing
- User-vs.-recipe predictive comparison
- Feature engineering
- Ridge regression
- One-hot encoding
- 5-fold cross-validation
- GridSearchCV
- Cold-start performance analysis

## Model Development

### Baseline Model

The baseline model uses Ridge regression with:

- `user_id`, one-hot encoded
- `minutes`
- `n_steps`

The baseline model achieved:

**MAE = 0.4447**

### Feature Engineering

The final model adds:

- `log_minutes`
- `steps_per_minute`
- `minutes_per_step`

These features were designed to capture recipe preparation complexity while reducing the influence of extreme cook-time values.

The model was tuned using **5-fold cross-validation** and `GridSearchCV`.

Best regularization parameter:

```text
alpha = 0.1
````

Final model performance:

**MAE = 0.4415**

The improvement over the baseline is modest, suggesting that recipe-level features provide some additional predictive signal while user-specific information remains a major source of predictive power.

## User vs. Recipe Predictive Analysis

Two simple predictors were compared:

* **User-only predictor:** predicts ratings using the user's average training-set rating
* **Recipe-only predictor:** predicts ratings using the recipe's average training-set rating

Both were evaluated on the same held-out test set using Mean Absolute Error.

The observed difference was:

```text
MAE_user - MAE_recipe = -0.0164
```

A paired permutation test produced:

```text
p-value < 0.0002
```

This provides strong evidence that the user-only predictor performs better than the recipe-only predictor.

## Cold-Start Analysis

Because user identity was highly predictive, I tested whether model performance changes based on the amount of historical information available for each user.

Users were divided into:

* **Cold-start users:** 5 or fewer historical ratings
* **Active users:** 50 or more historical ratings

Results:

| User Group       |    MAE |
| :--------------- | -----: |
| Cold-start users | 0.6348 |
| Active users     | 0.3206 |

The difference in prediction error was:

```text
0.6348 - 0.3206 = 0.3143
```

A permutation test produced:

```text
p-value < 0.0002
```

The model's error for cold-start users is therefore nearly **twice as high** as its error for active users.

This is consistent with the earlier finding that user identity is highly predictive: when a user has substantial historical data, the model can better learn their typical rating behavior.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Plotly
* Jupyter

## Project Structure

```text
Recipe-Prediction-Modeling/
├── analysis.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── modeling.py
│   └── statistical_tests.py
├── assets/
├── index.md
└── _config.yml
```

## Code Structure

Reusable modeling and statistical-testing logic is separated from the notebook:

### `src/modeling.py`

Contains the custom `RecipeFeatureEngineer` transformer used by the final scikit-learn pipeline.

### `src/statistical_tests.py`

Contains reusable functions for:

* missingness permutation testing
* paired permutation testing of model errors

## Full Analysis

The complete analysis is available in:

[`analysis.ipynb`](analysis.ipynb)

The notebook covers:

1. Data preparation and exploratory analysis
2. Missing-data analysis
3. User vs. recipe predictive analysis
4. Baseline regression modeling
5. Feature engineering and hyperparameter tuning
6. Cold-start analysis
7. Conclusions

## Reproducing the Project

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then launch the notebook:

```bash
jupyter notebook analysis.ipynb
```

The Food.com datasets are not included in this repository.

## Conclusions

This project produced three primary findings:

1. **User identity is more predictive of ratings than recipe identity.**
2. **A regularized regression model predicts ratings with an MAE of approximately 0.44.**
3. **Prediction accuracy decreases substantially for users with limited historical data.**

Together, these results suggest that rating-prediction systems should account for persistent differences in user behavior while also addressing the challenges associated with new or infrequent users.