# Food.com Recipe Rating Prediction

**Author: Josh Hu**

## Overview

This project analyzes Food.com recipe ratings to answer a central question:

**Are ratings driven more by the recipe itself, or by the individual user giving the rating?**

Using more than 730,000 user-recipe interactions, I explored rating behavior, tested whether user identity or recipe identity had greater predictive value, and built a regression model to predict future ratings.

The analysis found that **user identity was significantly more predictive of ratings than recipe identity**, suggesting that individual users have persistent rating tendencies. I also found a substantial **cold-start problem**: prediction error was nearly twice as high for users with little historical rating data compared with highly active users.

---

## Dataset

The project uses two Food.com datasets:

- `RAW_recipes.csv`, containing recipe-level information such as cook time, ingredients, number of steps, and recipe descriptions
- `interactions.csv`, containing user-recipe interactions including ratings, review dates, and user IDs

After merging the datasets on recipe ID, each row represents a user's interaction with a specific recipe.

The recipe dataset contains **83,782 recipes**, while the interactions dataset contains **731,927 user-recipe interactions**.

### Recipe Data

| Column | Description |
| :--- | :--- |
| `name` | Recipe name |
| `id` | Recipe ID |
| `minutes` | Preparation time |
| `contributor_id` | User who submitted the recipe |
| `submitted` | Submission date |
| `tags` | Food.com recipe tags |
| `nutrition` | Nutrition information |
| `n_steps` | Number of preparation steps |
| `steps` | Recipe instructions |
| `description` | User-provided description |
| `ingredients` | Recipe ingredients |
| `n_ingredients` | Number of ingredients |

### Interaction Data

| Column | Description |
| :--- | :--- |
| `user_id` | User ID |
| `recipe_id` | Recipe ID |
| `date` | Date of interaction |
| `rating` | Rating given |
| `review` | Review text |

---

## Data Preparation

I merged the recipe and interaction datasets using recipe ID, producing one row per user-recipe interaction.

Ratings of `0` in the interaction data do not represent valid values on the 1–5 rating scale, so I treated them as missing values. I also converted interaction dates to datetime format and retained the variables most relevant to the analysis.

---

## Exploratory Analysis

### Rating Distribution

<iframe
  src="assets/rating_dist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Ratings are strongly concentrated near 5, showing that Food.com users tend to give highly positive ratings. This creates an important modeling challenge because the target distribution is heavily skewed toward the upper end of the rating scale.

### Cook Time Distribution

<iframe
  src="assets/minutes_dist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Recipe preparation times are heavily right-skewed because a small number of recipes contain extremely large cook-time values. The visualization caps extreme values to make the typical distribution easier to interpret.

### Cook Time vs. Rating

<iframe
  src="assets/minutes_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

There is no strong relationship between cook time and rating. Recipes receiving high and low ratings exhibit substantial overlap in preparation time, suggesting that cook time alone provides limited predictive information.

### User Rating Behavior

<iframe
  src="assets/user_mean_vs_count.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Users with few ratings display highly variable average scores, while users with many historical ratings tend to stabilize around consistent personal rating patterns. This motivated a deeper investigation into whether **who is rating** may matter more than **what is being rated**.

---

## Missing Data Analysis

Approximately 6% of interaction records contain a missing rating after converting `0` values to missing data.

The missingness of `rating` is plausibly **Missing Not At Random (MNAR)** because a user's decision to submit a rating may depend on their unobserved experience with the recipe.

I also tested whether rating missingness was associated with observable recipe characteristics using permutation tests.

The results showed:

- Rating missingness was significantly associated with `n_steps`
- There was not sufficient evidence that rating missingness depended on `minutes`

<iframe
  src="assets/missingness_nsteps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

These results suggest that missing ratings may not occur completely at random and should be considered when interpreting downstream results.

---

## User vs. Recipe Predictive Analysis

To determine whether ratings were driven more strongly by users or recipes, I compared two simple baseline predictors:

- **User-only model:** Predict each rating using that user's average rating from the training data
- **Recipe-only model:** Predict each rating using that recipe's average rating from the training data

For previously unseen users or recipes, I used the overall training-set mean rating.

I evaluated both predictors using **Mean Absolute Error (MAE)** on the same held-out test set.

The observed difference was:

`MAE_user - MAE_recipe = -0.0164`

I then performed a paired permutation test by randomly swapping the two models' per-interaction errors.

The resulting p-value was effectively zero (`p < 1/B`), providing strong evidence that the user-only predictor performs better than the recipe-only predictor.

### Key Finding

**User identity is significantly more predictive of recipe ratings than recipe identity.**

This suggests that ratings reflect persistent differences in how individual users evaluate recipes—for example, some users may consistently rate more generously or critically than others.

---

## Rating Prediction Model

I then built a regression model to predict the numeric rating a user would assign to a recipe.

### Baseline Model

The baseline model used Ridge regression with the following features:

- `user_id`, one-hot encoded
- `minutes`
- `n_steps`

The baseline model achieved:

**MAE = 0.4447**

This means predictions were off by approximately 0.44 rating points on average.

---

## Feature Engineering and Model Tuning

To improve the model, I added several engineered recipe features:

- `log_minutes = log(1 + minutes)`
- `steps_per_minute = n_steps / (minutes + 1)`
- `minutes_per_step = minutes / (n_steps + 1)`

These features were designed to capture recipe complexity while reducing the influence of extreme cook-time outliers.

I used a scikit-learn pipeline with:

- One-hot encoding for `user_id`
- Median imputation
- Standardization of numerical features
- Ridge regression
- 5-fold cross-validation
- `GridSearchCV` for regularization tuning

The best-performing regularization parameter was:

`alpha = 0.1`

The final model achieved:

**MAE = 0.4415**

The improvement over the baseline was modest, suggesting that the engineered recipe features provide some additional predictive value, but much of the model's performance is still driven by user-specific information.

---

## Cold-Start Analysis

Because the model relies heavily on user identity, I investigated whether performance changes depending on how much historical data is available for a user.

I defined two groups using only the training data:

- **Cold-start users:** 5 or fewer historical ratings
- **Active users:** 50 or more historical ratings

The final model produced:

| User Group | MAE |
| :--- | ---: |
| Cold-start users | 0.6348 |
| Active users | 0.3206 |

The difference in MAE was:

`0.6348 - 0.3206 = 0.3143`

A permutation test produced `p < 1/B`, providing strong evidence that model performance differs between the two groups.

### Key Finding

The model's error for cold-start users is nearly **twice as high** as its error for active users.

This is consistent with the earlier finding that user identity is highly predictive. When a user has substantial historical data, the model can learn that person's typical rating behavior. For new or infrequent users, much less information is available, which leads to substantially worse predictions.

---

## Conclusions

This analysis produced three main findings:

1. **User identity is more predictive of ratings than recipe identity.** Individual rating behavior appears to play a major role in determining observed Food.com ratings.

2. **A Ridge regression model can predict ratings with an MAE of approximately 0.44.** Feature engineering produced a small improvement over the baseline model.

3. **The model has a substantial cold-start problem.** Prediction error for users with little rating history is nearly twice as large as the error for active users.

Together, these results suggest that recommendation and rating-prediction systems should account for differences in individual user behavior while also developing strategies for users with limited historical data.
```
