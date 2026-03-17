# Recipe-Prediction-Modeling

# Food.com Recipe Ratings: User vs Recipe Predictiveness

## Step 1: Introduction and Question

This project uses two files from the Food.com dataset: `RAW_recipes.csv` (recipe attributes like cook time and number of steps) and `interactions.csv` (user–recipe interactions including ratings and dates). After merging, each row represents a specific user interacting with a specific recipe.

**Project question:** Are recipe ratings more predictably determined by **who** is rating (user identity) or by **what** is being rated (recipe identity/qualities)?

This question matters because if user identity dominates, ratings are driven largely by individual rating behavior (generous vs. strict raters). If recipe identity dominates, ratings reflect stable recipe “quality” more than the rater.

---

## Step 2: Data Cleaning and Exploratory Data Analysis

### Step 2.1 Data Cleaning

I merged recipe-level data with interaction-level data using a left join on recipe ID (`id` in `RAW_recipes.csv` and `recipe_id` in `interactions.csv`). This creates one row per (user, recipe) interaction and repeats recipe attributes across multiple ratings.

In `interactions.csv`, a rating value of `0` does not represent a real 1–5 rating, so I replaced `rating == 0` with `NaN` to treat it as missing. I also parsed the `date` column into a datetime format.

### Step 2.2 Univariate Analysis

#### Distribution of Ratings (1–5)
Ratings are heavily concentrated near 5, indicating strong positive skew and suggesting that users tend to leave ratings when they like a recipe.

<iframe
  src="assets/rating_dist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Distribution of Cook Time (minutes)
Cook times are extremely right-skewed with a long tail of very large values, so a log count scale helps reveal the distribution of typical recipes.

<iframe
  src="assets/minutes_dist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Step 2.3 Bivariate Analysis

#### Cook Time vs Rating
This plot compares cook time across different rating values. Using a log scale helps reveal whether typical cook times differ meaningfully between lower- and higher-rated recipes despite extreme outliers.

<iframe
  src="assets/minutes_vs_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### User vs Rating (User Mean Rating vs Number of Ratings)
Users with few ratings show highly variable mean ratings, while users with many ratings have mean ratings that cluster near the high end, suggesting stable user-specific rating tendencies and overall positivity bias.

<iframe
  src="assets/user_mean_vs_count.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Step 2.4 Interesting Aggregates

To better understand how ratings behave at different levels of aggregation, I computed recipe-level statistics. The table below shows recipes with the largest number of ratings (more reliable estimates of typical rating).

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>id</th>
      <th>average_rating</th>
      <th>num_ratings</th>
      <th>rating_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>349246</td>
      <td>4.24</td>
      <td>295</td>
      <td>1.25</td>
    </tr>
    <tr>
      <td>486261</td>
      <td>4.99</td>
      <td>217</td>
      <td>0.10</td>
    </tr>
    <tr>
      <td>302120</td>
      <td>4.76</td>
...
      <td>0.53</td>
    </tr>
  </tbody>
</table>

---

## Step 3: Assessment of Missingness

### Step 3.1 MNAR Analysis

I believe the `rating` column is plausibly **MNAR** (Missing Not At Random). Users may be more likely to leave a rating when they had a particularly positive (or negative) experience, so the probability that a rating is missing can depend on the unobserved rating value itself.

Additional behavioral data (e.g., whether the user cooked the recipe, how long they viewed the page, whether they bookmarked it, or whether they left a written review) could help explain why ratings are missing and potentially make the missingness closer to MAR.

### Step 3.2 Missingness Dependency (Permutation Tests)

I created an indicator `rating_missing` equal to True when `rating` is missing. I used a permutation test with the following test statistic:

**Test statistic:** absolute difference in means  
T = | mean(X when rating_missing is True) − mean(X when rating_missing is False) |

Results:
- Missingness of `rating` **depends on** `n_steps` (p-value ≈ 0.0, i.e., < 1/B).
- Missingness of `rating` **does not depend on** `minutes` (p-value = 0.13).

The plot below shows the distribution of `n_steps` split by whether `rating` is missing.

<iframe
  src="assets/missingness_nsteps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

## Step 4: Hypothesis Testing (Not Missingness)

### Is user identity more predictive than recipe identity?

I compared two identity-only baseline predictors for rating:
- **User-only predictor:** predict a rating using the user’s mean training-set rating.
- **Recipe-only predictor:** predict a rating using the recipe’s mean training-set rating.
- For unseen users/recipes in the test set, I used the global mean rating from the training set.

**Null hypothesis (H0):** User identity is not more predictive than recipe identity (MAE_user − MAE_recipe = 0).  
**Alternative hypothesis (H1):** User identity is more predictive than recipe identity (MAE_user − MAE_recipe < 0).

**Test statistic:**  
T = MAE_user − MAE_recipe (computed on the same held-out test set)

**Result:** T_obs = −0.0164 and p ≈ 0.0 (p < 1/B), so I reject H0 at alpha = 0.05. This provides strong evidence that user identity is more predictive of ratings than recipe identity under these identity-only baseline predictors.

---

## Step 5: Framing a Prediction Problem

**Prediction task:** Predict the numeric `rating` (1–5) that a user will give to a recipe.  
**Type:** Regression  
**Metric:** MAE (mean absolute error), because it is interpretable in “rating points” (e.g., MAE = 0.5 means the model is off by half a star on average).

**Time of prediction:** I assume the user is logged in (`user_id` known) and recipe attributes (e.g., cook time and steps) are known once the recipe is selected. I avoid leakage features such as average ratings computed from the full dataset.

---

## Step 6: Baseline Model

**Model:** Ridge regression in an sklearn Pipeline.

**Features:**
- `user_id` (nominal; OneHot encoded)
- `minutes` (quantitative)
- `n_steps` (quantitative)

**Baseline performance:** test MAE = 0.4447, meaning predictions are off by about 0.44 rating points on average.

---

## Step 7: Final Model

### Planned hyperparameter tuning
I tuned Ridge’s regularization strength `alpha`. Because `user_id` is one-hot encoded, the model has a high-dimensional feature space and can overfit; regularization controls how much coefficients are shrunk and can improve generalization.

### Feature engineering
In addition to the baseline features, I added engineered recipe features:
- `log_minutes = log(1 + minutes)` to reduce the impact of extreme cook-time outliers
- `steps_per_minute = n_steps/(minutes+1)` and `minutes_per_step = minutes/(n_steps+1)` to capture process intensity/complexity

I used GridSearchCV (5-fold CV) to select the best `alpha` based on MAE.

**Final performance:** test MAE = 0.4415, improving over the baseline MAE of 0.4447 on the same test set.

---

## Step 8: Fairness Analysis

I tested whether the final model performs worse for users with little historical rating data (a cold-start concern).

**Groups (defined using training data only):**
- **Group X:** cold-start users (<= 5 ratings in the training set)
- **Group Y:** active users (>= 50 ratings in the training set)

**Metric:** MAE.

**Null hypothesis (H0):** MAE is the same across groups (any difference is due to chance).  
**Alternative hypothesis (H1):** The model performs worse for Group X (MAE_X > MAE_Y).

**Test statistic:** T = MAE_X − MAE_Y

**Result:** MAE_X = 0.6349, MAE_Y = 0.3206, so T_obs = 0.3143. The permutation test gave p ≈ 0.0 (p < 1/B), so I reject H0. This provides strong evidence that the model performs substantially worse for cold-start users than for active users.