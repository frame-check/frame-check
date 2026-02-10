from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import (  # type: ignore [import-not-found]
    MinMaxScaler,
    StandardScaler,
)

# Initialize scalers
scaler = MinMaxScaler()
std_scaler = StandardScaler()

# Initialize a comprehensive DataFrame with multiple columns
df = pd.DataFrame(
    {
        "customer_id": range(1, 10001),
        "first_name": [f"FirstName_{i}" for i in range(1, 10001)],
        "last_name": [f"LastName_{i}" for i in range(1, 10001)],
        "email": [f"user{i}@email.com" for i in range(1, 10001)],
        "age": np.random.randint(18, 80, 10000),
        "gender": np.random.choice(["M", "F", "Other"], 10000),
        "registration_date": pd.date_range("2020-01-01", periods=10000, freq="H"),
        "city": np.random.choice(
            [
                "New York",
                "Los Angeles",
                "Chicago",
                "Houston",
                "Phoenix",
                "Philadelphia",
                "Miami",
                "Boston",
                "Seattle",
                "Denver",
            ],
            10000,
        ),
        "state": np.random.choice(
            ["NY", "CA", "IL", "TX", "AZ", "PA", "FL", "MA", "WA", "CO"], 10000
        ),
        "zip_code": np.random.randint(10000, 99999, 10000),
        "income": np.random.normal(50000, 15000, 10000).round(2),
        "credit_score": np.random.randint(300, 850, 10000),
        "subscription_type": np.random.choice(
            ["Basic", "Premium", "Enterprise"], 10000, p=[0.5, 0.3, 0.2]
        ),
        "monthly_spend": np.random.exponential(100, 10000).round(2),
        "last_login": pd.date_range("2023-01-01", periods=10000, freq="2H"),
        "account_status": np.random.choice(
            ["Active", "Inactive", "Suspended"], 10000, p=[0.7, 0.25, 0.05]
        ),
    }
)

# Phase 1: Basic column additions and name concatenation
df["full_name"] = df["first_name"] + " " + df["last_name"]
df["initials"] = df["first_name"].str[0] + df["last_name"].str[0]
df["email_username"] = df["email"].str.split("@").str[0]
df["email_domain"] = df["email"].str.split("@").str[1]


# Phase 2: Age-based categorizations
def categorize_age(age):
    if age < 25:
        return "Young Adult"
    elif age < 35:
        return "Adult"
    elif age < 50:
        return "Middle Aged"
    else:
        return "Senior"


df["age_group"] = df["age"].apply(categorize_age)
df["is_millennial"] = (df["age"] >= 28) & (df["age"] <= 43)
df["is_gen_z"] = df["age"] < 28
df["is_gen_x"] = (df["age"] >= 44) & (df["age"] <= 59)
df["is_boomer"] = df["age"] >= 60

# Phase 3: Date calculations
df["days_since_registration"] = (datetime.now() - df["registration_date"]).dt.days
df["weeks_since_registration"] = (df["days_since_registration"] / 7).round(2)
df["months_since_registration"] = (df["days_since_registration"] / 30).round(2)
df["years_since_registration"] = (df["days_since_registration"] / 365).round(2)
df["registration_year"] = df["registration_date"].dt.year
df["registration_month"] = df["registration_date"].dt.month
df["registration_day"] = df["registration_date"].dt.day
df["registration_dayofweek"] = df["registration_date"].dt.dayofweek
df["registration_quarter"] = df["registration_date"].dt.quarter
df["is_weekend_registration"] = df["registration_dayofweek"].isin([5, 6])

# Phase 4: Last login calculations
df["days_since_last_login"] = (datetime.now() - df["last_login"]).dt.days
df["weeks_since_last_login"] = (df["days_since_last_login"] / 7).round(2)
df["last_login_year"] = df["last_login"].dt.year
df["last_login_month"] = df["last_login"].dt.month
df["last_login_day"] = df["last_login"].dt.day
df["last_login_hour"] = df["last_login"].dt.hour
df["is_recent_login"] = df["days_since_last_login"] <= 7
df["is_stale_account"] = df["days_since_last_login"] > 90

# Phase 5: Income analysis
df["income_bracket"] = pd.cut(
    df["income"],
    bins=[0, 30000, 50000, 75000, 100000, float("inf")],
    labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
)
df["income_thousands"] = (df["income"] / 1000).round(2)
df["is_high_income"] = df["income"] > 75000
df["is_low_income"] = df["income"] < 30000
df["income_log"] = np.log1p(df["income"])
df["income_sqrt"] = np.sqrt(df["income"])

# Phase 6: Credit score analysis
df["credit_category"] = pd.cut(
    df["credit_score"],
    bins=[0, 579, 669, 739, 799, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
)
df["is_excellent_credit"] = df["credit_score"] >= 800
df["is_good_credit"] = df["credit_score"] >= 670
df["is_poor_credit"] = df["credit_score"] < 580
df["credit_score_normalized"] = (df["credit_score"] - 300) / (850 - 300)
df["credit_risk_level"] = np.where(
    df["credit_score"] >= 740,
    "Low Risk",
    np.where(df["credit_score"] >= 670, "Medium Risk", "High Risk"),
)

# Phase 7: Spending analysis
df["monthly_spend_log"] = np.log1p(df["monthly_spend"])
df["monthly_spend_sqrt"] = np.sqrt(df["monthly_spend"])
df["is_high_spender"] = df["monthly_spend"] > df["monthly_spend"].quantile(0.75)
df["is_low_spender"] = df["monthly_spend"] < df["monthly_spend"].quantile(0.25)
df["annual_spend_estimate"] = df["monthly_spend"] * 12
df["quarterly_spend_estimate"] = df["monthly_spend"] * 3
df["weekly_spend_estimate"] = (df["monthly_spend"] / 4).round(2)
df["daily_spend_estimate"] = (df["monthly_spend"] / 30).round(2)

# Phase 8: Subscription analysis
subscription_values = {"Basic": 9.99, "Premium": 19.99, "Enterprise": 49.99}
df["subscription_value"] = df["subscription_type"].map(subscription_values)
df["is_basic_subscriber"] = df["subscription_type"] == "Basic"
df["is_premium_subscriber"] = df["subscription_type"] == "Premium"
df["is_enterprise_subscriber"] = df["subscription_type"] == "Enterprise"
df["annual_subscription_cost"] = df["subscription_value"] * 12
df["subscription_to_spend_ratio"] = (
    df["subscription_value"] / (df["monthly_spend"] + 1)
).round(4)

# Phase 9: Account status analysis
df["is_active"] = df["account_status"] == "Active"
df["is_inactive"] = df["account_status"] == "Inactive"
df["is_suspended"] = df["account_status"] == "Suspended"
df["account_health_score"] = np.where(
    df["is_active"], 100, np.where(df["is_inactive"], 50, 0)
)

# Phase 10: Geographic analysis
region_mapping = {
    "NY": "Northeast",
    "PA": "Northeast",
    "MA": "Northeast",
    "CA": "West",
    "AZ": "West",
    "WA": "West",
    "CO": "West",
    "IL": "Midwest",
    "TX": "South",
    "FL": "South",
}
df["region"] = df["state"].map(region_mapping)
df["is_northeast"] = df["region"] == "Northeast"
df["is_west"] = df["region"] == "West"
df["is_midwest"] = df["region"] == "Midwest"
df["is_south"] = df["region"] == "South"
df["is_coastal"] = df["state"].isin(["CA", "NY", "FL", "WA", "MA"])
df["is_major_city"] = df["city"].isin(["New York", "Los Angeles", "Chicago"])

# Phase 11: Season calculations
df["registration_season"] = df["registration_date"].dt.month.map(
    {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }
)
df["is_winter_registration"] = df["registration_season"] == "Winter"
df["is_summer_registration"] = df["registration_season"] == "Summer"
df["is_spring_registration"] = df["registration_season"] == "Spring"
df["is_fall_registration"] = df["registration_season"] == "Fall"

# Phase 12: Combined scoring metrics
df["customer_value_score"] = (
    (df["monthly_spend"] * 0.4)
    + (df["credit_score"] / 10 * 0.3)
    + (df["days_since_registration"] / 10 * 0.2)
    + (df["income"] / 1000 * 0.1)
).round(2)
df["engagement_score_numeric"] = np.where(
    df["days_since_last_login"] <= 7,
    100,
    np.where(df["days_since_last_login"] <= 30, 60, 20),
)
df["engagement_score"] = np.where(
    df["days_since_last_login"] <= 7,
    "High",
    np.where(df["days_since_last_login"] <= 30, "Medium", "Low"),
)
df["is_high_engagement"] = df["engagement_score"] == "High"
df["is_low_engagement"] = df["engagement_score"] == "Low"


# Phase 13: Risk scoring
def calculate_risk_score(row):
    base_score = (850 - row["credit_score"]) / 10
    if row["account_status"] == "Suspended":
        base_score += 20
    elif row["account_status"] == "Inactive":
        base_score += 10
    return round(base_score, 2)


df["risk_score"] = df.apply(calculate_risk_score, axis=1)
df["is_high_risk"] = df["risk_score"] > 30
df["is_low_risk"] = df["risk_score"] < 15
df["risk_category"] = pd.cut(
    df["risk_score"],
    bins=[0, 15, 30, 45, float("inf")],
    labels=["Low", "Medium", "High", "Critical"],
)

# Phase 14: Lifetime value calculations
df["estimated_lifetime_value"] = (
    df["monthly_spend"] * 12 + df["subscription_value"] * 12
).round(2)
df["lifetime_value_thousands"] = (df["estimated_lifetime_value"] / 1000).round(2)
df["lifetime_value_per_month_registered"] = (
    df["estimated_lifetime_value"] / (df["months_since_registration"] + 1)
).round(2)
df["ltv_to_income_ratio"] = (df["estimated_lifetime_value"] / df["income"]).round(4)

# Phase 15: Loyalty metrics
df["loyalty_tier"] = pd.cut(
    df["customer_value_score"],
    bins=[0, 100, 200, 300, float("inf")],
    labels=["Bronze", "Silver", "Gold", "Platinum"],
)
df["is_platinum"] = df["loyalty_tier"] == "Platinum"
df["is_gold"] = df["loyalty_tier"] == "Gold"
df["is_silver"] = df["loyalty_tier"] == "Silver"
df["is_bronze"] = df["loyalty_tier"] == "Bronze"
df["loyalty_points"] = (df["customer_value_score"] * 10).astype(int)
df["loyalty_level"] = (df["loyalty_points"] / 1000).astype(int)

# Phase 16: Data cleaning - filter out negative income
df = df[df["income"] > 0].copy()
df["post_filter_customer_count"] = len(df)

# Phase 17: High value customer identification
df["is_high_value"] = df["customer_value_score"] > df["customer_value_score"].quantile(
    0.8
)
df["is_top_10_percent"] = df["customer_value_score"] > df[
    "customer_value_score"
].quantile(0.9)
df["is_top_5_percent"] = df["customer_value_score"] > df[
    "customer_value_score"
].quantile(0.95)
df["is_bottom_20_percent"] = df["customer_value_score"] < df[
    "customer_value_score"
].quantile(0.2)

# Phase 18: Spending velocity calculations
df["spend_per_day"] = (df["monthly_spend"] / 30).round(4)
df["spend_per_day_registered"] = (
    df["monthly_spend"] / (df["days_since_registration"] + 1)
).round(4)
df["spend_acceleration"] = (
    df["monthly_spend"] / (df["months_since_registration"] + 1)
).round(2)


# Phase 19: Advanced segmentation
def segment_customer(row):
    if row["engagement_score"] == "High" and row["is_high_value"]:
        return "Champions"
    elif row["engagement_score"] == "High":
        return "Loyal Customers"
    elif row["is_high_value"]:
        return "Big Spenders"
    elif row["engagement_score"] == "Low" and row["days_since_last_login"] > 90:
        return "At Risk"
    else:
        return "Regular"


df["customer_segment"] = df.apply(segment_customer, axis=1)
df["is_champion"] = df["customer_segment"] == "Champions"
df["is_at_risk"] = df["customer_segment"] == "At Risk"
df["is_big_spender"] = df["customer_segment"] == "Big Spenders"
df["is_regular_customer"] = df["customer_segment"] == "Regular"


# Phase 20: Contact preferences
def determine_contact_preference(row):
    if (
        row["age_group"] in ["Young Adult", "Adult"]
        and row["engagement_score"] == "High"
    ):
        return "Email + SMS"
    elif row["age_group"] in ["Middle Aged", "Senior"]:
        return "Email + Phone"
    else:
        return "Email Only"


df["preferred_contact_method"] = df.apply(determine_contact_preference, axis=1)
df["accepts_sms"] = df["preferred_contact_method"].str.contains("SMS")
df["accepts_phone"] = df["preferred_contact_method"].str.contains("Phone")
df["email_only"] = df["preferred_contact_method"] == "Email Only"

# Phase 21: Normalization and scaling
df["normalized_income"] = scaler.fit_transform(df[["income"]]).flatten()
df["normalized_spend"] = scaler.fit_transform(df[["monthly_spend"]]).flatten()
df["normalized_credit_score"] = scaler.fit_transform(df[["credit_score"]]).flatten()
df["normalized_age"] = scaler.fit_transform(df[["age"]]).flatten()
df["standardized_income"] = std_scaler.fit_transform(df[["income"]]).flatten()
df["standardized_spend"] = std_scaler.fit_transform(df[["monthly_spend"]]).flatten()

# Phase 22: Percentile rankings
df["percentile_rank_income"] = df["income"].rank(pct=True).round(3)
df["percentile_rank_spend"] = df["monthly_spend"].rank(pct=True).round(3)
df["percentile_rank_credit"] = df["credit_score"].rank(pct=True).round(3)
df["percentile_rank_age"] = df["age"].rank(pct=True).round(3)
df["percentile_rank_value"] = df["customer_value_score"].rank(pct=True).round(3)

# Phase 23: Composite scores
df["financial_health_score"] = (
    (df["normalized_income"] * 0.4)
    + (df["normalized_credit_score"] * 0.4)
    + (df["normalized_spend"] * 0.2)
).round(3)
df["engagement_value_score"] = (
    (df["engagement_score_numeric"] / 100 * 0.5) + (df["normalized_spend"] * 0.5)
).round(3)
df["overall_customer_score"] = (
    (df["customer_value_score"] / df["customer_value_score"].max() * 0.4)
    + (df["financial_health_score"] * 0.3)
    + (df["engagement_value_score"] * 0.3)
).round(3)

# Phase 24: Outlier detection and removal
spend_lower = df["monthly_spend"].quantile(0.01)
spend_upper = df["monthly_spend"].quantile(0.99)
df["is_spend_outlier"] = (df["monthly_spend"] < spend_lower) | (
    df["monthly_spend"] > spend_upper
)
df = df[~df["is_spend_outlier"]].copy()

# Phase 25: Income outlier detection
income_lower = df["income"].quantile(0.01)
income_upper = df["income"].quantile(0.99)
df["is_income_outlier"] = (df["income"] < income_lower) | (df["income"] > income_upper)
df = df[~df["is_income_outlier"]].copy()

# Phase 26: Recalculate ranks after filtering
df["post_filter_income_rank"] = df["income"].rank(pct=True).round(3)
df["post_filter_spend_rank"] = df["monthly_spend"].rank(pct=True).round(3)
df["post_filter_value_rank"] = df["customer_value_score"].rank(pct=True).round(3)

# Phase 27: Time-based cohort analysis
df["registration_cohort"] = df["registration_date"].dt.to_period("M")
df["registration_cohort_str"] = df["registration_cohort"].astype(str)
df["is_early_cohort"] = df["registration_year"] == 2020
df["is_late_cohort"] = df["registration_year"] >= 2023
df["cohort_age_days"] = df["days_since_registration"]

# Phase 28: Additional binary flags
df["needs_attention"] = (
    (df["is_at_risk"]) | (df["is_suspended"]) | (df["is_low_engagement"])
)
df["vip_customer"] = (
    (df["is_champion"]) & (df["is_excellent_credit"]) & (df["is_high_income"])
)
df["growth_potential"] = (
    (df["is_good_credit"]) & (df["is_low_spender"]) & (df["is_active"])
)
df["churn_risk"] = (
    (df["is_stale_account"]) | (df["is_low_engagement"]) | (df["is_inactive"])
)

# Phase 29: Marketing flags
df["email_campaign_eligible"] = (df["is_active"]) & (~df["is_suspended"])
df["premium_upgrade_target"] = (df["is_basic_subscriber"]) & (df["is_high_value"])
df["retention_campaign_target"] = (df["is_at_risk"]) | (df["churn_risk"])
df["cross_sell_target"] = (df["is_high_engagement"]) & (df["monthly_spend"] > 100)

# Phase 30: Value bands
df["value_band"] = pd.cut(
    df["customer_value_score"], bins=10, labels=[f"Band_{i}" for i in range(1, 11)]
)
df["income_decile"] = pd.qcut(df["income"], 10, labels=[f"D{i}" for i in range(1, 11)])
df["spend_quintile"] = pd.qcut(
    df["monthly_spend"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
)

# Phase 31: Drop redundant temporary columns
df = df.drop(["is_spend_outlier", "is_income_outlier"], axis=1)
df = df.drop(["post_filter_customer_count"], axis=1)

# Phase 32: Gender-based analysis
df["gender_encoded"] = df["gender"].map({"M": 1, "F": 2, "Other": 3})
df["is_male"] = df["gender"] == "M"
df["is_female"] = df["gender"] == "F"

# Phase 33: Email analysis
df["email_length"] = df["email"].str.len()
df["username_length"] = df["email_username"].str.len()
df["has_numeric_email"] = df["email_username"].str.contains(r"\d")
df["email_domain_length"] = df["email_domain"].str.len()

# Phase 34: Name analysis
df["full_name_length"] = df["full_name"].str.len()
df["name_has_space"] = df["full_name"].str.contains(" ")
df["first_name_length"] = df["first_name"].str.len()
df["last_name_length"] = df["last_name"].str.len()

# Phase 35: Drop name columns after analysis
df = df.drop(["first_name", "last_name", "initials"], axis=1)

# Phase 36: Recalculate some metrics after drops
df["active_columns_count"] = len(df.columns)
df["row_number"] = range(1, len(df) + 1)

# Phase 37: Zip code analysis
df["zip_first_digit"] = df["zip_code"].astype(str).str[0].astype(int)
df["zip_region_code"] = df["zip_first_digit"].map(
    {
        0: "Northeast",
        1: "Northeast",
        2: "Mid-Atlantic",
        3: "Southeast",
        4: "Great Lakes",
        5: "South Central",
        6: "North Central",
        7: "South Central",
        8: "Western",
        9: "Pacific",
    }
)
df["is_high_zip"] = df["zip_code"] > 50000

# Phase 38: Drop intermediate zip columns
df = df.drop(["zip_first_digit"], axis=1)

# Phase 39: Activity recency scoring
df["activity_recency_score"] = np.where(
    df["days_since_last_login"] <= 7,
    5,
    np.where(
        df["days_since_last_login"] <= 14,
        4,
        np.where(
            df["days_since_last_login"] <= 30,
            3,
            np.where(df["days_since_last_login"] <= 60, 2, 1),
        ),
    ),
)
df["registration_recency_score"] = np.where(
    df["days_since_registration"] <= 30,
    5,
    np.where(
        df["days_since_registration"] <= 90,
        4,
        np.where(
            df["days_since_registration"] <= 180,
            3,
            np.where(df["days_since_registration"] <= 365, 2, 1),
        ),
    ),
)

# Phase 40: Combined recency score
df["combined_recency_score"] = (
    df["activity_recency_score"] * 0.6 + df["registration_recency_score"] * 0.4
).round(2)

# Phase 41: Drop some day-based columns
df = df.drop(["registration_day", "last_login_day"], axis=1)

# Phase 42: Subscription tenure
df["subscription_tenure_months"] = df["months_since_registration"]
df["subscription_tenure_years"] = df["years_since_registration"]
df["is_long_term_customer"] = df["subscription_tenure_years"] >= 3
df["is_new_customer"] = df["subscription_tenure_months"] <= 6

# Phase 43: Drop hourly login data
df = df.drop(["last_login_hour"], axis=1)

# Phase 44: Revenue projections
df["projected_revenue_1yr"] = (
    df["monthly_spend"] * 12 + df["annual_subscription_cost"]
).round(2)
df["projected_revenue_2yr"] = (df["projected_revenue_1yr"] * 2).round(2)
df["projected_revenue_5yr"] = (df["projected_revenue_1yr"] * 5).round(2)
df["revenue_growth_potential"] = (
    df["projected_revenue_5yr"] * df["engagement_score_numeric"] / 100
).round(2)

# Phase 45: Credit utilization proxy
df["credit_to_income_ratio"] = (df["credit_score"] / df["income_thousands"]).round(2)
df["spend_to_credit_ratio"] = (df["monthly_spend"] / df["credit_score"] * 100).round(4)

# Phase 46: Drop some boolean generation flags
df = df.drop(["is_millennial", "is_gen_z", "is_gen_x", "is_boomer"], axis=1)

# Phase 47: Seasonal spend analysis
df["is_q4_registration"] = df["registration_quarter"] == 4
df["is_q1_registration"] = df["registration_quarter"] == 1
df["registration_half"] = np.where(df["registration_quarter"].isin([1, 2]), "H1", "H2")
df["is_h1_registration"] = df["registration_half"] == "H1"

# Phase 48: Drop some seasonal flags
df = df.drop(["is_winter_registration", "is_summer_registration"], axis=1)
df = df.drop(["is_spring_registration", "is_fall_registration"], axis=1)

# Phase 49: Engagement metrics combinations
df["engagement_value_product"] = (
    df["engagement_score_numeric"] * df["customer_value_score"]
)
df["engagement_spend_product"] = df["engagement_score_numeric"] * df["monthly_spend"]
df["loyalty_engagement_score"] = (
    df["loyalty_points"] / 100 + df["engagement_score_numeric"]
).round(2)

# Phase 50: Drop some engagement booleans
df = df.drop(["is_high_engagement", "is_low_engagement"], axis=1)

# Phase 51: Customer lifecycle stage
df["lifecycle_stage"] = np.where(
    df["is_new_customer"],
    "New",
    np.where(
        df["is_long_term_customer"],
        "Mature",
        np.where(df["is_active"], "Growing", "Declining"),
    ),
)
df["is_mature_customer"] = df["lifecycle_stage"] == "Mature"
df["is_declining_customer"] = df["lifecycle_stage"] == "Declining"

# Phase 52: Geographic spending patterns
df["urban_premium"] = np.where(
    df["is_major_city"], df["monthly_spend"] * 1.1, df["monthly_spend"]
)
df["regional_spend_adjustment"] = np.where(
    df["is_northeast"],
    df["monthly_spend"] * 1.15,
    np.where(
        df["is_west"],
        df["monthly_spend"] * 1.10,
        np.where(df["is_south"], df["monthly_spend"] * 0.95, df["monthly_spend"]),
    ),
)

# Phase 53: Drop some geographic booleans
df = df.drop(["is_northeast", "is_west", "is_midwest", "is_south"], axis=1)
df = df.drop(["is_coastal", "is_major_city"], axis=1)

# Phase 54: Risk-adjusted metrics
df["risk_adjusted_value"] = (
    df["customer_value_score"] * (100 - df["risk_score"]) / 100
).round(2)
df["risk_adjusted_ltv"] = (
    df["estimated_lifetime_value"] * (100 - df["risk_score"]) / 100
).round(2)

# Phase 55: Drop some risk booleans
df = df.drop(["is_high_risk", "is_low_risk"], axis=1)

# Phase 56: Subscription economics
df["subscription_roi"] = (df["monthly_spend"] / df["subscription_value"]).round(2)
df["subscription_efficiency"] = (
    df["estimated_lifetime_value"] / df["annual_subscription_cost"]
).round(2)
df["is_subscription_efficient"] = df["subscription_efficiency"] > 10

# Phase 57: Drop subscription booleans
df = df.drop(
    ["is_basic_subscriber", "is_premium_subscriber", "is_enterprise_subscriber"], axis=1
)

# Phase 58: Account health indicators
df["account_health_composite"] = (
    df["account_health_score"] * 0.4
    + df["engagement_score_numeric"] * 0.3
    + df["activity_recency_score"] * 10 * 0.3
).round(2)
df["is_healthy_account"] = df["account_health_composite"] > 75

# Phase 59: Drop account status booleans
df = df.drop(["is_active", "is_inactive", "is_suspended"], axis=1)

# Phase 60: Income-spending relationships
df["discretionary_income"] = (df["income"] / 12 - df["monthly_spend"]).round(2)
df["spend_to_income_pct"] = (df["monthly_spend"] / (df["income"] / 12) * 100).round(2)
df["has_high_discretionary"] = df["discretionary_income"] > 1000
df["is_overspending"] = df["spend_to_income_pct"] > 50

# Phase 61: Drop some income booleans
df = df.drop(["is_high_income", "is_low_income"], axis=1)

# Phase 62: Credit score brackets
df["credit_score_bracket"] = (df["credit_score"] // 50) * 50
df["is_prime_credit"] = df["credit_score"] >= 660
df["is_subprime_credit"] = df["credit_score"] < 660

# Phase 63: Drop credit booleans
df = df.drop(["is_excellent_credit", "is_good_credit", "is_poor_credit"], axis=1)

# Phase 64: Spending patterns
df["is_consistent_spender"] = (
    df["monthly_spend"] >= df["monthly_spend"].quantile(0.4)
) & (df["monthly_spend"] <= df["monthly_spend"].quantile(0.6))
df["spend_volatility_proxy"] = abs(df["monthly_spend"] - df["monthly_spend"].median())

# Phase 65: Drop high/low spender flags
df = df.drop(["is_high_spender", "is_low_spender"], axis=1)

# Phase 66: Loyalty program features
df["loyalty_multiplier"] = np.where(
    df["is_platinum"], 3, np.where(df["is_gold"], 2, np.where(df["is_silver"], 1.5, 1))
)
df["adjusted_loyalty_points"] = (
    df["loyalty_points"] * df["loyalty_multiplier"]
).astype(int)

# Phase 67: Drop loyalty tier booleans
df = df.drop(["is_platinum", "is_gold", "is_silver", "is_bronze"], axis=1)

# Phase 68: Value percentile groups
df["value_percentile_group"] = pd.cut(
    df["percentile_rank_value"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["Bottom 20%", "Lower Middle", "Middle", "Upper Middle", "Top 20%"],
)

# Phase 69: Drop value quantile flags
df = df.drop(["is_top_10_percent", "is_top_5_percent", "is_bottom_20_percent"], axis=1)

# Phase 70: Segment-specific metrics
df["segment_value"] = df.apply(
    lambda x: (
        x["customer_value_score"] * 1.5
        if x["is_champion"]
        else x["customer_value_score"] * 1.2
        if x["is_big_spender"]
        else x["customer_value_score"]
    ),
    axis=1,
).round(2)

# Phase 71: Drop segment booleans
df = df.drop(["is_champion", "is_big_spender", "is_regular_customer"], axis=1)

# Phase 72: Contact preference optimization
df["contact_frequency"] = np.where(
    df["vip_customer"], "Weekly", np.where(df["is_high_value"], "Biweekly", "Monthly")
)
df["preferred_contact_day"] = np.where(
    df["is_weekend_registration"], "Weekend", "Weekday"
)

# Phase 73: Drop contact booleans
df = df.drop(["accepts_sms", "accepts_phone", "email_only"], axis=1)

# Phase 74: Advanced customer flags
df["needs_reengagement"] = (df["is_stale_account"]) & (df["is_high_value"])
df["upsell_ready"] = (df["is_recent_login"]) & (df["monthly_spend"] > 150)
df["winback_candidate"] = (df["is_at_risk"]) & (df["customer_value_score"] > 200)

# Phase 75: Drop reengagement flags
df = df.drop(["is_recent_login", "is_stale_account"], axis=1)

# Phase 76: Marketing campaign scores
df["email_campaign_score"] = (
    df["engagement_score_numeric"] * 0.4
    + df["customer_value_score"] / 10 * 0.3
    + df["activity_recency_score"] * 10 * 0.3
).round(2)
df["retention_campaign_score"] = (
    df["churn_risk"].astype(int) * 50 + df["customer_value_score"] / 10 * 0.5
).round(2)

# Phase 77: Drop campaign flags
df = df.drop(["email_campaign_eligible", "premium_upgrade_target"], axis=1)
df = df.drop(["retention_campaign_target", "cross_sell_target"], axis=1)

# Phase 78: Time-based value metrics
df["value_per_day_registered"] = (
    df["customer_value_score"] / (df["days_since_registration"] + 1)
).round(4)
df["value_acceleration"] = (
    df["customer_value_score"] / (df["months_since_registration"] + 1)
).round(2)

# Phase 79: Drop some time columns
df = df.drop(["weeks_since_registration", "weeks_since_last_login"], axis=1)

# Phase 80: Final composite scores
df["master_customer_score"] = (
    df["overall_customer_score"] * 0.3
    + df["financial_health_score"] * 0.25
    + df["engagement_value_score"] * 0.25
    + df["account_health_composite"] / 100 * 0.2
).round(3)

# Phase 81: Drop intermediate scores
df = df.drop(["overall_customer_score"], axis=1)

# Phase 82: Gender-based patterns
df["gender_spend_interaction"] = df["gender_encoded"] * df["normalized_spend"]

# Phase 83: Drop gender flags
df = df.drop(["is_male", "is_female"], axis=1)

# Phase 84: Email pattern analysis
df["email_complexity_score"] = (
    df["email_length"] * 0.4
    + df["username_length"] * 0.3
    + df["email_domain_length"] * 0.3
).round(2)

# Phase 85: Drop email detail columns
df = df.drop(["email_username", "email_length", "username_length"], axis=1)
df = df.drop(["email_domain_length", "has_numeric_email"], axis=1)

# Phase 86: Name analysis cleanup
df = df.drop(["full_name_length", "name_has_space"], axis=1)
df = df.drop(["first_name_length", "last_name_length"], axis=1)

# Phase 87: Drop row metadata
df = df.drop(["active_columns_count", "row_number"], axis=1)

# Phase 88: Zip code cleanup
df = df.drop(["is_high_zip"], axis=1)

# Phase 89: Activity score cleanup
df = df.drop(["activity_recency_score", "registration_recency_score"], axis=1)

# Phase 90: Drop quarter flags
df = df.drop(["is_q4_registration", "is_q1_registration", "is_h1_registration"], axis=1)

# Phase 91: Drop lifecycle flags
df = df.drop(["is_mature_customer", "is_declining_customer"], axis=1)

# Phase 92: Drop subscription efficiency flag
df = df.drop(["is_subscription_efficient"], axis=1)

# Phase 93: Drop account health flag
df = df.drop(["is_healthy_account"], axis=1)

# Phase 94: Drop discretionary flags
df = df.drop(["has_high_discretionary", "is_overspending"], axis=1)

# Phase 95: Drop credit flags
df = df.drop(["is_prime_credit", "is_subprime_credit"], axis=1)

# Phase 96: Drop spending pattern flag
df = df.drop(["is_consistent_spender"], axis=1)

# Phase 97: Drop marketing flags
df = df.drop(["needs_reengagement", "upsell_ready", "winback_candidate"], axis=1)

# Phase 98: Final data quality
df = df.dropna()

# Phase 99: Sort and reset
df = df.sort_values("customer_value_score", ascending=False).reset_index(drop=True)

# Phase 100: Final metadata
df["final_column_count"] = len(df.columns)
df["processing_complete"] = True
df["data_version"] = "v1.0"

print(f"\nFinal DataFrame shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
print(f"\nTotal rows: {len(df)}")
print("\nSample data:")
print(df.head(3))
