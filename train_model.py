# ============================================================
# CARPREDICT AI - CAR PRICE PREDICTION MODEL TRAINING
# ============================================================
#
# Project:
#   CarPredict AI
#
# Purpose:
#   Train a machine learning model to predict the selling price
#   of Indian used cars using the supplied 100K-row dataset.
#
# Target:
#   selling_price_lakh
#
# Model:
#   CatBoostRegressor
#
# Why CatBoost?
#   - Excellent for tabular data
#   - Handles categorical variables naturally
#   - Handles missing values
#   - Works well with large datasets
#   - Avoids manually assigning numbers to categories
#
# IMPORTANT:
#   We intentionally exclude price-derived columns from training
#   to reduce target leakage.
#
# ============================================================


from pathlib import Path
import json
import sys
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from catboost import CatBoostRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

MODEL_FILE = MODEL_DIR / "car_price_model.cbm"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


# ============================================================
# 3. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "selling_price_lakh"


# ============================================================
# 4. FEATURES USED BY THE MODEL
# ============================================================
#
# These are fields that can reasonably be supplied by the
# website user.
#
# We deliberately DO NOT use:
#
# listing_id
# car_name
# vehicle_age
# original_price_lakh
# estimated_monthly_emi
# listing_date
# price_per_km
# depreciation_percent
#
# The excluded price-derived fields could leak information
# about the target price.
#
# ============================================================

FEATURE_COLUMNS = [

    # -----------------------------
    # Basic vehicle information
    # -----------------------------

    "brand",
    "model",
    "variant",
    "body_type",

    "manufacturing_year",
    "registration_year",

    "kilometers_driven",

    "fuel_type",
    "transmission",

    # -----------------------------
    # Engine / performance
    # -----------------------------

    "engine_cc",
    "max_power_bhp",
    "mileage_kmpl",
    "seats",

    # -----------------------------
    # Ownership / seller
    # -----------------------------

    "ownership",
    "seller_type",

    # -----------------------------
    # Location
    # -----------------------------

    "city",
    "state",

    # -----------------------------
    # Appearance / condition
    # -----------------------------

    "color",
    "condition",

    # -----------------------------
    # History
    # -----------------------------

    "insurance_type",
    "service_history",
    "accident_history",
    "warranty_available",
    "tyre_condition",

    # -----------------------------
    # Features
    # -----------------------------

    "features_count",
]


# ============================================================
# 5. CATEGORICAL COLUMNS
# ============================================================
#
# CatBoost will handle these columns directly.
#
# ============================================================

CATEGORICAL_COLUMNS = [

    "brand",
    "model",
    "variant",
    "body_type",

    "fuel_type",
    "transmission",

    "ownership",
    "seller_type",

    "city",
    "state",

    "color",
    "condition",

    "insurance_type",
    "service_history",
    "accident_history",
    "warranty_available",
    "tyre_condition",
]


# ============================================================
# 6. NUMERICAL COLUMNS
# ============================================================

NUMERICAL_COLUMNS = [

    "manufacturing_year",
    "registration_year",
    "kilometers_driven",

    "engine_cc",
    "max_power_bhp",
    "mileage_kmpl",
    "seats",

    "features_count",
]


# ============================================================
# 7. FIND DATASET
# ============================================================

def find_dataset():
    """
    Locate the CSV dataset.

    Priority:
        1. dataset/*.csv
        2. project_root/*.csv

    This makes the project more tolerant of filename changes.
    """

    dataset_files = list(DATASET_DIR.glob("*.csv"))

    if len(dataset_files) == 0:
        dataset_files = list(BASE_DIR.glob("*.csv"))

    if len(dataset_files) == 0:

        raise FileNotFoundError(
            "\n"
            "====================================================\n"
            "DATASET NOT FOUND\n"
            "====================================================\n"
            "No CSV file was found.\n\n"
            "Please put your CSV file inside:\n\n"
            f"{DATASET_DIR}\n\n"
            "Example:\n"
            "dataset/CarPredict_AI_100K_Indian_Used_Cars(1).csv\n"
            "====================================================\n"
        )

    if len(dataset_files) > 1:

        print("\nMultiple CSV files were found.")

        print("The following files are available:")

        for index, file in enumerate(dataset_files, start=1):
            print(f"{index}. {file.name}")

        print("\nThe first CSV file will be selected automatically.")

    return dataset_files[0]


# ============================================================
# 8. CLEAN DATA
# ============================================================

def clean_dataset(df):
    """
    Basic and safe dataset cleaning.

    We do NOT randomly delete rows just because a field is missing.

    CatBoost can handle missing numeric values and we convert missing
    categorical values into a consistent 'Unknown' category.
    """

    print("\n")
    print("=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    before_rows = len(df)

    df = df.dropna(how="all")

    after_rows = len(df)

    if before_rows != after_rows:

        print(
            f"Removed {before_rows - after_rows} completely empty rows."
        )

    # --------------------------------------------------------
    # Remove duplicate complete rows
    # --------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    print(f"Exact duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:

        df = df.drop_duplicates()

        print(
            f"Removed {duplicate_count} duplicate rows."
        )

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' does not exist "
            "in the dataset."
        )

    # --------------------------------------------------------
    # Convert target to numeric
    # --------------------------------------------------------

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    missing_target = int(df[TARGET_COLUMN].isna().sum())

    if missing_target > 0:

        print(
            f"Rows with missing target price: {missing_target}"
        )

        df = df.dropna(subset=[TARGET_COLUMN])

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    for column in NUMERICAL_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:
            continue

        # Convert to string only after replacing missing values.
        #
        # This is important because CatBoost categorical features
        # should not receive floating-point NaN values.

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        # Empty strings become Unknown

        df[column] = df[column].replace(
            "",
            "Unknown"
        )

    # --------------------------------------------------------
    # Remove impossible target values
    # --------------------------------------------------------

    invalid_target = int(
        (df[TARGET_COLUMN] <= 0).sum()
    )

    if invalid_target > 0:

        print(
            f"Invalid target prices found: {invalid_target}"
        )

        df = df[
            df[TARGET_COLUMN] > 0
        ]

    # --------------------------------------------------------
    # Final reset
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    print(
        f"Final training rows available: {len(df):,}"
    )

    return df


# ============================================================
# 9. VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_columns(df):
    """
    Make sure every column required by the model exists.
    """

    print("\n")
    print("=" * 70)
    print("COLUMN VALIDATION")
    print("=" * 70)

    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    actual_columns = set(df.columns)

    missing_columns = sorted(
        required_columns - actual_columns
    )

    if missing_columns:

        print("\nMissing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "\nThe dataset does not contain all required "
            "columns for this project."
        )

    print("All required model columns are present.")

    print("\nFeatures used:")

    for index, column in enumerate(
        FEATURE_COLUMNS,
        start=1
    ):

        print(
            f"{index:02d}. {column}"
        )


# ============================================================
# 10. DATASET QUALITY REPORT
# ============================================================

def print_dataset_report(df):
    """
    Print a useful summary before training.
    """

    print("\n")
    print("=" * 70)
    print("DATASET REPORT")
    print("=" * 70)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns):,}"
    )

    print(
        f"Target     : {TARGET_COLUMN}"
    )

    print(
        f"Min price  : ₹{df[TARGET_COLUMN].min():.2f} Lakh"
    )

    print(
        f"Max price  : ₹{df[TARGET_COLUMN].max():.2f} Lakh"
    )

    print(
        f"Mean price : ₹{df[TARGET_COLUMN].mean():.2f} Lakh"
    )

    print(
        f"Median     : ₹{df[TARGET_COLUMN].median():.2f} Lakh"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values in model features:")

    missing = df[FEATURE_COLUMNS].isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:

        print("No missing values found.")

    else:

        for column, count in missing.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"{column:25s} "
                f"{count:8,d} "
                f"({percentage:.2f}%)"
            )

    # --------------------------------------------------------
    # Unique categories
    # --------------------------------------------------------

    print("\nCategorical cardinality:")

    for column in CATEGORICAL_COLUMNS:

        unique_count = (
            df[column]
            .nunique(dropna=False)
        )

        print(
            f"{column:25s}: {unique_count}"
        )


# ============================================================
# 11. CREATE TRAINING DATA
# ============================================================

def prepare_training_data(df):
    """
    Prepare X and y.
    """

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    # CatBoost categorical columns are already strings.
    #
    # Make sure numeric columns remain numeric.

    for column in NUMERICAL_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    return X, y


# ============================================================
# 12. TRAIN MODEL
# ============================================================

def train_model(X_train, X_valid, y_train, y_valid):
    """
    Train CatBoost regression model.

    CatBoost is especially suitable here because the dataset contains
    many categorical features such as brand, model, variant, city,
    fuel type, etc.
    """

    print("\n")
    print("=" * 70)
    print("TRAINING CATBOOST MODEL")
    print("=" * 70)

    print(
        "\nTraining rows : "
        f"{len(X_train):,}"
    )

    print(
        "Validation rows : "
        f"{len(X_valid):,}"
    )

    print(
        "\nThis can take some time because the dataset contains "
        "100,000 records."
    )

    model = CatBoostRegressor(

        # ----------------------------------------------------
        # Number of boosting rounds
        # ----------------------------------------------------

        iterations=1200,

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        learning_rate=0.05,

        # ----------------------------------------------------
        # Tree depth
        # ----------------------------------------------------

        depth=9,

        # ----------------------------------------------------
        # Regularization
        # ----------------------------------------------------

        l2_leaf_reg=5,

        # ----------------------------------------------------
        # Randomness
        # ----------------------------------------------------

        random_strength=1,

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        bootstrap_type="Bernoulli",

        subsample=0.8,

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss_function="RMSE",

        eval_metric="RMSE",

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        random_seed=42,

        # ----------------------------------------------------
        # CPU usage
        # ----------------------------------------------------

        thread_count=-1,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        verbose=100,

        # ----------------------------------------------------
        # Stop if validation stops improving
        # ----------------------------------------------------

        allow_writing_files=False
    )

    model.fit(

        X_train,
        y_train,

        cat_features=CATEGORICAL_COLUMNS,

        eval_set=(
            X_valid,
            y_valid
        ),

        use_best_model=True,

        early_stopping_rounds=100
    )

    print("\nTraining completed.")

    return model


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_valid, y_valid):
    """
    Calculate regression metrics.
    """

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = model.predict(
        X_valid
    )

    predictions = np.asarray(
        predictions
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    mse = mean_squared_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    actual = np.asarray(
        y_valid
    )

    non_zero_mask = (
        actual != 0
    )

    if non_zero_mask.any():

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero_mask]
                    -
                    predictions[non_zero_mask]
                )
                /
                actual[non_zero_mask]
            )
        ) * 100

    else:

        mape = float("nan")

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"\nR² Score              : {r2:.6f}"
    )

    print(
        f"R² Percentage        : {r2 * 100:.2f}%"
    )

    print(
        f"MAE                  : ₹{mae:.4f} Lakh"
    )

    print(
        f"RMSE                 : ₹{rmse:.4f} Lakh"
    )

    if not np.isnan(mape):

        print(
            f"MAPE                 : {mape:.2f}%"
        )

    # --------------------------------------------------------
    # Interpret R²
    # --------------------------------------------------------

    if r2 >= 0.90:

        quality = "Excellent"

    elif r2 >= 0.80:

        quality = "Very Good"

    elif r2 >= 0.70:

        quality = "Good"

    elif r2 >= 0.60:

        quality = "Moderate"

    else:

        quality = "Needs Improvement"

    print(
        f"\nModel quality        : {quality}"
    )

    return {

        "r2_score": float(r2),

        "r2_percentage": float(
            r2 * 100
        ),

        "mae_lakh": float(mae),

        "rmse_lakh": float(rmse),

        "mape_percentage": (
            float(mape)
            if not np.isnan(mape)
            else None
        ),

        "quality": quality
    }


# ============================================================
# 14. BUILD BRAND → MODEL → VARIANT DATA
# ============================================================

def build_vehicle_hierarchy(df):
    """
    Build the dependent dropdown structure.

    Example:

    Tata
        Nexon
            Fearless
            Creative
            Pure

    Maruti Suzuki
        Swift
            VXi
            ZXi

    The website will later use this metadata to dynamically
    populate Model and Variant dropdowns.
    """

    print("\n")
    print("=" * 70)
    print("BUILDING VEHICLE DROPDOWN HIERARCHY")
    print("=" * 70)

    hierarchy = {}

    grouped_brand = df.groupby(
        "brand",
        sort=True
    )

    for brand, brand_df in grouped_brand:

        brand_name = str(
            brand
        )

        hierarchy[brand_name] = {}

        grouped_model = brand_df.groupby(
            "model",
            sort=True
        )

        for model, model_df in grouped_model:

            model_name = str(
                model
            )

            variants = sorted(
                model_df["variant"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            hierarchy[brand_name][model_name] = variants

    print(
        f"Brands available: {len(hierarchy)}"
    )

    model_count = sum(
        len(models)
        for models in hierarchy.values()
    )

    print(
        f"Models available: {model_count}"
    )

    variant_count = sum(
        len(variants)
        for models in hierarchy.values()
        for variants in models.values()
    )

    print(
        f"Brand-model variant combinations: "
        f"{variant_count}"
    )

    return hierarchy


# ============================================================
# 15. BUILD WEBSITE OPTIONS
# ============================================================

def build_dropdown_options(df):
    """
    Generate all useful dropdown values for the frontend.
    """

    options = {}

    dropdown_columns = [

        "brand",
        "body_type",
        "fuel_type",
        "transmission",

        "ownership",
        "seller_type",

        "city",
        "state",

        "color",
        "condition",

        "insurance_type",
        "service_history",
        "accident_history",
        "warranty_available",
        "tyre_condition"
    ]

    for column in dropdown_columns:

        values = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .unique()
            .tolist()
        )

        values = sorted(
            values
        )

        options[column] = values

    # --------------------------------------------------------
    # Years
    # --------------------------------------------------------

    options["manufacturing_year"] = sorted(
        df["manufacturing_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist(),
        reverse=True
    )

    options["registration_year"] = sorted(
        df["registration_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist(),
        reverse=True
    )

    # --------------------------------------------------------
    # Seats
    # --------------------------------------------------------

    options["seats"] = sorted(
        df["seats"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return options


# ============================================================
# 16. BUILD NUMERICAL RANGES
# ============================================================

def build_numeric_ranges(df):
    """
    Save realistic ranges for frontend validation.
    """

    ranges = {}

    for column in NUMERICAL_COLUMNS:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) == 0:
            continue

        ranges[column] = {

            "min": float(
                series.min()
            ),

            "max": float(
                series.max()
            ),

            "median": float(
                series.median()
            ),

            "mean": float(
                series.mean()
            )
        }

    return ranges


# ============================================================
# 17. SAVE METADATA
# ============================================================

def save_metadata(
    df,
    metrics,
    hierarchy,
    dropdown_options,
    numeric_ranges,
    model
):
    """
    Save everything the Flask backend will later need.
    """

    print("\n")
    print("=" * 70)
    print("SAVING MODEL METADATA")
    print("=" * 70)

    best_iteration = None

    try:

        best_iteration = int(
            model.get_best_iteration()
        )

    except Exception:

        best_iteration = None

    metadata = {

        # ----------------------------------------------------
        # Project information
        # ----------------------------------------------------

        "project_name":
            "CarPredict AI",

        "project_description":
            "AI powered Indian used car price prediction system",

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        "dataset_rows":
            int(len(df)),

        "dataset_columns":
            int(len(df.columns)),

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        "target_column":
            TARGET_COLUMN,

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        "feature_columns":
            FEATURE_COLUMNS,

        "categorical_columns":
            CATEGORICAL_COLUMNS,

        "numerical_columns":
            NUMERICAL_COLUMNS,

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "model_type":
            "CatBoostRegressor",

        "best_iteration":
            best_iteration,

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        "metrics":
            metrics,

        # ----------------------------------------------------
        # Dropdowns
        # ----------------------------------------------------

        "dropdown_options":
            dropdown_options,

        # ----------------------------------------------------
        # Brand → Model → Variant
        # ----------------------------------------------------

        "vehicle_hierarchy":
            hierarchy,

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        "numeric_ranges":
            numeric_ranges,

        # ----------------------------------------------------
        # Dataset target statistics
        # ----------------------------------------------------

        "price_statistics": {

            "minimum_lakh":
                float(
                    df[TARGET_COLUMN].min()
                ),

            "maximum_lakh":
                float(
                    df[TARGET_COLUMN].max()
                ),

            "average_lakh":
                float(
                    df[TARGET_COLUMN].mean()
                ),

            "median_lakh":
                float(
                    df[TARGET_COLUMN].median()
                )
        },

        # ----------------------------------------------------
        # Training seed
        # ----------------------------------------------------

        "random_seed":
            42
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Metadata saved to:\n{METADATA_FILE}"
    )


# ============================================================
# 18. PRINT SAMPLE PREDICTIONS
# ============================================================

def print_sample_predictions(
    model,
    X_valid,
    y_valid
):
    """
    Show a few real vs predicted prices.
    """

    print("\n")
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    predictions = model.predict(
        X_valid
    )

    sample_count = min(
        10,
        len(X_valid)
    )

    sample_indices = (
        X_valid.index[:sample_count]
    )

    print()

    print(
        f"{'Actual':>15} "
        f"{'Predicted':>15} "
        f"{'Difference':>15}"
    )

    print("-" * 50)

    for index in sample_indices:

        actual_price = float(
            y_valid.loc[index]
        )

        predicted_price = float(
            predictions[
                X_valid.index.get_loc(index)
            ]
        )

        difference = (
            predicted_price
            -
            actual_price
        )

        print(
            f"₹{actual_price:>10.2f} "
            f"₹{predicted_price:>10.2f} "
            f"₹{difference:>10.2f}"
        )


# ============================================================
# 19. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CARPREDICT AI")
    print("100K INDIAN USED CAR PRICE PREDICTION")
    print("=" * 70)

    print(
        "\nStarting model training..."
    )

    # --------------------------------------------------------
    # Find dataset
    # --------------------------------------------------------

    dataset_path = find_dataset()

    print(
        f"\nDataset selected:\n{dataset_path}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        dataset_path
    )

    print(
        f"Loaded {len(df):,} rows."
    )

    print(
        f"Loaded {len(df.columns):,} columns."
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    validate_columns(
        df
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    df = clean_dataset(
        df
    )

    # --------------------------------------------------------
    # Dataset report
    # --------------------------------------------------------

    print_dataset_report(
        df
    )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_training_data(
        df
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    X_train, X_valid, y_train, y_valid = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    print(
        f"Training data   : {len(X_train):,} rows"
    )

    print(
        f"Validation data : {len(X_valid):,} rows"
    )

    print(
        f"Training ratio  : 80%"
    )

    print(
        f"Validation ratio: 20%"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(

        X_train,
        X_valid,

        y_train,
        y_valid
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(

        model,
        X_valid,
        y_valid
    )

    # --------------------------------------------------------
    # Show predictions
    # --------------------------------------------------------

    print_sample_predictions(

        model,
        X_valid,
        y_valid
    )

    # --------------------------------------------------------
    # Build dropdown hierarchy
    # --------------------------------------------------------

    hierarchy = build_vehicle_hierarchy(
        df
    )

    # --------------------------------------------------------
    # Build other dropdown options
    # --------------------------------------------------------

    dropdown_options = build_dropdown_options(
        df
    )

    # --------------------------------------------------------
    # Build numerical ranges
    # --------------------------------------------------------

    numeric_ranges = build_numeric_ranges(
        df
    )

    # --------------------------------------------------------
    # Save CatBoost model
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SAVING TRAINED MODEL")
    print("=" * 70)

    model.save_model(
        str(MODEL_FILE)
    )

    print(
        f"\nModel saved successfully:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(

        df,

        metrics,

        hierarchy,

        dropdown_options,

        numeric_ranges,

        model
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nYour trained model is ready."
    )

    print(
        f"\nModel file:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"\nMetadata file:"
        f"\n{METADATA_FILE}"
    )

    print(
        "\nThe next step will be connecting this model"
        "\nto the Flask backend and website."
    )

    print("\n")


# ============================================================
# 20. ERROR HANDLING
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nTraining cancelled by user."
        )

        sys.exit(1)

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)

        print(
            f"\nError:\n{error}"
        )

        print(
            "\nFull technical traceback:"
        )

        traceback.print_exc()

        print(
            "\nCheck the error above before continuing."
        )

        sys.exit(1)