# ============================================================
# CARPREDICT AI
# FLASK BACKEND
# ============================================================
#
# This backend:
#
# 1. Loads the trained CatBoost model
# 2. Loads model metadata
# 3. Provides vehicle dropdown information
# 4. Provides Brand -> Model -> Variant relationships
# 5. Accepts prediction requests
# 6. Handles optional fields
# 7. Validates user input
# 8. Generates a real prediction
# 9. Returns prediction information as JSON
#
# ============================================================


from pathlib import Path
import json
import math
import re
import traceback
import uuid

import numpy as np
import pandas as pd

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory,
)

import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from catboost import CatBoostRegressor


# ============================================================
# 1. FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# USER DATABASE CONFIGURATION
# ============================================================

app.secret_key = "CarPredict_AI_Change_This_Secret_Key"

DATABASE_FILE = BASE_DIR / "users.db"
MARKETPLACE_DATABASE_FILE = BASE_DIR / "marketplace.db"
MARKETPLACE_UPLOAD_DIR = BASE_DIR / "static" / "uploads" / "marketplace"
MARKETPLACE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# ============================================================
# ADMIN SETTINGS
# ============================================================

ADMIN_USERNAME = "suyash"
ADMIN_PASSWORD = "suyash@123"

MODEL_DIR = BASE_DIR / "model"

# ============================================================
# USER DATABASE FUNCTIONS
# ============================================================

def get_db_connection():
    connection = sqlite3.connect(
        str(DATABASE_FILE)
    )

    connection.row_factory = sqlite3.Row

    return connection

def initialize_database():
    connection = get_db_connection()

    # Create users table if it does not already exist
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'buyer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ---------------------------------------------------------
    # SAFE DATABASE MIGRATION
    # ---------------------------------------------------------
    # If the existing users table was created before the
    # role system was added, add the role column now.
    columns = connection.execute(
        "PRAGMA table_info(users)"
    ).fetchall()

    column_names = [column["name"] for column in columns]

    if "role" not in column_names:
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN role TEXT NOT NULL DEFAULT 'buyer'
        """)

        print("Added 'role' column to users table.")

    connection.commit()
    connection.close()

    marketplace_connection = sqlite3.connect(
        str(MARKETPLACE_DATABASE_FILE)
    )
    marketplace_connection.row_factory = sqlite3.Row
    marketplace_connection.execute("""
        CREATE TABLE IF NOT EXISTS car_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seller_name TEXT NOT NULL,
            seller_email TEXT,
            seller_phone TEXT,
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            year INTEGER NOT NULL,
            kilometers INTEGER NOT NULL,
            price_rupees INTEGER NOT NULL,
            city TEXT NOT NULL,
            fuel_type TEXT,
            transmission TEXT,
            condition TEXT,
            description TEXT,
            photos TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    marketplace_columns = {
        column["name"]
        for column in marketplace_connection.execute(
            "PRAGMA table_info(car_listings)"
        ).fetchall()
    }
    if "photos" not in marketplace_columns:
        marketplace_connection.execute(
            "ALTER TABLE car_listings ADD COLUMN photos TEXT DEFAULT '[]'"
        )
    marketplace_connection.commit()
    marketplace_connection.close()

    print("User database initialized successfully.")

# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:
            return redirect(
                url_for("login")
            )

        return function(*args, **kwargs)

    return decorated_function

# ============================================================
# ADMIN LOGIN REQUIRED
# ============================================================

def admin_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if not session.get("is_admin"):
            return redirect(
                url_for("admin_login")
            )

        return function(*args, **kwargs)

    return decorated_function

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"

MODEL_FILE = (
    MODEL_DIR /
    "car_price_model.cbm"
)

METADATA_FILE = (
    MODEL_DIR /
    "model_metadata.json"
)


# ============================================================
# 3. GLOBAL VARIABLES
# ============================================================

model = None

metadata = None

FEATURE_COLUMNS = []

CATEGORICAL_COLUMNS = []

NUMERICAL_COLUMNS = []

VEHICLE_HIERARCHY = {}

DROPDOWN_OPTIONS = {}

NUMERIC_RANGES = {}

MODEL_METRICS = {}

PRICE_STATISTICS = {}


# ============================================================
# 4. LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained CatBoost model from disk.
    """

    global model

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "\n"
            "Trained model not found.\n\n"
            f"Expected location:\n{MODEL_FILE}\n\n"
            "Run train_model.py first."
        )

    print(
        "\nLoading trained CatBoost model..."
    )

    loaded_model = CatBoostRegressor()

    loaded_model.load_model(
        str(MODEL_FILE)
    )

    model = loaded_model

    print(
        "Model loaded successfully."
    )


# ============================================================
# 5. LOAD METADATA
# ============================================================

def load_metadata():
    """
    Load model metadata generated during training.
    """

    global metadata
    global FEATURE_COLUMNS
    global CATEGORICAL_COLUMNS
    global NUMERICAL_COLUMNS
    global VEHICLE_HIERARCHY
    global DROPDOWN_OPTIONS
    global NUMERIC_RANGES
    global MODEL_METRICS
    global PRICE_STATISTICS

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "\n"
            "Metadata file not found.\n\n"
            f"Expected location:\n{METADATA_FILE}\n\n"
            "Run train_model.py first."
        )

    print(
        "Loading model metadata..."
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    FEATURE_COLUMNS = metadata.get(
        "feature_columns",
        []
    )

    CATEGORICAL_COLUMNS = metadata.get(
        "categorical_columns",
        []
    )

    NUMERICAL_COLUMNS = metadata.get(
        "numerical_columns",
        []
    )

    VEHICLE_HIERARCHY = metadata.get(
        "vehicle_hierarchy",
        {}
    )

    DROPDOWN_OPTIONS = metadata.get(
        "dropdown_options",
        {}
    )

    NUMERIC_RANGES = metadata.get(
        "numeric_ranges",
        {}
    )

    MODEL_METRICS = metadata.get(
        "metrics",
        {}
    )

    PRICE_STATISTICS = metadata.get(
        "price_statistics",
        {}
    )

    print(
        "Metadata loaded successfully."
    )


# ============================================================
# 6. STARTUP
# ============================================================

def initialize_application():

    print("\n")
    print("=" * 70)
    print("CARPREDICT AI BACKEND")
    print("=" * 70)

    initialize_database()
    
    load_model()

    load_metadata()

    print("\nBackend initialization completed.")

    print(
        f"Features loaded: {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Brands loaded: "
        f"{len(VEHICLE_HIERARCHY)}"
    )

    print(
        f"Model R²: "
        f"{MODEL_METRICS.get('r2_percentage', 'N/A')}%"
    )

    print("=" * 70)


# ============================================================
# 7. HELPER - SAFE FLOAT
# ============================================================

def safe_float(value, default=np.nan):
    """
    Convert a value to float safely.
    """

    if value is None:

        return default

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return default

    try:

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# 8. HELPER - SAFE INTEGER
# ============================================================

def safe_int(value, default=np.nan):
    """
    Convert a value to integer safely.
    """

    if value is None:

        return default

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return default

    try:

        result = float(value)

        if not math.isfinite(result):
            return default

        return int(round(result))

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# 9. HELPER - SAFE STRING
# ============================================================

def safe_string(value, default="Unknown"):
    """
    Convert user input into a safe categorical string.
    """

    if value is None:

        return default

    value = str(value).strip()

    if value == "":

        return default

    return value


# ============================================================
# 10. DEFAULT NUMERICAL VALUES
# ============================================================

def get_numeric_default(column):
    """
    Get a sensible default value from the training metadata.

    We use the dataset median rather than 0.

    Zero would be unrealistic for many car properties.
    """

    column_info = NUMERIC_RANGES.get(
        column
    )

    if not column_info:

        return 0

    median = column_info.get(
        "median"
    )

    if median is None:

        return 0

    return median


# ============================================================
# 11. PREPARE USER INPUT
# ============================================================

def prepare_prediction_data(data):
    """
    Convert frontend JSON into exactly the feature structure
    expected by the trained model.
    """

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "Prediction data must be a JSON object."
        )

    row = {}

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    for column in CATEGORICAL_COLUMNS:

        value = data.get(
            column
        )

        row[column] = safe_string(
            value
        )

    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    for column in NUMERICAL_COLUMNS:

        value = data.get(
            column
        )

        numeric_value = safe_float(
            value
        )

        # ----------------------------------------------------
        # Optional fields
        # ----------------------------------------------------

        if np.isnan(
            numeric_value
        ):

            numeric_value = get_numeric_default(
                column
            )

        row[column] = numeric_value

    # ========================================================
    # BUILD DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )

    # ========================================================
    # FORCE CORRECT TYPES
    # ========================================================

    for column in CATEGORICAL_COLUMNS:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
            )

    for column in NUMERICAL_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # ========================================================
    # FINAL NUMERIC FALLBACK
    # ========================================================

    for column in NUMERICAL_COLUMNS:

        if column not in df.columns:
            continue

        if df[column].isna().any():

            df[column] = df[column].fillna(
                get_numeric_default(
                    column
                )
            )

    return df


# ============================================================
# 12. VALIDATE BRAND / MODEL / VARIANT
# ============================================================

def validate_vehicle_selection(data):
    """
    Validate the dependent vehicle dropdown selections.

    Brand -> Model -> Variant
    """

    brand = safe_string(
        data.get("brand"),
        default=""
    )

    model_name = safe_string(
        data.get("model"),
        default=""
    )

    variant = safe_string(
        data.get("variant"),
        default=""
    )

    # --------------------------------------------------------
    # Brand validation
    # --------------------------------------------------------

    if brand == "":

        return

    if brand not in VEHICLE_HIERARCHY:

        raise ValueError(
            f"Invalid car company selected: {brand}"
        )

    # --------------------------------------------------------
    # Model validation
    # --------------------------------------------------------

    if model_name == "":

        return

    available_models = (
        VEHICLE_HIERARCHY
        .get(brand, {})
    )

    if model_name not in available_models:

        raise ValueError(
            f"Model '{model_name}' "
            f"is not available for '{brand}'."
        )

    # --------------------------------------------------------
    # Variant validation
    # --------------------------------------------------------

    if variant == "":

        return

    available_variants = (
        available_models
        .get(model_name, [])
    )

    if variant not in available_variants:

        raise ValueError(
            f"Variant '{variant}' "
            f"is not available for "
            f"'{brand} {model_name}'."
        )


# ============================================================
# 13. PRICE FORMATTER
# ============================================================

def format_price_lakh(price_lakh):
    """
    Format lakh price for the website.
    """

    price_lakh = float(
        price_lakh
    )

    if price_lakh >= 100:

        crore = price_lakh / 100

        return (
            f"₹ {crore:.2f} Crore"
        )

    return (
        f"₹ {price_lakh:.2f} Lakh"
    )


# ============================================================
# 14. PRICE IN RUPEES
# ============================================================

def price_in_rupees(price_lakh):

    return int(
        round(
            float(price_lakh)
            *
            100000
        )
    )


# ============================================================
# 15. MARKET SCORE
# ============================================================

def calculate_market_score(
    predicted_price,
    data
):
    """
    Generate a simple market score.

    IMPORTANT:
    This is NOT the ML model's accuracy.

    It is a presentation score intended for the website UI.

    It is calculated from:
      - predicted price position
      - car condition
      - ownership
      - service history
      - accident history

    The actual price itself always comes from the ML model.
    """

    score = 7.5

    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------

    condition = safe_string(
        data.get(
            "condition"
        )
    ).lower()

    condition_scores = {

        "excellent": 1.0,

        "very good": 0.7,

        "good": 0.4,

        "fair": 0.0,

        "poor": -1.0
    }

    score += condition_scores.get(
        condition,
        0
    )

    # --------------------------------------------------------
    # Ownership
    # --------------------------------------------------------

    ownership = safe_string(
        data.get(
            "ownership"
        )
    ).lower()

    if (
        "first" in ownership
        or
        ownership == "1"
    ):

        score += 0.5

    elif (
        "second" in ownership
        or
        ownership == "2"
    ):

        score += 0.2

    elif (
        "third" in ownership
        or
        ownership == "3"
    ):

        score -= 0.2

    else:

        score -= 0.4

    # --------------------------------------------------------
    # Service history
    # --------------------------------------------------------

    service_history = safe_string(
        data.get(
            "service_history"
        )
    ).lower()

    if (
        "full" in service_history
        or
        "complete" in service_history
    ):

        score += 0.5

    elif (
        "partial" in service_history
    ):

        score += 0.1

    # --------------------------------------------------------
    # Accident history
    # --------------------------------------------------------

    accident_history = safe_string(
        data.get(
            "accident_history"
        )
    ).lower()

    if (
        "no" in accident_history
        or
        "none" in accident_history
    ):

        score += 0.4

    elif (
        "major" in accident_history
    ):

        score -= 1.0

    elif (
        "minor" in accident_history
    ):

        score -= 0.3

    # --------------------------------------------------------
    # Price position
    # --------------------------------------------------------

    minimum_price = PRICE_STATISTICS.get(
        "minimum_lakh"
    )

    maximum_price = PRICE_STATISTICS.get(
        "maximum_lakh"
    )

    if (
        minimum_price is not None
        and
        maximum_price is not None
        and
        maximum_price > minimum_price
    ):

        normalized = (
            predicted_price
            -
            minimum_price
        ) / (
            maximum_price
            -
            minimum_price
        )

        normalized = max(
            0,
            min(
                1,
                normalized
            )
        )

        # High-end cars are not automatically "better".
        # This adjustment is intentionally small.

        score += (
            normalized * 0.5
        )

    # --------------------------------------------------------
    # Clamp score
    # --------------------------------------------------------

    score = max(
        1.0,
        min(
            10.0,
            score
        )
    )

    return round(
        score,
        1
    )


# ============================================================
# 16. PRICE BREAKDOWN
# ============================================================

def calculate_price_breakdown(
    predicted_price,
    data
):
    """
    Generate a transparent UI breakdown.

    NOTE:
    These adjustments are for presentation only.

    The actual predicted price is produced by the ML model.
    """

    predicted_price = float(
        predicted_price
    )

    # --------------------------------------------------------
    # Base price
    # --------------------------------------------------------

    base_price = predicted_price

    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------

    condition = safe_string(
        data.get(
            "condition"
        )
    ).lower()

    if condition == "excellent":

        condition_adjustment = (
            predicted_price * 0.04
        )

    elif condition == "very good":

        condition_adjustment = (
            predicted_price * 0.02
        )

    elif condition == "good":

        condition_adjustment = 0

    elif condition == "fair":

        condition_adjustment = (
            -predicted_price * 0.04
        )

    elif condition == "poor":

        condition_adjustment = (
            -predicted_price * 0.08
        )

    else:

        condition_adjustment = 0

    # --------------------------------------------------------
    # Market demand
    # --------------------------------------------------------

    demand_adjustment = 0

    fuel_type = safe_string(
        data.get(
            "fuel_type"
        )
    ).lower()

    transmission = safe_string(
        data.get(
            "transmission"
        )
    ).lower()

    # Small presentation adjustment only.

    if (
        "automatic" in transmission
    ):

        demand_adjustment += (
            predicted_price * 0.015
        )

    if (
        "petrol" in fuel_type
    ):

        demand_adjustment += (
            predicted_price * 0.005
        )

    # --------------------------------------------------------
    # Calculate displayed components
    # --------------------------------------------------------

    estimated_price = (
        base_price
        +
        condition_adjustment
        +
        demand_adjustment
    )

    # Keep the displayed estimate close to the model result.
    #
    # The model prediction remains the official prediction.

    if estimated_price <= 0:

        estimated_price = predicted_price

    return {

        "base_price_lakh":
            round(
                base_price,
                2
            ),

        "condition_adjustment_lakh":
            round(
                condition_adjustment,
                2
            ),

        "market_demand_adjustment_lakh":
            round(
                demand_adjustment,
                2
            ),

        "displayed_estimated_price_lakh":
            round(
                predicted_price,
                2
            )
    }


# ============================================================
# 17. HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "success": True,

        "status": "online",

        "model_loaded":
            model is not None,

        "project":
            "CarPredict AI"
    })


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(
            url_for("home")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        role = request.form.get("role", "buyer")
        if role not in {"buyer", "seller", "both"}:
            role = "buyer"

        if not username or not email or not password:
            flash(
                "Please fill in all fields.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 6:
            flash(
                "Password must be at least 6 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )

        connection = get_db_connection()

        existing_user = connection.execute(
            """
            SELECT id
            FROM users
            WHERE username = ? OR email = ?
            """,
            (username, email)
        ).fetchone()

        if existing_user:
            connection.close()

            flash(
                "Username or email already exists.",
                "error"
            )

            return render_template(
                "register.html"
            )

        password_hash = generate_password_hash(
            password
        )

        connection.execute(
            """
            INSERT INTO users
            (
                username,
                email,
                password_hash,
                role
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                email,
                password_hash,
                role
            )
        )

        connection.commit()
        connection.close()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )

# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(
            url_for("home")
        )

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        connection = get_db_connection()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        connection.close()

        if user is None:

            flash(
                "Invalid email or password.",
                "error"
            )

            return render_template(
                "login.html"
            )

        if not check_password_hash(
            user["password_hash"],
            password
        ):

            flash(
                "Invalid email or password.",
                "error"
            )

            return render_template(
                "login.html"
            )

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["email"] = user["email"]
        session["role"] = user["role"]

        return redirect(
            url_for("home")
        )

    return render_template(
        "login.html"
    )

# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )
# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():

    if session.get("is_admin"):
        return redirect(
            url_for("admin_dashboard")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):

            session.clear()

            session["is_admin"] = True
            session["admin_username"] = username

            return redirect(
                url_for("admin_dashboard")
            )

        flash(
            "Invalid admin username or password.",
            "error"
        )

    return render_template(
        "admin_login.html"
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin")
@admin_required
def admin_dashboard():

    connection = get_db_connection()

    users = connection.execute(
        """
        SELECT
            id,
            username,
            email,
            created_at
        FROM users
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "admin.html",
        users=users
    )


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():

    session.clear()

    return redirect(
        url_for("admin_login")
    )
# ============================================================
# 18. HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
@login_required
def home():

    return render_template(
        "index.html"
    )

# ============================================================
# 18.1 BUY & SELL MARKETPLACE
# ============================================================

def validate_listing(payload):
    if not isinstance(payload, dict):
        raise ValueError("Please submit a valid listing.")

    text_fields = {
        "seller_name": ("Seller name", 80, True),
        "seller_email": ("Email address", 120, False),
        "seller_phone": ("Phone number", 30, False),
        "make": ("Car make", 80, True),
        "model": ("Car model", 80, True),
        "city": ("City", 80, True),
        "fuel_type": ("Fuel type", 40, False),
        "transmission": ("Transmission", 40, False),
        "condition": ("Condition", 40, False),
        "description": ("Description", 1000, False),
    }
    listing = {}
    for field, (label, limit, required) in text_fields.items():
        value = payload.get(field)
        value = value.strip() if isinstance(value, str) else ""
        if required and not value:
            raise ValueError(f"{label} is required.")
        if len(value) > limit:
            raise ValueError(f"{label} must be {limit} characters or fewer.")
        listing[field] = value or None

    if listing["seller_email"] and not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+", listing["seller_email"]
    ):
        raise ValueError("Enter a valid email address.")

    phone = listing["seller_phone"]
    if phone and (
        not re.fullmatch(r"[\d+\-() ]{7,30}", phone)
        or sum(character.isdigit() for character in phone) < 7
    ):
        raise ValueError("Enter a valid phone number.")
    if not listing["seller_email"] and not phone:
        raise ValueError("Add an email address or phone number so buyers can reach you.")

    for field, label in (
        ("year", "Manufacturing year"),
        ("kilometers", "Kilometers driven"),
        ("price_rupees", "Asking price"),
    ):
        raw_value = "" if payload.get(field) is None else str(payload.get(field)).strip()
        if isinstance(payload.get(field), bool) or not re.fullmatch(r"\d+", raw_value):
            raise ValueError(f"{label} must be a whole number.")
        listing[field] = int(raw_value)

    if not 1950 <= listing["year"] <= 2100:
        raise ValueError("Manufacturing year must be between 1950 and 2100.")
    if not 0 <= listing["kilometers"] <= 5000000:
        raise ValueError("Kilometers driven must be between 0 and 5,000,000.")
    if not 1 <= listing["price_rupees"] <= 1000000000:
        raise ValueError("Asking price is outside the allowed range.")
    return listing


ALLOWED_CAR_PHOTO_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}
MAX_CAR_PHOTO_SIZE = 5 * 1024 * 1024


def parse_listing_photos(photo_json):
    try:
        photos = json.loads(photo_json or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return photos if isinstance(photos, list) else []


def save_listing_photos(files):
    if len(files) > 5:
        raise ValueError("Upload no more than 5 photos.")

    saved_files = []
    try:
        for file in files:
            if not file or not file.filename:
                continue

            extension = secure_filename(file.filename).rsplit(".", 1)[-1].lower()
            if extension not in ALLOWED_CAR_PHOTO_EXTENSIONS:
                raise ValueError("Photos must be JPG, PNG, WEBP, or GIF files.")

            content = file.read(MAX_CAR_PHOTO_SIZE + 1)
            if len(content) > MAX_CAR_PHOTO_SIZE:
                raise ValueError("Each photo must be 5 MB or smaller.")

            signature_is_valid = (
                content.startswith(b"\xff\xd8\xff") and extension in {"jpg", "jpeg"}
            ) or (
                content.startswith(b"\x89PNG\r\n\x1a\n") and extension == "png"
            ) or (
                content.startswith(b"RIFF") and content[8:12] == b"WEBP" and extension == "webp"
            ) or (
                content.startswith((b"GIF87a", b"GIF89a")) and extension == "gif"
            )
            if not signature_is_valid:
                raise ValueError("One of the uploaded files is not a valid image.")

            filename = f"{uuid.uuid4().hex}.{extension}"
            (MARKETPLACE_UPLOAD_DIR / filename).write_bytes(content)
            saved_files.append(filename)
    except ValueError:
        for filename in saved_files:
            (MARKETPLACE_UPLOAD_DIR / filename).unlink(missing_ok=True)
        raise

    return saved_files


@app.route("/uploads/marketplace/<path:filename>")
def marketplace_photo(filename):
    return send_from_directory(str(MARKETPLACE_UPLOAD_DIR), filename)


@app.route("/marketplace")
@login_required
def marketplace():
    return render_template("marketplace.html")


@app.route("/api/listings", methods=["GET", "POST"])
@login_required
def listings():
    connection = sqlite3.connect(str(MARKETPLACE_DATABASE_FILE))
    connection.row_factory = sqlite3.Row
    saved_photo_files = []
    try:
        if request.method == "GET":
            rows = connection.execute(
                "SELECT * FROM car_listings ORDER BY id DESC"
            ).fetchall()
            result = [dict(row) for row in rows]
            for listing in result:
                listing["photos"] = parse_listing_photos(listing.get("photos"))
            return jsonify({
                "success": True,
                "listings": result
            })

        payload = request.form.to_dict() if request.form else request.get_json(silent=True)
        listing = validate_listing(payload)
        uploaded_files = save_listing_photos(request.files.getlist("photos"))
        saved_photo_files = uploaded_files
        listing["photos"] = json.dumps(uploaded_files)
        fields = (
            "seller_name", "seller_email", "seller_phone", "make", "model",
            "year", "kilometers", "price_rupees", "city", "fuel_type",
            "transmission", "condition", "description", "photos",
        )
        cursor = connection.execute(
            f"INSERT INTO car_listings ({', '.join(fields)}) "
            f"VALUES ({', '.join('?' for _ in fields)})",
            tuple(listing[field] for field in fields),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM car_listings WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        result = dict(row)
        result["photos"] = parse_listing_photos(result.get("photos"))
        return jsonify({"success": True, "listing": result}), 201
    except ValueError as error:
        for filename in saved_photo_files:
            (MARKETPLACE_UPLOAD_DIR / filename).unlink(missing_ok=True)
        return jsonify({"success": False, "error": str(error)}), 400
    except sqlite3.Error:
        connection.rollback()
        for filename in saved_photo_files:
            (MARKETPLACE_UPLOAD_DIR / filename).unlink(missing_ok=True)
        app.logger.exception("Could not access marketplace listings")
        return jsonify({"success": False, "error": "Could not save or load listings."}), 500
    finally:
        connection.close()

# ============================================================
# 19. GET WEBSITE OPTIONS
# ============================================================

@app.route(
    "/api/options",
    methods=["GET"]
)
def get_options():

    return jsonify({

        "success": True,

        "options":
            DROPDOWN_OPTIONS,

        "vehicle_hierarchy":
            VEHICLE_HIERARCHY,

        "numeric_ranges":
            NUMERIC_RANGES,

        "metrics":
            MODEL_METRICS,

        "price_statistics":
            PRICE_STATISTICS
    })


# ============================================================
# 20. GET MODELS FOR BRAND
# ============================================================

@app.route(
    "/api/models",
    methods=["GET"]
)
def get_models():

    brand = request.args.get(
        "brand",
        ""
    ).strip()

    if brand == "":

        return jsonify({

            "success": True,

            "models": []
        })

    models = (
        VEHICLE_HIERARCHY
        .get(
            brand,
            {}
        )
        .keys()
    )

    return jsonify({

        "success": True,

        "brand": brand,

        "models": sorted(
            list(models)
        )
    })


# ============================================================
# 21. GET VARIANTS FOR MODEL
# ============================================================

@app.route(
    "/api/variants",
    methods=["GET"]
)
def get_variants():

    brand = request.args.get(
        "brand",
        ""
    ).strip()

    model_name = request.args.get(
        "model",
        ""
    ).strip()

    if (
        brand == ""
        or
        model_name == ""
    ):

        return jsonify({

            "success": True,

            "variants": []
        })

    variants = (
        VEHICLE_HIERARCHY
        .get(
            brand,
            {}
        )
        .get(
            model_name,
            []
        )
    )

    return jsonify({

        "success": True,

        "brand": brand,

        "model": model_name,

        "variants": sorted(
            list(variants)
        )
    })


# ============================================================
# 22. PREDICT PRICE
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict_price():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "ML model is not loaded."
            }), 500

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if data is None:

            return jsonify({

                "success": False,

                "error":
                    "Request body must contain valid JSON."
            }), 400

        # ----------------------------------------------------
        # Validate vehicle hierarchy
        # ----------------------------------------------------

        validate_vehicle_selection(
            data
        )

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        input_df = prepare_prediction_data(
            data
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )

        predicted_price_lakh = float(
            prediction[0]
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not math.isfinite(
            predicted_price_lakh
        ):

            raise ValueError(
                "The model returned an invalid prediction."
            )

        # ----------------------------------------------------
        # Prevent negative prediction
        # ----------------------------------------------------

        predicted_price_lakh = max(
            0,
            predicted_price_lakh
        )

        # ----------------------------------------------------
        # Market score
        # ----------------------------------------------------

        market_score = calculate_market_score(

            predicted_price_lakh,

            data
        )

        # ----------------------------------------------------
        # Price breakdown
        # ----------------------------------------------------

        price_breakdown = calculate_price_breakdown(

            predicted_price_lakh,

            data
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        response = {

            "success": True,

            "prediction": {

                "price_lakh":
                    round(
                        predicted_price_lakh,
                        2
                    ),

                "price_rupees":
                    price_in_rupees(
                        predicted_price_lakh
                    ),

                "formatted_price":
                    format_price_lakh(
                        predicted_price_lakh
                    )
            },

            "market_score":
                market_score,

            "price_breakdown":
                price_breakdown,

            "model": {

                "type":
                    metadata.get(
                        "model_type",
                        "CatBoostRegressor"
                    ),

                "r2_score":
                    MODEL_METRICS.get(
                        "r2_score"
                    ),

                "r2_percentage":
                    MODEL_METRICS.get(
                        "r2_percentage"
                    ),

                "mae_lakh":
                    MODEL_METRICS.get(
                        "mae_lakh"
                    ),

                "rmse_lakh":
                    MODEL_METRICS.get(
                        "rmse_lakh"
                    ),

                "mape_percentage":
                    MODEL_METRICS.get(
                        "mape_percentage"
                    )
            },

            "input": data
        }

        return jsonify(
            response
        )

    except ValueError as error:

        return jsonify({

            "success": False,

            "error":
                str(error)
        }), 400

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        print(
            traceback.format_exc()
        )

        return jsonify({

            "success": False,

            "error":
                "An unexpected error occurred while generating "
                "the prediction.",

            "details":
                str(error)
        }), 500


# ============================================================
# 23. MODEL INFORMATION
# ============================================================

@app.route(
    "/api/model-info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "success": True,

        "project":
            metadata.get(
                "project_name",
                "CarPredict AI"
            ),

        "description":
            metadata.get(
                "project_description",
                ""
            ),

        "model_type":
            metadata.get(
                "model_type",
                "CatBoostRegressor"
            ),

        "dataset_rows":
            metadata.get(
                "dataset_rows"
            ),

        "dataset_columns":
            metadata.get(
                "dataset_columns"
            ),

        "features":
            FEATURE_COLUMNS,

        "metrics":
            MODEL_METRICS,

        "price_statistics":
            PRICE_STATISTICS
    })


# ============================================================
# 24. ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error":
            "The requested API endpoint was not found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({

        "success": False,

        "error":
            "This HTTP method is not allowed for this endpoint."
    }), 405


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "success": False,

        "error":
            "Internal server error."
    }), 500


# ============================================================
# 25. INITIALIZE APPLICATION
# ============================================================

initialize_application()


# ============================================================
# 26. RUN DEVELOPMENT SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("CARPREDICT AI SERVER")
    print("=" * 70)

    print(
        "\nStarting Flask development server..."
    )

    print(
        "\nOpen your browser at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nPress CTRL+C to stop the server."
    )

    print("=" * 70)

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )