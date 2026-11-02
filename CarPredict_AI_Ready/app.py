from flask import Flask, render_template, request, jsonify
from pathlib import Path
import json, math, re, sqlite3
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / 'model' / 'car_price_model.joblib'
META_FILE = BASE_DIR / 'model' / 'metadata.json'
LISTINGS_DB = BASE_DIR / 'marketplace.db'
app = Flask(__name__)
model = None
metadata = {}
FEATURES = ['brand','model','variant','body_type','manufacturing_year','vehicle_age','registration_year','kilometers_driven','fuel_type','transmission','engine_cc','max_power_bhp','mileage_kmpl','seats','ownership','seller_type','city','state','color','condition','insurance_type','service_history','accident_history','warranty_available','tyre_condition','features_count']
NUMERIC = {'manufacturing_year','vehicle_age','registration_year','kilometers_driven','engine_cc','max_power_bhp','mileage_kmpl','seats','features_count'}
REQUIRED = ['brand','model','variant','body_type','manufacturing_year','kilometers_driven','fuel_type','transmission']

def load_artifacts():
    global model, metadata
    model=None; metadata={}
    if MODEL_FILE.exists():
        try: model=joblib.load(MODEL_FILE)
        except Exception as e: print('[ERROR] Model load failed:', e)
    if META_FILE.exists():
        try: metadata=json.loads(META_FILE.read_text(encoding='utf-8'))
        except Exception as e: print('[ERROR] Metadata load failed:', e)
load_artifacts()

def initialize_listings_db():
    connection = sqlite3.connect(LISTINGS_DB)
    try:
        connection.execute('''
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        connection.commit()
    finally:
        connection.close()

initialize_listings_db()

@app.route('/')
def home(): return render_template('index.html')

@app.route('/marketplace')
def marketplace(): return render_template('marketplace.html')

def validate_listing(payload):
    if not isinstance(payload, dict):
        raise ValueError('Please submit a valid listing.')

    text_fields = {
        'seller_name': ('Seller name', 80, True),
        'seller_email': ('Email address', 120, False),
        'seller_phone': ('Phone number', 30, False),
        'make': ('Car make', 80, True),
        'model': ('Car model', 80, True),
        'city': ('City', 80, True),
        'fuel_type': ('Fuel type', 40, False),
        'transmission': ('Transmission', 40, False),
        'condition': ('Condition', 40, False),
        'description': ('Description', 1000, False),
    }
    listing = {}
    for field, (label, limit, required) in text_fields.items():
        value = payload.get(field)
        value = value.strip() if isinstance(value, str) else ''
        if required and not value:
            raise ValueError(f'{label} is required.')
        if len(value) > limit:
            raise ValueError(f'{label} must be {limit} characters or fewer.')
        listing[field] = value or None

    email = listing['seller_email']
    if email and not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email):
        raise ValueError('Enter a valid email address.')

    phone = listing['seller_phone']
    if phone and (
        not re.fullmatch(r'[\d+\-() ]{7,30}', phone)
        or sum(character.isdigit() for character in phone) < 7
    ):
        raise ValueError('Enter a valid phone number.')
    if not email and not phone:
        raise ValueError('Provide an email address or phone number so buyers can contact you.')

    for field, label in (
        ('year', 'Manufacturing year'),
        ('kilometers', 'Kilometers driven'),
        ('price_rupees', 'Asking price'),
    ):
        value = payload.get(field)
        raw_value = '' if value is None else str(value).strip()
        if isinstance(value, bool) or not re.fullmatch(r'\d+', raw_value):
            raise ValueError(f'{label} must be a whole number.')
        listing[field] = int(raw_value)

    if not 1950 <= listing['year'] <= 2100:
        raise ValueError('Manufacturing year must be between 1950 and 2100.')
    if not 0 <= listing['kilometers'] <= 5000000:
        raise ValueError('Kilometers driven must be between 0 and 5,000,000.')
    if not 1 <= listing['price_rupees'] <= 1000000000:
        raise ValueError('Asking price must be between ₹1 and ₹1,00,00,00,000.')
    return listing

@app.route('/api/listings', methods=['GET', 'POST'])
def listings():
    if request.method == 'GET':
        connection = sqlite3.connect(LISTINGS_DB)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                'SELECT * FROM car_listings ORDER BY id DESC'
            ).fetchall()
            return jsonify({'success': True, 'listings': [dict(row) for row in rows]})
        except sqlite3.Error as e:
            app.logger.exception('Could not load marketplace listings: %s', e)
            return jsonify({'success': False, 'error': 'Could not load car listings.'}), 500
        finally:
            connection.close()

    try:
        listing = validate_listing(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    fields = (
        'seller_name', 'seller_email', 'seller_phone', 'make', 'model',
        'year', 'kilometers', 'price_rupees', 'city', 'fuel_type',
        'transmission', 'condition', 'description',
    )
    connection = sqlite3.connect(LISTINGS_DB)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            f"INSERT INTO car_listings ({', '.join(fields)}) "
            f"VALUES ({', '.join('?' for _ in fields)})",
            tuple(listing[field] for field in fields),
        )
        connection.commit()
        row = connection.execute(
            'SELECT * FROM car_listings WHERE id = ?', (cursor.lastrowid,)
        ).fetchone()
        return jsonify({'success': True, 'listing': dict(row)}), 201
    except sqlite3.Error as e:
        connection.rollback()
        app.logger.exception('Could not save marketplace listing: %s', e)
        return jsonify({'success': False, 'error': 'Could not save your listing.'}), 500
    finally:
        connection.close()

@app.route('/api/status')
def status():
    return jsonify({'success':True,'model_loaded':model is not None,'accuracy_percent':metadata.get('accuracy_percent'),'r2':metadata.get('r2'),'mae_lakh':metadata.get('mae_lakh'),'training_rows':metadata.get('training_rows'),'model_name':metadata.get('model_name','HistGradientBoostingRegressor')})

@app.route('/api/options')
def options():
    return jsonify({'success':True,'options':metadata.get('options',{}),'brand_to_models':metadata.get('brand_to_models',{}),'model_to_variants':metadata.get('model_to_variants',{})})

def clean(field, value):
    if value in (None,''): return None
    if field in NUMERIC:
        try:
            n=float(value)
            if not math.isfinite(n): return None
            return int(n) if field in {'manufacturing_year','vehicle_age','registration_year','kilometers_driven','seats','features_count'} else n
        except (TypeError,ValueError): return None
    return str(value).strip() or None

def make_frame(payload):
    row={f:clean(f,payload.get(f)) for f in FEATURES}
    row['accident_history']=row.get('accident_history') or 'No Record'
    if row.get('vehicle_age') is None and row.get('manufacturing_year') is not None:
        row['vehicle_age']=max(0,2026-int(row['manufacturing_year']))
    missing=[f for f in REQUIRED if row.get(f) in (None,'')]
    if missing: raise ValueError('Please fill these required fields: '+', '.join(missing))
    if not 1980 <= row['manufacturing_year'] <= 2035: raise ValueError('Manufacturing year must be between 1980 and 2035.')
    if row.get('registration_year') is not None:
        if not 1980 <= row['registration_year'] <= 2035: raise ValueError('Registration year must be between 1980 and 2035.')
        if row['registration_year'] < row['manufacturing_year']: raise ValueError('Registration year cannot be earlier than manufacturing year.')
    if row['kilometers_driven'] < 0: raise ValueError('Kilometers driven cannot be negative.')
    return pd.DataFrame([row], columns=FEATURES)

def market_score(row):
    score=7.6
    if row.get('condition')=='Excellent': score+=0.6
    elif row.get('condition')=='Fair': score-=0.8
    if row.get('ownership')=='1st Owner': score+=0.4
    elif row.get('ownership')=='3rd Owner': score-=0.35
    elif row.get('ownership')=='4th+ Owner': score-=0.65
    if row.get('service_history')=='Full Service History': score+=0.4
    elif row.get('service_history')=='No Service History': score-=0.45
    if row.get('tyre_condition')=='Excellent': score+=0.2
    elif row.get('tyre_condition')=='Needs Replacement': score-=0.5
    if row.get('warranty_available')=='Yes': score+=0.2
    km=row.get('kilometers_driven',0)
    if km<30000: score+=0.35
    elif km>120000: score-=0.55
    elif km>80000: score-=0.25
    if row.get('accident_history')=='Major': score-=0.7
    elif row.get('accident_history')=='Minor': score-=0.15
    return max(1,min(10,round(score,1)))

def breakdown(price,row):
    c=0.35 if row.get('condition')=='Excellent' else (-0.35 if row.get('condition')=='Fair' else 0)
    h=0.20 if row.get('service_history')=='Full Service History' else (-0.20 if row.get('service_history')=='No Service History' else 0)
    o=0.20 if row.get('ownership')=='1st Owner' else (-0.20 if row.get('ownership')=='3rd Owner' else (-0.30 if row.get('ownership')=='4th+ Owner' else 0))
    return {'base_price_lakh':round(max(0.01,price-c-h-o),2),'condition_adjustment_lakh':round(c,2),'history_adjustment_lakh':round(h+o,2)}

@app.route('/api/predict',methods=['POST'])
def predict():
    if model is None: return jsonify({'success':False,'error':'Model is not trained. Run python train_model.py and restart the server.'}),503
    try:
        X=make_frame(request.get_json(silent=True) or {})
        price=max(0,round(float(model.predict(X)[0]),2))
        row=X.iloc[0].to_dict(); score=market_score(row); b=breakdown(price,row)
        if score>=8.5: msg,desc='Excellent Choice!','This car has strong value in the current market.'
        elif score>=7: msg,desc='Great Choice!','This car holds good value in the current market.'
        elif score>=5.5: msg,desc='Good Value','The estimated price is reasonable for these inputs.'
        else: msg,desc='Review Carefully','Check condition, history and comparable listings.'
        return jsonify({'success':True,'price_lakh':price,'price_rupees':int(round(price*100000)),'score':score,'message':msg,'description':desc,'vehicle_age':row.get('vehicle_age'),'kilometers_driven':row.get('kilometers_driven'),'model_r2':metadata.get('r2'),**b})
    except ValueError as e: return jsonify({'success':False,'error':str(e)}),400
    except Exception as e:
        print('[ERROR] Prediction:',e)
        return jsonify({'success':False,'error':'Prediction failed. Check the VS Code terminal.'}),500

if __name__=='__main__':
    print('='*68); print('CarPredict AI'); print('Open: http://127.0.0.1:5000'); print('='*68)
    app.run(host='127.0.0.1',port=5000,debug=True)
