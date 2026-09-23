CARPREDICT AI - RUN STEPS

1. Open this folder in VS Code.
2. Open the terminal in this exact folder.
3. Create venv: python -m venv venv
4. Activate: venv\Scripts\activate
5. Install: python -m pip install -r requirements.txt
6. Train: python train_model.py
7. Run: python app.py
8. Open: http://127.0.0.1:5000

The CSV is already copied as car_data.csv, so the earlier Dataset not found error is fixed.
The model predicts selling_price_lakh and does not use target-leakage columns such as original_price_lakh, estimated_monthly_emi, price_per_km, depreciation_percent or listing_id as user inputs.
