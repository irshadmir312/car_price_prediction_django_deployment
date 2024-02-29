from django.shortcuts import render
import joblib
import pandas as pd

# Load the models
loaded_rf_model = joblib.load('/Users/apple/Desktop/CARPRICEPRED/MODEL DEVELOPMENT/rf_regressor.joblib')
loaded_knn_model = joblib.load('/Users/apple/Desktop/CARPRICEPRED/MODEL DEVELOPMENT/knn_regressor.joblib')
loaded_gb_model = joblib.load('/Users/apple/Desktop/CARPRICEPRED/MODEL DEVELOPMENT/gb_regressor.joblib')
loaded_ridge_model = joblib.load('/Users/apple/Desktop/CARPRICEPRED/MODEL DEVELOPMENT/ridge_regressor.joblib')

# Create your views here.
def home(request):
    context = {}
    if request.method == 'POST':
        name = request.POST.get('name')
        fuel_type = request.POST.get('fuel-type')
        year_of_make = request.POST.get('year_of_make')
        kim = request.POST.get('kim')
        power = request.POST.get('power')
        data = {
            'name': [name],
            'fuel-type': [fuel_type],
            'year_of_make': [year_of_make],
            'kim': [kim],
            'power': [power]
        }
        df = pd.DataFrame(data)
        df['name'] = df['name'].apply(hash)
        df['fuel-type'] = df['fuel-type'].apply(hash)
        rf_res = loaded_rf_model.predict(df)
        knn_res = loaded_knn_model.predict(df)
        gb_res = loaded_gb_model.predict(df)
        ridge_res = loaded_ridge_model.predict(df)
        context['rf_res'] = rf_res[0]
        context['knn_res'] = knn_res[0]
        context['gb_res'] = gb_res[0]
        context['ridge_res'] = ridge_res[0]

    return render(request, "home.html", context)
