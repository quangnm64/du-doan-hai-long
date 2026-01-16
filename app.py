from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# Load model và scaler
MODEL_PATH = 'models/customer_satisfaction_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.json'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = json.load(f)
except FileNotFoundError:
    print("Warning: Model files not found. Please run train_model.py first.")
    model = None
    scaler = None
    feature_names = []

@app.route('/')
def index():
    """Trang chủ - Giới thiệu dự án"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Trang nhập dữ liệu để dự đoán"""
    return render_template('predict.html')

@app.route('/analysis')
def analysis_page():
    """Trang phân tích dữ liệu và mô hình"""
    return render_template('analysis.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán mức độ hài lòng"""
    if model is None:
        return jsonify({'error': 'Model chưa được huấn luyện. Vui lòng chạy train_model.py trước.'}), 500
    
    try:
        data = request.json
        
        # Tạo DataFrame từ dữ liệu đầu vào
        input_data = pd.DataFrame([{
            'Age': float(data.get('Age', 0)),
            'Flight_Distance': float(data.get('Flight_Distance', 0)),
            'Inflight_wifi_service': int(data.get('Inflight_wifi_service', 0)),
            'Departure_Arrival_time_convenient': int(data.get('Departure_Arrival_time_convenient', 0)),
            'Ease_of_Online_booking': int(data.get('Ease_of_Online_booking', 0)),
            'Gate_location': int(data.get('Gate_location', 0)),
            'Food_and_drink': int(data.get('Food_and_drink', 0)),
            'Online_boarding': int(data.get('Online_boarding', 0)),
            'Seat_comfort': int(data.get('Seat_comfort', 0)),
            'Inflight_entertainment': int(data.get('Inflight_entertainment', 0)),
            'On_board_service': int(data.get('On_board_service', 0)),
            'Leg_room_service': int(data.get('Leg_room_service', 0)),
            'Baggage_handling': int(data.get('Baggage_handling', 0)),
            'Checkin_service': int(data.get('Checkin_service', 0)),
            'Inflight_service': int(data.get('Inflight_service', 0)),
            'Cleanliness': int(data.get('Cleanliness', 0)),
            'Gender_Male': 1 if data.get('Gender') == 'Male' else 0,
            'Customer_Type_Loyal Customer': 1 if data.get('Customer_Type') == 'Loyal Customer' else 0,
            'Type_of_Travel_Business travel': 1 if data.get('Type_of_Travel') == 'Business travel' else 0,
            'Class_Business': 1 if data.get('Class') == 'Business' else 0,
            'Class_Eco Plus': 1 if data.get('Class') == 'Eco Plus' else 0,
        }])
        
        # Đảm bảo thứ tự features đúng
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        
        # Chuẩn hóa dữ liệu
        input_scaled = scaler.transform(input_data)
        
        # Dự đoán
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = {
            'prediction': 'Hài lòng' if prediction == 1 else 'Không hài lòng',
            'probability_satisfied': float(probability[1]),
            'probability_unsatisfied': float(probability[0]),
            'confidence': float(max(probability))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """API endpoint để lấy thông tin về mô hình"""
    try:
        if model is None:
            return jsonify({'error': 'Model chưa được huấn luyện'}), 404
        
        # Lấy feature importance nếu có
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances.tolist()))
            # Sắp xếp theo độ quan trọng
            feature_importance = dict(sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'model_type': type(model).__name__,
            'feature_importance': feature_importance,
            'num_features': len(feature_names)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
