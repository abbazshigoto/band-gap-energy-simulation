from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import math

# Initialize the Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Data Storage ---
EXPERIMENTAL_DATA_LOG = []

# --- Routes (Serving HTML Pages and Static Files) ---

@app.route('/')
def home():
    """Renders the main simulation page (index.html)."""
    return render_template('index.html')

@app.route('/readings_graphs.html')
def readings_graphs():
    """Renders the results and graphing page."""
    return render_template('readings_graphs.html')

@app.route('/static/images/<path:filename>')
def serve_images(filename):
    """Safely serves the image files requested by the front-end."""
    # Note: Ensure all image files referenced in index.html are present in static/images
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'), filename)

# --- API Endpoints (Data Handling) ---

@app.route('/api/log_data', methods=['POST'])
def log_data():
    """Endpoint for the front-end to submit experimental readings."""
    try:
        data = request.get_json()
        
        # Validation and conversion
        data['temperature'] = float(data.get('temperature'))
        data['current'] = float(data.get('current'))
        data['voltage'] = float(data.get('voltage'))
        
        EXPERIMENTAL_DATA_LOG.append(data)
        
        print(f"Data logged: T={data['temperature']:.1f}°C, I={data['current']:.4f}A")
        return jsonify({"message": "Data logged successfully", "id": len(EXPERIMENTAL_DATA_LOG) - 1}), 201

    except Exception as e:
        return jsonify({"error": f"Failed to log data: {str(e)}"}), 500

@app.route('/api/get_all_data', methods=['GET'])
def get_all_data():
    """Endpoint to retrieve all logged data."""
    return jsonify({"data": EXPERIMENTAL_DATA_LOG, "count": len(EXPERIMENTAL_DATA_LOG)})

@app.route('/api/calculate_band_gap', methods=['GET'])
def calculate_band_gap():
    """Calculates the band gap (Eg) from the logged data using linear regression (NumPy)."""
    
    X_inv_T = [] # 1/T (Inverse Absolute Temperature in K^-1)
    Y_ln_I = []  # ln(I) (Natural Log of Current)

    for data in EXPERIMENTAL_DATA_LOG:
        T_kelvin = data['temperature'] + 273.15
        current = data['current']
        
        # Current must be non-zero (or above a small tolerance)
        if current > 1e-9:
            X_inv_T.append(1 / T_kelvin)
            Y_ln_I.append(math.log(current))

    if len(X_inv_T) < 2:
        return jsonify({"error": "Not enough valid data points for calculation (need at least 2 non-zero current readings)."}), 400

    try:
        # Perform Linear Regression: returns the coefficients [slope, intercept]
        slope, intercept = np.polyfit(X_inv_T, Y_ln_I, 1)
        
        # Boltzmann constant in electron-volts per Kelvin (eV/K)
        k_B = 8.617e-5 

        # Eg = -2 * kB * Slope
        Eg = -2 * k_B * slope
        
        # Calculate R-squared value for fit quality
        correlation_matrix = np.corrcoef(X_inv_T, Y_ln_I)
        r_squared = correlation_matrix[0,1]**2

    except Exception as e:
        return jsonify({"error": f"Linear fit failed: {str(e)}"}), 500

    return jsonify({
        "message": "Calculation successful",
        "band_gap_eV": float(Eg),
        "r_squared": float(r_squared),
        "slope_value": float(slope),
        "regression_points": {
            "inv_T": [float(x) for x in X_inv_T],
            "ln_I": [float(y) for y in Y_ln_I]
        }
    })

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5500)