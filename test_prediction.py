# #!/usr/bin/env python
# """
# Simple test - just run with: python test_prediction.py
# """

# from predict_crop import predict_crop

# # Test with your blackgram parameters
# print("="*60)
# print("CROP PREDICTION TEST")
# print("="*60)

# # Your input values

# input_data = {
#     'N': 90,
#     'P': 40,
#     'K': 40,
#     'temperature': 24,
#     'humidity': 80,
#     'ph': 6.0,
#     'rainfall': 200
# }

# nitrogen = input_data['N']
# phosphorus = input_data['P']
# potassium = input_data['K']
# temperature = input_data['temperature']
# humidity = input_data['humidity']
# ph = input_data['ph']
# rainfall = input_data['rainfall']

# print(f"\nInput Parameters:")
# print(f"  Nitrogen: {nitrogen}")
# print(f"  Phosphorus: {phosphorus}")
# print(f"  Potassium: {potassium}")
# print(f"  Temperature: {temperature}°C")
# print(f"  Humidity: {humidity}%")
# print(f"  pH: {ph}")
# print(f"  Rainfall: {rainfall}mm")

# # Get prediction
# crop, confidence = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

# print(f"\n" + "="*60)
# print(f"RESULT")
# print(f"="*60)
# print(f"Predicted Crop: {crop.upper()}")
# print(f"Confidence: {confidence:.2f}%")
# print("="*60)
