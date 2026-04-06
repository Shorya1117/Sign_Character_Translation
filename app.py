from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import torch

from scripts.models import StaticMLP
from scripts.preprocess_images import extract_landmarks, normalize_landmarks

app = Flask(__name__)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model and checkpoint...")
checkpoint = torch.load('models/static_mlp.pth', map_location=device)


class_map = checkpoint['class_map']
idx_to_class = {v: k for k, v in class_map.items()} 
num_classes = len(class_map)
print(f"Loaded {num_classes} classes successfully!")

# Model initialize 
model = StaticMLP(input_dim=42, n_classes=num_classes) 
model.load_state_dict(checkpoint['model_state'])
model.eval()
model.to(device)
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.json['image']
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        
        lm = extract_landmarks(img)
        
        
        if lm is None:
            return jsonify({'prediction': 'nothing', 'success': True})
            
        
        lm_norm = normalize_landmarks(lm)
        lm_flat = lm_norm[:, :2].flatten() 
        
        
        input_tensor = torch.tensor(lm_flat, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            
            
            predicted_char = idx_to_class[predicted_idx]
            
            
            print(f"✅ Prediction: {predicted_char}")

        return jsonify({'prediction': predicted_char, 'success': True})
    
    except Exception as e:
        print(f"❌ [ERROR] {str(e)}")
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    app.run(debug=True)