from flask import Flask, request, jsonify
import torch

app = Flask(__name__)


model = TransformerModel(input_size=1, num_heads=4, num_layers=2)
model.load_state_dict(torch.load("model.pth"))  
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  
    signal = np.array(data['signal'])  
    signal_denoised = preprocess_signal(signal)


    signal_tensor = torch.tensor(signal_denoised, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        anomaly_score = model(signal_tensor).item()

    
    return jsonify({"anomaly_score": anomaly_score})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
