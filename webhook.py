from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/v1/hackrx/run', methods=['POST'])
def run_hackrx():
    data = request.json
    # Process the data here
    return jsonify({"status": "success", "data": data}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
