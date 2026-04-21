import os
import json
import threading
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from optimizer import run_iteration

app = Flask(__name__, static_url_path="", static_folder=".")
CORS(app)

HISTORY_FILE = "history.json"
STATUS_FILE = "status.json"
LATEST_GROUP_FILE = "latest_group.json"

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/api/history")
def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
                # Show everything, just ensure it's a list
                return jsonify(data if isinstance(data, list) else [])
        except Exception as e:
            print(f"History Filter Error: {e}")
    return jsonify([])

@app.route("/api/latest")
def get_latest_group():
    if os.path.exists(LATEST_GROUP_FILE):
        try:
            with open(LATEST_GROUP_FILE) as f:
                data = json.load(f)
                return jsonify(data if isinstance(data, list) else [])
        except Exception as e:
            print(f"Latest Group Filter Error: {e}")
    return jsonify([])

@app.route("/api/status")
def get_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                return jsonify(json.load(f))
        except: pass
    return jsonify({"message": "System Idle", "progress": 0, "busy": False})

def worker_task(it_num):
    try:
        run_iteration(it_num)
    except Exception as e:
        print(f"Worker task failed: {e}")
        with open(STATUS_FILE, "w") as f:
            json.dump({"message": f"Error: {str(e)}", "progress": 0, "busy": False}, f)

@app.route("/api/iterate", methods=["POST"])
def iterate():
    # Check if busy
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                curr = json.load(f)
                if curr.get("busy"):
                    return jsonify({"error": "Optimizer is already running"}), 400
        except: pass

    it_num = 1
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                h = json.load(f)
                it_num = len(h) + 1
        except: pass
            
    # Start in background thread to keep Flask responsive
    thread = threading.Thread(target=worker_task, args=(it_num,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started", "iteration": it_num})

if __name__ == "__main__":
    # ONLY clear status if we aren't already running in another process
    is_busy = False
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                is_busy = json.load(f).get("busy", False)
        except: pass
    
    if not is_busy:
        if os.path.exists(STATUS_FILE): os.remove(STATUS_FILE)
        if os.path.exists(LATEST_GROUP_FILE): os.remove(LATEST_GROUP_FILE)
    
    app.run(port=8080, debug=False)
