
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import subprocess
import os
import signal
import json
from datetime import datetime

app = Flask(__name__)

PARKING_LOTS = {
    "Lot A": "parking1.mp4",
    "Lot B": "parking2.mp4",
    "Lot C": "parking3.mp4"
}

status_file = "status.json"
active_process = None
active_lot = None

# Load parking status
def load_status():
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return json.load(f)
    return {lot: {"occupied": 0, "available": 0, "last_updated": ""} for lot in PARKING_LOTS}

def save_status(status):
    with open(status_file, "w") as f:
        json.dump(status, f, indent=4)

# Home page: choose parking lot
@app.route("/", methods=["GET", "POST"])
def choose_lot():
    global active_process, active_lot

    if request.method == "POST":
        selected_lot = request.form["lot"]

        # Stop previous detection if running
        if active_process:
            os.kill(active_process.pid, signal.SIGTERM)

        # Start new detection for selected lot
        video_file = PARKING_LOTS[selected_lot]
        active_process = subprocess.Popen(["python", "main.py", selected_lot, video_file])
        active_lot = selected_lot

        return redirect(url_for("dashboard", lot=selected_lot))

    return render_template_string("""
    <h1>Select Parking Lot</h1>
    <form method="post">
        <select name="lot">
            {% for lot in lots %}
                <option value="{{ lot }}">{{ lot }}</option>
            {% endfor %}
        </select>
        <button type="submit">Start</button>
    </form>
    """, lots=PARKING_LOTS.keys())

# Dashboard for live data
@app.route("/dashboard")
def dashboard():
    selected_lot = request.args.get("lot", list(PARKING_LOTS.keys())[0])
    return render_template_string("""
    <h1>Dashboard - {{ selected }}</h1>
    <a href="/">← Choose another lot</a>
    <div id="stats">
        <p><strong>Occupied:</strong> <span id="occupied">Loading...</span></p>
        <p><strong>Available:</strong> <span id="available">Loading...</span></p>
        <p><strong>Last Updated:</strong> <span id="last_updated">Loading...</span></p>
    </div>

    <script>
        const selectedLot = "{{ selected }}";
        async function fetchStatus() {
            const res = await fetch(`/status?lot=${selectedLot}`);
            const data = await res.json();
            document.getElementById("occupied").innerText = data.occupied;
            document.getElementById("available").innerText = data.available;
            document.getElementById("last_updated").innerText = data.last_updated;
        }
        fetchStatus();
        
    </script>
    """, selected=selected_lot)

# Return live JSON data
@app.route("/status")
def get_status():
    lot = request.args.get("lot", list(PARKING_LOTS.keys())[0])
    status = load_status()
    return jsonify(status.get(lot, {
        "occupied": 0,
        "available": 0,
        "last_updated": ""
    }))

# API for main.py updates
@app.route("/update", methods=["POST"])
def update():
    data = request.get_json()
    lot = data.get("lot", "Lot A")
    status = load_status()
    status[lot] = {
        "occupied": data.get("occupied", 0),
        "available": data.get("available", 0),
        "last_updated": data.get("last_updated", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    }
    save_status(status)
    return jsonify({"message": "Updated successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True)
