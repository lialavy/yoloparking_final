# app.py - Main application file with correct image/video workflow

from flask import Flask, jsonify, request, render_template, redirect, url_for, session, Response, flash, send_file
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import os
import json
import threading
import time
import numpy as np
from functools import wraps
from parking_management import ParkingManagement, ParkingPtsSelection
import subprocess

# Configuration
USER_FILE = 'users.json'
PARKING_LOTS_FILE = 'parking_lots.json'
DEVICE = "cuda"  # Change to "cpu" if no GPU available
ADMIN_EMAIL = "lia.ninio24@gmail.com"  # Admin email

app = Flask(__name__, static_folder='static')
app.secret_key = 'secret_key_here'  # In production, use a secure random key
app.permanent_session_lifetime = timedelta(days=7)

# Shared resources
latest_frames = {}
frame_lock = threading.Lock()
parking_threads = {}
stop_events = {}

# Data storage
parking_status = {}
vehicle_logs = {}
active_parking_lot = None

# User management
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users_data):
    with open(USER_FILE, 'w') as f:
        json.dump(users_data, f)

# Parking lots management
def load_parking_lots():
    if os.path.exists(PARKING_LOTS_FILE):
        with open(PARKING_LOTS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_parking_lots(parking_lots_data):
    with open(PARKING_LOTS_FILE, 'w') as f:
        json.dump(parking_lots_data, f)

def initialize_parking_lots():
    # Default parking lots if none exists
    if not os.path.exists(PARKING_LOTS_FILE):
        default_lots = {
            "main_lot": {
                "name": "Main Parking Lot",
                "video_source": "carPark.mp4",
                "image_source": "carPark.jpg",  # Image used for bbox selection
                "model_path": "best (4).pt",
                "bounding_boxes": "bounding_boxes.json"
            }
        }
        save_parking_lots(default_lots)
    
    # Load parking lots
    lots = load_parking_lots()
    
    # Initialize status and logs for each lot
    global active_parking_lot
    for lot_id, lot_info in lots.items():
        if active_parking_lot is None:
            active_parking_lot = lot_id
            
        parking_status[lot_id] = {
            'occupied': 0,
            'available': 0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if lot_id not in vehicle_logs:
            vehicle_logs[lot_id] = []
    
    return lots

users = load_users()
parking_lots = initialize_parking_lots()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        
        username = session['user']
        user = users.get(username)
        
        # For debugging
        print(f"Admin check for user {username}: {user}")
        
        # Check if user exists
        if not user:
            flash("User not found")
            return redirect(url_for('homepage'))
            
        # Check email directly and normalize it for comparison
        user_email = user.get('email', '').strip().lower()
        admin_email = ADMIN_EMAIL.strip().lower()
        
        # For debugging
        print(f"Comparing emails: '{user_email}' vs '{admin_email}'")
        
        if user_email != admin_email:
            flash(f"Administrator access required. Your email: {user_email}")
            return redirect(url_for('homepage'))
        
        return f(*args, **kwargs)
    return decorated_function

# Update parking status
def update_parking_status(lot_id, occupied, available):
    parking_status[lot_id] = {
        'occupied': occupied,
        'available': available,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    vehicle_logs[lot_id].append({
        'time': parking_status[lot_id]['last_updated'],
        'occupied': occupied,
        'available': available
    })
    print(f"Status updated for {lot_id}: {occupied} occupied, {available} available")

# Background parking detection process
def run_parking_detection(lot_id, stop_event):
    global latest_frames
    
    lot_info = parking_lots[lot_id]
    video_source = lot_info["video_source"]  # Use video for detection
    model_path = lot_info["model_path"]
    bounding_boxes = lot_info["bounding_boxes"]  # These were created from the image
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_source}")
        return
    
    # Initialize parking management
    try:
        parkingmanager = ParkingManagement(
            model=model_path,
            json_file=bounding_boxes,
            device=DEVICE
        )
    except Exception as e:
        print(f"Error initializing parking management: {e}")
        cap.release()
        return
    
    # Track last known parking status
    last_occupied = -1
    last_available = -1
    
    print(f"Parking detection started for {lot_id}")
    
    # Main processing loop
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Reset video to beginning when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Process frame with parking manager
        try:
            results = parkingmanager(frame)
            processed_frame = results.plot_im
            
            # Update latest frame for video feed
            with frame_lock:
                latest_frames[lot_id] = processed_frame.copy()
            
            # Get current occupancy info
            current_occupied = parkingmanager.pr_info.get("Occupancy", 0)
            current_available = parkingmanager.pr_info.get("Available", 0)
            
            # Update status if changed
            if current_occupied != last_occupied or current_available != last_available:
                update_parking_status(lot_id, current_occupied, current_available)
                last_occupied = current_occupied
                last_available = current_available
        
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        # Brief sleep to prevent CPU overuse
        time.sleep(0.01)
    
    cap.release()
    print(f"Parking detection stopped for {lot_id}")

# Start parking detection for a specific lot
def start_parking_detection(lot_id):
    if lot_id in parking_threads and parking_threads[lot_id].is_alive():
        print(f"Parking detection already running for {lot_id}")
        return
    
    # Create stop event
    stop_events[lot_id] = threading.Event()
    
    # Create and start thread
    parking_threads[lot_id] = threading.Thread(
        target=run_parking_detection,
        args=(lot_id, stop_events[lot_id]),
        daemon=True
    )
    parking_threads[lot_id].start()

# Stop parking detection for a specific lot
def stop_parking_detection(lot_id):
    if lot_id in stop_events:
        stop_events[lot_id].set()
        if lot_id in parking_threads and parking_threads[lot_id].is_alive():
            parking_threads[lot_id].join(timeout=1.0)
            print(f"Parking detection stopped for {lot_id}")

# Video feed generator for web interface
def generate_frames(lot_id):
    while True:
        with frame_lock:
            if lot_id not in latest_frames or latest_frames[lot_id] is None:
                # If no frame is available yet, return a blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                frame_bytes = buffer.tobytes()
            else:
                # Return the latest processed frame
                _, buffer = cv2.imencode('.jpg', latest_frames[lot_id])
                frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Sleep briefly to control frame rate
        time.sleep(0.03)  # ~30 FPS

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session.permanent = True
            session['user'] = username
            return redirect(url_for('homepage'))
        error = 'Invalid username or password. Please try again.'

    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email'].strip()  # Strip whitespace

        if username in users:
            error = 'User already exists. Please choose a different username or login.'
        else:
            # Store email and explicitly set admin status
            is_admin = email.lower() == ADMIN_EMAIL.lower()
            users[username] = {
                'password': generate_password_hash(password),
                'email': email,
                'is_admin': is_admin
            }
            save_users(users)
            
            if is_admin:
                print(f"Admin user created: {username} with email {email}")
                
            return redirect(url_for('login'))

    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    user = session.pop('user', None)
    return render_template('logout.html', user=user)

@app.route('/')
@login_required
def homepage():
    is_admin = False
    if 'user' in session:
        user = users.get(session['user'])
        if user and user.get('email', '').lower() == ADMIN_EMAIL.lower():
            is_admin = True
            
    return render_template('home.html', is_admin=is_admin)

@app.route('/set_active_lot', methods=['POST'])
@login_required
def set_active_lot():
    lot_id = request.form.get('lot_id')
    if lot_id in parking_lots:
        global active_parking_lot
        active_parking_lot = lot_id
        
        # Ensure detection is running for this lot
        start_parking_detection(lot_id)
        
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    global active_parking_lot
    
    # Start detection for active lot if not already running
    if active_parking_lot:
        start_parking_detection(active_parking_lot)
    
    return render_template(
        'dashboard.html', 
        parking_lots=parking_lots,
        active_lot=active_parking_lot,
        active_lot_name=parking_lots.get(active_parking_lot, {}).get('name', 'Unknown')
    )

@app.route('/history')
@login_required
def history():
    global active_parking_lot
    
    log = []
    if active_parking_lot and active_parking_lot in vehicle_logs:
        log = vehicle_logs[active_parking_lot][::-1]  # Reverse for newest first
    
    return render_template(
        'history.html', 
        log=log, 
        parking_lots=parking_lots,
        active_lot=active_parking_lot,
        active_lot_name=parking_lots.get(active_parking_lot, {}).get('name', 'Unknown')
    )

@app.route('/video_feed')
@login_required
def video_feed():
    global active_parking_lot
    if not active_parking_lot:
        # If no active lot, return first available
        if parking_lots:
            active_parking_lot = next(iter(parking_lots))
    
    return Response(
        generate_frames(active_parking_lot), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
@login_required
def get_status():
    global active_parking_lot
    if active_parking_lot and active_parking_lot in parking_status:
        return jsonify(parking_status[active_parking_lot])
    return jsonify({'occupied': 0, 'available': 0, 'last_updated': 'N/A'})

@app.route('/update', methods=['POST'])
def update_status():
    data = request.json
    lot_id = data.get('lot_id', active_parking_lot)
    
    try:
        occupied = data.get('occupied')
        available = data.get('available')
        if occupied is not None and available is not None and lot_id in parking_lots:
            update_parking_status(lot_id, occupied, available)
            return jsonify({"message": "Status updated."}), 200
        else:
            return jsonify({"error": "Invalid data or parking lot."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Admin routes
@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin.html', parking_lots=parking_lots)

@app.route('/admin/add_parking_lot', methods=['GET', 'POST'])
@admin_required
def add_parking_lot():
    if request.method == 'POST':
        lot_id = request.form['lot_id']
        name = request.form['name']
        video_source = request.form['video_source']
        image_source = request.form['image_source']  # Image for bbox selection
        model_path = request.form['model_path']
        bounding_boxes = request.form['bounding_boxes']
        
        # Validate inputs
        if not all([lot_id, name, video_source, image_source, model_path, bounding_boxes]):
            flash("All fields are required")
            return redirect(url_for('add_parking_lot'))
        
        # Check if lot_id already exists
        if lot_id in parking_lots:
            flash(f"Parking lot ID '{lot_id}' already exists")
            return redirect(url_for('add_parking_lot'))
        
        # Add new parking lot
        parking_lots[lot_id] = {
            "name": name,
            "video_source": video_source,
            "image_source": image_source,  # Image used only for bbox selection
            "model_path": model_path,
            "bounding_boxes": bounding_boxes
        }
        
        # Initialize status and logs
        parking_status[lot_id] = {
            'occupied': 0,
            'available': 0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        vehicle_logs[lot_id] = []
        
        # Save to file
        save_parking_lots(parking_lots)
        
        flash(f"Parking lot '{name}' added successfully")
        return redirect(url_for('admin_dashboard'))
    
    return render_template('add_parking_lot.html')

@app.route('/admin/edit_parking_lot/<lot_id>', methods=['GET', 'POST'])
@admin_required
def edit_parking_lot(lot_id):
    if lot_id not in parking_lots:
        flash(f"Parking lot '{lot_id}' not found")
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        name = request.form['name']
        video_source = request.form['video_source']
        image_source = request.form['image_source']  # Image for bbox selection
        model_path = request.form['model_path']
        bounding_boxes = request.form['bounding_boxes']
        
        # Validate inputs
        if not all([name, video_source, image_source, model_path, bounding_boxes]):
            flash("All fields are required")
            return redirect(url_for('edit_parking_lot', lot_id=lot_id))
        
        # Stop detection if running
        stop_parking_detection(lot_id)
        
        # Update parking lot
        parking_lots[lot_id] = {
            "name": name,
            "video_source": video_source,
            "image_source": image_source,  # Image used only for bbox selection
            "model_path": model_path,
            "bounding_boxes": bounding_boxes
        }
        
        # Save to file
        save_parking_lots(parking_lots)
        
        flash(f"Parking lot '{name}' updated successfully")
        return redirect(url_for('admin_dashboard'))
    
    return render_template('edit_parking_lot.html', lot_id=lot_id, lot=parking_lots[lot_id])

@app.route('/admin/delete_parking_lot/<lot_id>', methods=['POST'])
@admin_required
def delete_parking_lot(lot_id):
    if lot_id not in parking_lots:
        flash(f"Parking lot '{lot_id}' not found")
        return redirect(url_for('admin_dashboard'))
    
    # Stop detection if running
    stop_parking_detection(lot_id)
    
    # Check if this is the active lot
    global active_parking_lot
    if active_parking_lot == lot_id:
        # Find another lot to make active
        remaining_lots = list(parking_lots.keys())
        remaining_lots.remove(lot_id)
        if remaining_lots:
            active_parking_lot = remaining_lots[0]
        else:
            active_parking_lot = None
    
    # Remove lot from data structures
    lot_name = parking_lots[lot_id]['name']
    del parking_lots[lot_id]
    if lot_id in parking_status:
        del parking_status[lot_id]
    if lot_id in vehicle_logs:
        del vehicle_logs[lot_id]
    
    # Save to file
    save_parking_lots(parking_lots)
    
    flash(f"Parking lot '{lot_name}' deleted successfully")
    return redirect(url_for('admin_dashboard'))

@app.route('/check_admin_status')
@login_required
def check_admin_status():
    username = session['user']
    user_data = users.get(username, {})
    user_email = user_data.get('email', 'No email found')
    
    is_admin_by_code = user_email.lower() == ADMIN_EMAIL.lower()
    is_admin_by_flag = user_data.get('is_admin', False)
    
    debug_info = {
        'username': username,
        'email': user_email,
        'admin_email': ADMIN_EMAIL,
        'is_admin_by_email_check': is_admin_by_code,
        'is_admin_by_flag': is_admin_by_flag,
        'all_users': {k: {'email': v.get('email'), 'is_admin': v.get('is_admin')} for k, v in users.items()}
    }
    
    return render_template('admin_check.html', debug_info=debug_info)

# New route for serving parking lot images (for bbox selection)
@app.route('/get_parking_image/<lot_id>')
@login_required
def get_parking_image(lot_id):
    if lot_id not in parking_lots:
        return "Parking lot not found", 404
    
    image_path = parking_lots[lot_id].get('image_source')
    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404
    
    return send_file(image_path, mimetype='image/jpeg')

# New routes for bounding box creation
@app.route('/admin/create_bbox', methods=['GET'])
@admin_required
def create_bbox():
    selected_lot = request.args.get('lot_id')
    return render_template('create_bbox.html', 
                         parking_lots=parking_lots, 
                         selected_lot=selected_lot)

@app.route('/admin/start_bbox_tool', methods=['POST'])
@admin_required
def start_bbox_tool():
    data = request.json
    lot_id = data.get('lot_id')
    
    if not lot_id or lot_id not in parking_lots:
        return jsonify({'success': False, 'error': 'Invalid parking lot ID'})
    
    lot_info = parking_lots[lot_id]
    image_path = lot_info.get('image_source')
    bounding_boxes_file = lot_info.get('bounding_boxes')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'Image file not found'})
    
    try:
        # Create a script to run the bbox tool with the image
        script_content = f"""
import os
import sys
from parking_management import ParkingPtsSelection

# Create a standalone script to run the selection tool
if __name__ == "__main__":
    # Use the image file for bbox selection
    image_path = "{image_path}"
    output_filename = "{bounding_boxes_file}"
    
    # Run the bbox selection tool
    selector = ParkingPtsSelection(output_filename=output_filename)
    
    # The selector will automatically open a file dialog
    # We'll pre-select the image by simulating the file selection
    from tkinter import filedialog
    import threading
    import time
    
    def select_file():
        time.sleep(1)  # Wait for the dialog to open
        selector.image_filepath = image_path
        from PIL import Image, ImageTk
        selector.image = Image.open(image_path)
        selector.imgw, selector.imgh = selector.image.size
        aspect_ratio = selector.imgw / selector.imgh
        canvas_width = (
            min(selector.canvas_max_width, selector.imgw) if aspect_ratio > 1 else int(selector.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(selector.canvas_max_height, selector.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )
        selector.canvas.config(width=canvas_width, height=canvas_height)
        selector.canvas_image = ImageTk.PhotoImage(selector.image.resize((canvas_width, canvas_height)))
        selector.canvas.create_image(0, 0, anchor=selector.tk.NW, image=selector.canvas_image)
        selector.canvas.bind("<Button-1>", selector.on_canvas_click)
        selector.rg_data.clear()
        selector.current_box.clear()
    
    # Run file selection in a separate thread
    threading.Thread(target=select_file, daemon=True).start()
"""
        script_path = 'temp_bbox_script.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run the script in a separate process
        subprocess.Popen([sys.executable, script_path])
        
        # Clean up the temporary script after a delay
        def cleanup_script():
            time.sleep(5)  # Wait for the script to start
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except:
                    pass  # Ignore errors if file is still in use
        
        threading.Thread(target=cleanup_script, daemon=True).start()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Setup function to create necessary files and directories
def setup_application():
    # Ensure directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Create template files
    templates = {
        'admin.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart Admin</h1>
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/dashboard" class="nav-link">Dashboard</a>
                <a href="/admin/create_bbox" class="nav-link">Create Bounding Boxes</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <div class="admin-header">
                <h2 class="page-title">Parking Lots Management</h2>
                <a href="/admin/add_parking_lot" class="btn btn-primary">Add New Parking Lot</a>
            </div>
            
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <div class="flash-message">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Video Source</th>
                            <th>Image Source</th>
                            <th>Model Path</th>
                            <th>Bounding Boxes</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for lot_id, lot in parking_lots.items() %}
                        <tr>
                            <td>{{ lot_id }}</td>
                            <td>{{ lot.name }}</td>
                            <td>{{ lot.video_source }}</td>
                            <td>{{ lot.image_source }}</td>
                            <td>{{ lot.model_path }}</td>
                            <td>{{ lot.bounding_boxes }}</td>
                            <td class="action-buttons">
                                <a href="/admin/edit_parking_lot/{{ lot_id }}" class="btn btn-outline btn-sm">Edit</a>
                                <form method="post" action="/admin/delete_parking_lot/{{ lot_id }}" onsubmit="return confirm('Are you sure you want to delete this parking lot?');">
                                    <button type="submit" class="btn btn-danger btn-sm">Delete</button>
                                </form>
                            </td>
                        </tr>
                        {% endfor %}
                        {% if not parking_lots %}
                        <tr>
                            <td colspan="7" class="no-data">No parking lots available</td>
                        </tr>
                        {% endif %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
''',
        'add_parking_lot.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Parking Lot - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart Admin</h1>
            <div class="nav-links">
                <a href="/admin" class="nav-link">Back to Admin</a>
                <a href="/" class="nav-link">Home</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <h2 class="page-title">Add New Parking Lot</h2>
            
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <div class="flash-message">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="form-container">
                <form method="post" class="admin-form">
                    <div class="input-group">
                        <label for="lot_id">Lot ID (unique identifier)</label>
                        <input id="lot_id" name="lot_id" placeholder="e.g., north_lot" required>
                        <small>Use lowercase letters, numbers, and underscores only</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="name">Lot Name (display name)</label>
                        <input id="name" name="name" placeholder="e.g., North Parking Lot" required>
                    </div>
                    
                    <div class="input-group">
                        <label for="video_source">Video Source (for monitoring)</label>
                        <input id="video_source" name="video_source" placeholder="e.g., north_lot.mp4" required>
                        <small>Video file that will be monitored for parking detection</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="image_source">Image Source (for bounding box selection)</label>
                        <input id="image_source" name="image_source" placeholder="e.g., north_lot.jpg" required>
                        <small>Static image file used to draw parking spot boundaries</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="model_path">Model Path</label>
                        <input id="model_path" name="model_path" value="best (4).pt" required>
                        <small>Path to YOLO model file relative to application directory</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="bounding_boxes">Bounding Boxes File</label>
                        <input id="bounding_boxes" name="bounding_boxes" placeholder="e.g., north_lot_boxes.json" required>
                        <small>JSON file that will store the parking spot coordinates drawn on the image</small>
                    </div>
                    
                    <div class="form-buttons">
                        <a href="/admin" class="btn btn-outline">Cancel</a>
                        <button type="submit" class="btn btn-primary">Add Parking Lot</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
''',
        'edit_parking_lot.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edit Parking Lot - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart Admin</h1>
            <div class="nav-links">
                <a href="/admin" class="nav-link">Back to Admin</a>
                <a href="/" class="nav-link">Home</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <h2 class="page-title">Edit Parking Lot: {{ lot.name }}</h2>
            
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <div class="flash-message">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="form-container">
                <form method="post" class="admin-form">
                    <div class="input-group">
                        <label for="lot_id">Lot ID</label>
                        <input id="lot_id" value="{{ lot_id }}" disabled>
                        <small>ID cannot be changed</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="name">Lot Name (display name)</label>
                        <input id="name" name="name" value="{{ lot.name }}" required>
                    </div>
                    
                    <div class="input-group">
                        <label for="video_source">Video Source (for monitoring)</label>
                        <input id="video_source" name="video_source" value="{{ lot.video_source }}" required>
                        <small>Video file that will be monitored for parking detection</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="image_source">Image Source (for bounding box selection)</label>
                        <input id="image_source" name="image_source" value="{{ lot.image_source }}" required>
                        <small>Static image file used to draw parking spot boundaries</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="model_path">Model Path</label>
                        <input id="model_path" name="model_path" value="{{ lot.model_path }}" required>
                        <small>Path to YOLO model file relative to application directory</small>
                    </div>
                    
                    <div class="input-group">
                        <label for="bounding_boxes">Bounding Boxes File</label>
                        <input id="bounding_boxes" name="bounding_boxes" value="{{ lot.bounding_boxes }}" required>
                        <small>JSON file that stores the parking spot coordinates drawn on the image</small>
                    </div>
                    
                    <div class="form-buttons">
                        <a href="/admin" class="btn btn-outline">Cancel</a>
                        <button type="submit" class="btn btn-primary">Update Parking Lot</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
''',
        'create_bbox.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Bounding Boxes - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart Admin</h1>
            <div class="nav-links">
                <a href="/admin" class="nav-link">Back to Admin</a>
                <a href="/" class="nav-link">Home</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <h2 class="page-title">Create Bounding Boxes</h2>
            
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <div class="flash-message">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="form-container">
                <div class="card">
                    <h3>Select Parking Lot</h3>
                    <form method="get" class="admin-form">
                        <div class="input-group">
                            <label for="lot_select">Select Parking Lot:</label>
                            <select id="lot_select" name="lot_id" onchange="this.form.submit()">
                                <option value="">-- Select a parking lot --</option>
                                {% for lot_id, lot in parking_lots.items() %}
                                    <option value="{{ lot_id }}" {% if lot_id == selected_lot %}selected{% endif %}>
                                        {{ lot.name }}
                                    </option>
                                {% endfor %}
                            </select>
                        </div>
                    </form>
                </div>
                
                {% if selected_lot %}
                <div class="card" style="margin-top: 20px;">
                    <h3>Parking Lot: {{ parking_lots[selected_lot].name }}</h3>
                    <div class="image-preview" style="margin: 20px 0;">
                        <img src="/get_parking_image/{{ selected_lot }}" 
                             alt="{{ parking_lots[selected_lot].name }}" 
                             style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    
                    <div class="info-box">
                        <p><strong>Image file:</strong> {{ parking_lots[selected_lot].image_source }}</p>
                        <p><strong>Video file:</strong> {{ parking_lots[selected_lot].video_source }}</p>
                        <p><strong>Bounding boxes file:</strong> {{ parking_lots[selected_lot].bounding_boxes }}</p>
                    </div>
                    
                    <button onclick="startBBoxTool('{{ selected_lot }}')" class="btn btn-primary">
                        Create Bounding Boxes on Image
                    </button>
                    
                    <p class="help-text" style="margin-top: 16px;">
                        This will open the bounding box selection tool with the image file. 
                        The bounding boxes you draw on the image will be used for detecting parking spots in the video:
                        <ul>
                            <li>The tool will automatically load the image file</li>
                            <li>Click 4 points to create a parking spot boundary</li>
                            <li>Right-click to remove the last box</li>
                            <li>Click "Save" to save the bounding boxes to: {{ parking_lots[selected_lot].bounding_boxes }}</li>
                            <li>These boundaries will be applied to the video for real-time detection</li>
                        </ul>
                    </p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <script>
    function startBBoxTool(lotId) {
        // Send request to start the bbox tool on the server
        fetch('/admin/start_bbox_tool', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({lot_id: lotId})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('The bounding box tool has been started. The image will be automatically loaded. Please check the application window.');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to start the bounding box tool');
        });
    }
    </script>
</body>
</html>
''',
        'login.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <div class="auth-card">
            <h2>Login</h2>
            <form method="post">
                <div class="input-group">
                    <label for="username">Username</label>
                    <input id="username" name="username" placeholder="Enter your username" required>
                </div>
                <div class="input-group">
                    <label for="password">Password</label>
                    <input id="password" name="password" type="password" placeholder="Enter your password" required>
                </div>
                <button type="submit" class="btn btn-primary">Login</button>
                {% if error %}<p class="error-message">{{ error }}</p>{% endif %}
            </form>
            <p class="auth-link">Don't have an account? <a href="/signup">Sign up</a></p>
        </div>
    </div>
</body>
</html>
''',
        'signup.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <div class="auth-card">
            <h2>Create Account</h2>
            <form method="post">
                <div class="input-group">
                    <label for="username">Username</label>
                    <input id="username" name="username" placeholder="Choose a username" required>
                </div>
                <div class="input-group">
                    <label for="email">Email</label>
                    <input id="email" name="email" type="email" placeholder="Enter your email" required>
                </div>
                <div class="input-group">
                    <label for="password">Password</label>
                    <input id="password" name="password" type="password" placeholder="Create a password" required>
                </div>
                <button type="submit" class="btn btn-primary">Sign Up</button>
                {% if error %}<p class="error-message">{{ error }}</p>{% endif %}
            </form>
            <p class="auth-link">Already have an account? <a href="/login">Login</a></p>
        </div>
    </div>
</body>
</html>
''',
        'logout.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logged Out - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <div class="auth-card">
            <h2>Goodbye{% if user %}, {{ user }}{% endif %}!</h2>
            <p>You have been successfully logged out.</p>
            <div class="center-btn">
                <a href="/login" class="btn btn-primary">Back to Login</a>
            </div>
        </div>
    </div>
</body>
</html>
''',
        'home.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart</h1>
            <div class="user-info">
                <span>Welcome, {{ session['user'] }}</span>
                {% if is_admin %}
                <a href="/admin" class="nav-link admin-link">Admin Dashboard</a>
                {% endif %}
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <div class="flash-message">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="welcome-card">
                <h2>Welcome to ParkSmart Management System</h2>
                <p>Monitor and manage your parking spaces efficiently with our intelligent system.</p>
                
                <div class="button-grid">
                    <a href="/dashboard" class="feature-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                        View Dashboard
                    </a>
                    <a href="/history" class="feature-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                        View History
                    </a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
''',
        'dashboard.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <script src="{{ url_for('static', filename='js/dashboard.js') }}" defer></script>
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart</h1>
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/history" class="nav-link">History</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <div class="lot-selector">
                <h2 class="page-title">Live Parking Dashboard: {{ active_lot_name }}</h2>
                
                <form method="post" action="/set_active_lot" class="lot-selection-form">
                    <label for="lot_select">Select Parking Lot:</label>
                    <select id="lot_select" name="lot_id" onchange="this.form.submit()">
                        {% for lot_id, lot in parking_lots.items() %}
                            <option value="{{ lot_id }}" {% if lot_id == active_lot %}selected{% endif %}>
                                {{ lot.name }}
                            </option>
                        {% endfor %}
                    </select>
                </form>
            </div>
            
            <div class="status-cards">
                <div class="status-card">
                    <div class="status-icon occupied">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6" y2="6"></line><line x1="6" y1="18" x2="6" y2="18"></line></svg>
                    </div>
                    <div class="status-info">
                        <div class="status-label">Occupied Slots</div>
                        <div class="status-value" id="occupied">0</div>
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-icon available">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="8.5" cy="7" r="4"></circle><polyline points="17 11 19 13 23 9"></polyline></svg>
                    </div>
                    <div class="status-info">
                        <div class="status-label">Available Slots</div>
                        <div class="status-value" id="available">0</div>
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-icon time">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                    </div>
                    <div class="status-info">
                        <div class="status-label">Last Updated</div>
                        <div class="status-value timestamp" id="last_updated">--</div>
                    </div>
                </div>
            </div>
            
            <div class="video-container">
                <img src="/video_feed" alt="Parking Camera Feed" class="video-feed">
            </div>
        </div>
    </div>
</body>
</html>
''',
        'history.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>History - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart</h1>
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/dashboard" class="nav-link">Dashboard</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <div class="lot-selector">
                <h2 class="page-title">Parking Log History: {{ active_lot_name }}</h2>
                
                <form method="post" action="/set_active_lot" class="lot-selection-form">
                    <label for="lot_select">Select Parking Lot:</label>
                    <select id="lot_select" name="lot_id" onchange="this.form.submit()">
                        {% for lot_id, lot in parking_lots.items() %}
                            <option value="{{ lot_id }}" {% if lot_id == active_lot %}selected{% endif %}>
                                {{ lot.name }}
                            </option>
                        {% endfor %}
                    </select>
                </form>
            </div>
            
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Occupied</th>
                            <th>Available</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for entry in log %}
                        <tr>
                            <td>{{ entry.time }}</td>
                            <td>{{ entry.occupied }}</td>
                            <td>{{ entry.available }}</td>
                        </tr>
                        {% endfor %}
                        {% if not log %}
                        <tr>
                            <td colspan="3" class="no-data">No history data available</td>
                        </tr>
                        {% endif %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
''',
        'admin_check.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Status Check - Parking Management System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <style>
        .debug-info {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .status-True {
            color: green;
            font-weight: bold;
        }
        .status-False {
            color: red;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <nav class="navbar">
            <h1 class="brand">ParkSmart Admin Check</h1>
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/logout" class="btn btn-outline">Logout</a>
            </div>
        </nav>
        
        <div class="content">
            <h2 class="page-title">Admin Status Check</h2>
            
            <div class="card">
                <h3>Your Account Status</h3>
                <p><strong>Username:</strong> {{ debug_info.username }}</p>
                <p><strong>Email:</strong> {{ debug_info.email }}</p>
                <p><strong>Admin Email Setting:</strong> {{ debug_info.admin_email }}</p>
                <p><strong>Admin by Email Check:</strong> 
                   <span class="status-{{ debug_info.is_admin_by_email_check }}">{{ debug_info.is_admin_by_email_check }}</span>
                </p>
                <p><strong>Admin by Flag:</strong> 
                   <span class="status-{{ debug_info.is_admin_by_flag }}">{{ debug_info.is_admin_by_flag }}</span>
                </p>
                
                <p class="help-text">
                    If your email matches the admin email but you don't have admin access:
                    <ol>
                        <li>Log out and sign up again with exactly: {{ debug_info.admin_email }}</li>
                        <li>Make sure there are no extra spaces in your email</li>
                        <li>If problems persist, try deleting the users.json file and restarting the application</li>
                    </ol>
                </p>
            </div>
            
            <h3>User Database</h3>
            <div class="debug-info">
{{ debug_info.all_users | pprint }}
            </div>
        </div>
    </div>
</body>
</html>
'''
    }
    
    # Create the template files
    for filename, content in templates.items():
        with open(f'templates/{filename}', 'w') as f:
            f.write(content)
    
    # Create CSS file
    css_content = '''
:root {
    --primary-color: #1a73e8;
    --secondary-color: #4285f4;
    --accent-color: #fbbc05;
    --text-color: #202124;
    --light-text: #5f6368;
    --border-color: #dadce0;
    --error-color: #d93025;
    --success-color: #0f9d58;
    --bg-color: #f8f9fa;
    --card-bg: #ffffff;
    --danger-color: #ea4335;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
    background-color: var(--bg-color);
    color: var(--text-color);
    line-height: 1.6;
}

.container {
    display: flex;
    min-height: 100vh;
    justify-content: center;
    align-items: center;
    padding: 20px;
}

.app-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Auth Forms */
.auth-card {
    background: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 40px;
    width: 100%;
    max-width: 400px;
}

.auth-card h2 {
    color: var(--primary-color);
    margin-bottom: 24px;
    text-align: center;
}

.input-group {
    margin-bottom: 20px;
}

.input-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
    color: var(--light-text);
}

.input-group input {
    width: 100%;
    padding: 12px 16px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 16px;
    transition: border 0.3s;
}

.input-group input:focus {
    border-color: var(--primary-color);
    outline: none;
}

.input-group small {
    display: block;
    color: var(--light-text);
    font-size: 12px;
    margin-top: 4px;
}

.input-group input:disabled {
    background-color: #f1f3f4;
    cursor: not-allowed;
}

.btn {
    display: inline-block;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 500;
    text-align: center;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.3s;
    border: none;
    text-decoration: none;
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background-color: var(--secondary-color);
}

.btn-outline {
    background-color: transparent;
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
    padding: 8px 16px;
    font-size: 14px;
}

.btn-outline:hover {
    background-color: rgba(26, 115, 232, 0.1);
}

.btn-danger {
    background-color: var(--danger-color);
    color: white;
}

.btn-danger:hover {
    background-color: #c62828;
}

.btn-sm {
    padding: 6px 12px;
    font-size: 12px;
}

.error-message {
    color: var(--error-color);
    margin-top: 16px;
    text-align: center;
}

.auth-link {
    margin-top: 24px;
    text-align: center;
    color: var(--light-text);
}

.auth-link a {
    color: var(--primary-color);
    text-decoration: none;
}

.center-btn {
    text-align: center;
    margin-top: 24px;
}

/* Navbar */
.navbar {
    background-color: var(--card-bg);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.brand {
    color: var(--primary-color);
    font-size: 24px;
    font-weight: 700;
}

.nav-links, .user-info {
    display: flex;
    align-items: center;
    gap: 20px;
}

.nav-link {
    color: var(--light-text);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s;
}

.nav-link:hover {
    color: var(--primary-color);
}

.admin-link {
    color: var(--accent-color);
    font-weight: 600;
}

.user-info span {
    color: var(--light-text);
    margin-right: 12px;
}

/* Content area */
.content {
    flex: 1;
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
}

.page-title {
    margin-bottom: 24px;
    color: var(--text-color);
    font-weight: 600;
}

/* Welcome page */
.welcome-card {
    background-color: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 32px;
    text-align: center;
}

.welcome-card h2 {
    color: var(--primary-color);
    margin-bottom: 16px;
}

.welcome-card p {
    color: var(--light-text);
    margin-bottom: 32px;
    font-size: 18px;
}

.button-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 24px;
}

.feature-button {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background-color: var(--bg-color);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    color: var(--primary-color);
    text-decoration: none;
    transition: all 0.3s;
    gap: 12px;
}

.feature-button:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.feature-button svg {
    margin-bottom: 8px;
}

/* Dashboard */
.lot-selector {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 16px;
}

.lot-selection-form {
    display: flex;
    align-items: center;
    gap: 12px;
}

.lot-selection-form select {
    padding: 8px 12px;
    border-radius: 4px;
    border: 1px solid var(--border-color);
    background-color: white;
    font-size: 14px;
    min-width: 200px;
}

.status-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 24px;
    margin-bottom: 24px;
}

.status-card {
    background-color: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 24px;
    display: flex;
    align-items: center;
}

.status-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 16px;
}

.occupied {
    background-color: rgba(217, 48, 37, 0.1);
    color: var(--error-color);
}

.available {
    background-color: rgba(15, 157, 88, 0.1);
    color: var(--success-color);
}

.time {
    background-color: rgba(251, 188, 5, 0.1);
    color: var(--accent-color);
}

.status-info {
    flex: 1;
}

.status-label {
    color: var(--light-text);
    font-size: 14px;
    margin-bottom: 4px;
}

.status-value {
    font-size: 24px;
    font-weight: 600;
}

.timestamp {
    font-size: 16px;
}

.video-container {
    background-color: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 16px;
    overflow: hidden;
}

.video-feed {
    width: 100%;
    height: auto;
    border-radius: 4px;
}

/* History table */
.data-table-container {
    background-color: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 16px;
    overflow-x: auto;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
}

.data-table th, .data-table td {
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.data-table th {
    background-color: var(--bg-color);
    font-weight: 600;
    color: var(--light-text);
}

.data-table tr:last-child td {
    border-bottom: none;
}

.no-data {
    text-align: center;
    color: var(--light-text);
    padding: 32px;
}

/* Admin pages */
.admin-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}

.admin-header .btn-primary {
    min-width: 180px;
}

.flash-message {
    background-color: #e8f0fe;
    color: var(--primary-color);
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 24px;
    border-left: 4px solid var(--primary-color);
}

.action-buttons {
    display: flex;
    gap: 8px;
}

.action-buttons form {
    margin: 0;
}

.form-container {
    background-color: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 24px;
    max-width: 800px;
    margin: 0 auto;
}

.admin-form {
    max-width: 100%;
}

.form-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 16px;
    margin-top: 32px;
}

.help-text {
    background-color: #fef8e7;
    border-left: 4px solid var(--accent-color);
    padding: 12px 16px;
    margin-top: 20px;
    border-radius: 4px;
}

.help-text ol {
    margin-left: 20px;
    margin-top: 10px;
}

.card {
    background-color: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    padding: 24px;
    margin-bottom: 24px;
}

.card h3 {
    margin-bottom: 16px;
    color: var(--text-color);
}

.image-preview {
    border: 1px solid var(--border-color);
    border-radius: 4px;
    overflow: hidden;
    margin: 16px 0;
}

.image-preview img {
    width: 100%;
    height: auto;
    display: block;
}

.info-box {
    background-color: #f1f3f4;
    padding: 16px;
    border-radius: 4px;
    margin: 16px 0;
}

.info-box p {
    margin: 4px 0;
    font-size: 14px;
    color: var(--text-color);
}

@media (max-width: 768px) {
    .status-cards {
        grid-template-columns: 1fr;
    }
    
    .auth-card {
        padding: 24px;
    }
    
    .lot-selector {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .admin-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 16px;
    }
    
    .form-buttons {
        flex-direction: column;
    }
    
    .form-buttons .btn {
        width: 100%;
    }
}
'''
    with open('static/css/style.css', 'w') as f:
        f.write(css_content)
    
    # Create JS file for dashboard
    js_content = '''
async function fetchData() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        
        document.getElementById('occupied').innerText = data.occupied;
        document.getElementById('available').innerText = data.available;
        document.getElementById('last_updated').innerText = data.last_updated;
    } catch (error) {
        console.error('Error fetching parking status:', error);
    }
}

// Fetch immediately and then every second
fetchData();
setInterval(fetchData, 1000);
'''
    with open('static/js/dashboard.js', 'w') as f:
        f.write(js_content)

if __name__ == '__main__':
    # Import numpy and sys here to avoid import errors
    import numpy as np
    import sys
    
    # Run setup to create necessary files
    setup_application()
    
    # Start the parking detection for the active lot
    if active_parking_lot:
        start_parking_detection(active_parking_lot)

    # You can uncomment these lines to reset the users file if needed
    # print("Resetting users file...")
    # if os.path.exists(USER_FILE):
    #     os.remove(USER_FILE)
    #     users = {}
    #     print("Users file has been reset")
    
    # Run the Flask app
    app.run(debug=True, use_reloader=False)  # Disable reloader when using threads