# modified_parking_management.py

import json
import cv2
import numpy as np
import os
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_imshow

def draw_text_with_rounded_bg(img, text, position, box_size=(220, 40), font=cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale=0.7, text_color=(255, 255, 255), bg_color=(0, 0, 0),
                                     thickness=2, radius=12, padding=10):
    box_w, box_h = box_size
    x, y = position
    box_x = x
    box_y = y - box_h

    # Create transparent overlay for rounded rectangle
    overlay = np.zeros((box_h, box_w, 4), dtype=np.uint8)

    # Draw main rounded box
    cv2.rectangle(overlay, (radius, 0), (box_w - radius, box_h), (*bg_color, 255), -1)
    cv2.rectangle(overlay, (0, radius), (box_w, box_h - radius), (*bg_color, 255), -1)

    # Draw corners
    cv2.circle(overlay, (radius, radius), radius, (*bg_color, 255), -1)
    cv2.circle(overlay, (box_w - radius, radius), radius, (*bg_color, 255), -1)
    cv2.circle(overlay, (radius, box_h - radius), radius, (*bg_color, 255), -1)
    cv2.circle(overlay, (box_w - radius, box_h - radius), radius, (*bg_color, 255), -1)

    # Extract region of interest from image
    roi = img[box_y:box_y + box_h, box_x:box_x + box_w]
    if roi.shape[:2] != overlay.shape[:2]:
        return  # Safety check

    mask = overlay[:, :, 3] / 255.0
    rounded_color = overlay[:, :, :3]

    for c in range(3):
        roi[:, :, c] = (1 - mask) * roi[:, :, c] + mask * rounded_color[:, :, c]

    # Draw text centered in box
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = box_x + (box_w - text_w) // 2
    text_y = box_y + (box_h + text_h) // 2 - 4  # adjust vertical alignment
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


class ParkingPtsSelection:
    """
    Modified version of ParkingPtsSelection that saves with custom filename format based on video/image name
    """

    def __init__(self, output_filename=None):
        """Initialize with optional output filename parameter"""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError:
            import platform
            install_cmd = {
                "Linux": "sudo apt install python3-tk (Debian/Ubuntu) | sudo dnf install python3-tkinter (Fedora) | "
                "sudo pacman -S tk (Arch)",
                "Windows": "reinstall Python and enable the checkbox `tcl/tk and IDLE` on **Optional Features** during installation",
                "Darwin": "reinstall Python from https://www.python.org/downloads/mac-osx/ or `brew install python-tk`",
            }.get(platform.system(), "Unknown OS. Check your Python installation.")
            LOGGER.warning(f"WARNING ⚠️  Tkinter is not configured or supported. Potential fix: {install_cmd}")
            return

        if not check_imshow(warn=True):
            return

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)

        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None
        self.canvas_image = None
        self.canvas_max_width = None
        self.canvas_max_height = None
        self.rg_data = None
        self.current_box = None
        self.imgh = None
        self.imgw = None
        
        # Add variable to store custom output filename
        self.output_filename = output_filename
        self.image_filepath = None  # Store the uploaded image filepath

        # Button frame with buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        self.initialize_properties()
        self.master.mainloop()

    def initialize_properties(self):
        """Initialize properties for image, canvas, bounding boxes, and dimensions."""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self):
        """Upload and display an image on the canvas, resizing it to fit within specified dimensions."""
        from PIL import Image, ImageTk
        
        # Store the selected file path
        self.image_filepath = self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not self.image_filepath:
            return
            
        self.image = Image.open(self.image_filepath)
        
        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event):
        """Handle mouse clicks to add points for bounding boxes on the canvas."""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box):
        """Draw a bounding box on the canvas using the provided coordinates."""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=1)

    def remove_last_bounding_box(self):
        """Remove the last bounding box from the list and redraw the canvas."""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self):
        """Redraw the canvas with the image and all bounding boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self):
        """Save the selected parking zone points to a JSON file with scaled coordinates."""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]

        # Determine the output filename
        if self.output_filename:
            save_filename = self.output_filename
        elif self.image_filepath:
            # Use the image filename to create the output filename
            base_filename = os.path.splitext(os.path.basename(self.image_filepath))[0]
            save_filename = f"{base_filename}_bbox.json"
        else:
            save_filename = "bounding_boxes.json"

        # Save to file
        from io import StringIO
        write_buffer = StringIO()
        json.dump(data, write_buffer, indent=4)
        with open(save_filename, "w", encoding="utf-8") as f:
            f.write(write_buffer.getvalue())
        self.messagebox.showinfo("Success", f"Bounding boxes saved to {save_filename}")


class ParkingManagement(BaseSolution):
    """
    Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization.
    """

    def __init__(self, **kwargs):
        """Initialize the parking management system with a YOLO model and visualization settings."""
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # Load JSON data
        if self.json_file is None:
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path can not be empty")

        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information
        #BGR
        self.arc = (0,255,0)  # available region color
        self.occ = (0,0,255)  # occupied region color
        self.dc = (0,0,255)  # centroid color for each box

    def process(self, im0):
        """
        Process the input image for parking lot management and visualization.
        """
        self.extract_tracks(im0)  # extract tracks from im0
        es, fs = len(self.json), 0  # empty slots, filled slots
        annotator = SolutionAnnotator(im0, self.line_width)  # init annotator

        for region in self.json:
            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            # Plotting regions
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=1)

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es
        plot_im = annotator.result()
        self.display_output(plot_im)  # display output with base class function

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            filled_slots=self.pr_info["Occupancy"],
            available_slots=self.pr_info["Available"],
            total_tracks=len(self.track_ids),
        )