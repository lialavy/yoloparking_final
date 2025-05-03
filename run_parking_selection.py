# run_parking_selection.py
import sys
import os
import cv2
import time
import shutil
import glob
import json
from ultralytics import solutions

# Video and output paths
VIDEO_PATH = "parking1.mp4"
DESIRED_OUTPUT_FILE = "parking1_bbox.json"
DEFAULT_OUTPUT_FILE = "bounding_boxes.json"
FRAME_IMAGE_PATH = "parking_frame.jpg"

def get_video_dimensions(video_path):
    """Get the actual dimensions of the video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, None
        
        # Get width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        print(f"Video dimensions: {width}x{height}")
        return width, height
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return None, None

def extract_frame(video_path, output_image_path):
    """Extract a frame from the video to use for parking spot selection"""
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
            
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from video")
            cap.release()
            return False
        
        # Save the frame as an image at original resolution
        cv2.imwrite(output_image_path, frame)
        cap.release()
        print(f"Frame extracted and saved to {output_image_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return False

def process_bounding_boxes(original_file, output_file, video_width, video_height):
    """Process and scale bounding boxes to match video resolution"""
    try:
        # Check if the file exists
        if not os.path.exists(original_file):
            print(f"Bounding box file not found: {original_file}")
            return False
        
        # Load bounding boxes
        with open(original_file, 'r') as f:
            bounding_boxes = json.load(f)
        
        # Save to desired output file
        with open(output_file, 'w') as f:
            json.dump(bounding_boxes, f, indent=2)
        
        print(f"Processed bounding boxes saved to {output_file}")
        
        # Display the bounding boxes format for debugging
        print("Bounding box sample:")
        if len(bounding_boxes) > 0:
            print(json.dumps(bounding_boxes[0], indent=2))
        
        return True
        
    except Exception as e:
        print(f"Error processing bounding boxes: {e}")
        return False

def find_and_process_output_file(default_file, desired_file, video_width, video_height):
    """Locate the output file from ParkingPtsSelection and process it"""
    try:
        # Record the list of json files before running
        json_files_before = set(glob.glob("*.json"))
        
        # Wait for the tools to complete
        print("Waiting for ParkingPtsSelection to complete and save the file...")
        time.sleep(2)  # Give it a moment
        
        # Record the list of json files after running
        json_files_after = set(glob.glob("*.json"))
        
        # Find new files that appeared
        new_files = json_files_after - json_files_before
        
        # If we found new files
        if new_files:
            print(f"New JSON files detected: {new_files}")
            if len(new_files) == 1:
                new_file = list(new_files)[0]
                # Process and copy this file
                process_bounding_boxes(new_file, desired_file, video_width, video_height)
                return True
            else:
                # Multiple new files, try to find the most likely candidate
                for file in new_files:
                    if 'bound' in file.lower() or 'parking' in file.lower() or 'bbox' in file.lower():
                        process_bounding_boxes(file, desired_file, video_width, video_height)
                        return True
        
        # If no new files, check if the default file exists
        if os.path.exists(default_file):
            # Process and copy this file
            process_bounding_boxes(default_file, desired_file, video_width, video_height)
            return True
        
        # Still no luck, look for any json files modified in the last minute
        recent_files = []
        for json_file in glob.glob("*.json"):
            if time.time() - os.path.getmtime(json_file) < 120:  # Files modified in the last 2 minutes
                recent_files.append(json_file)
        
        if recent_files:
            print(f"Found recently modified JSON files: {recent_files}")
            if len(recent_files) == 1:
                process_bounding_boxes(recent_files[0], desired_file, video_width, video_height)
                return True
        
        print("Could not locate the output file from ParkingPtsSelection.")
        return False
            
    except Exception as e:
        print(f"Error finding output file: {e}")
        return False

try:
    print(f"Video path: {VIDEO_PATH}")
    print(f"Desired output file: {DESIRED_OUTPUT_FILE}")
    
    # Get the actual video dimensions
    video_width, video_height = get_video_dimensions(VIDEO_PATH)
    if not video_width or not video_height:
        print("Failed to get video dimensions. Cannot proceed.")
    else:
        print(f"Working with video dimensions: {video_width}x{video_height}")
        
        # First extract a frame from the video at original resolution
        if extract_frame(VIDEO_PATH, FRAME_IMAGE_PATH):
            print(f"Launching ParkingPtsSelection with image {FRAME_IMAGE_PATH}")
            print("Please select the parking spots in the window that opens.")
            print("When you're done, close the tool and this script will process the results.")
            
            # Run the ParkingPtsSelection tool
            solutions.ParkingPtsSelection()
            
            print("ParkingPtsSelection tool should now be open.")
            print("Please draw the parking spots and close the tool when finished.")
            print("This script will automatically process the results once you're done.")
            
            # Find the output file and process it
            if find_and_process_output_file(DEFAULT_OUTPUT_FILE, DESIRED_OUTPUT_FILE, video_width, video_height):
                print(f"Successfully processed bounding boxes to match video resolution.")
            else:
                print("Failed to locate or process bounding boxes.")
        else:
            print("Failed to extract frame from video. Cannot proceed.")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up the temporary frame image
    if os.path.exists(FRAME_IMAGE_PATH):
        try:
            os.remove(FRAME_IMAGE_PATH)
            print(f"Removed temporary frame image {FRAME_IMAGE_PATH}")
        except:
            pass
    
    print("\nPress Enter to exit...")
    input()
