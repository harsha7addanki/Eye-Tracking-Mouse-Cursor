import tkinter as tk
import time
import uuid
import csv
import cv2

# Function to capture image from camera
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image_uuid = str(uuid.uuid4())
        image_path = f"data/{image_uuid}.jpg"
        cv2.imwrite(image_path, frame)
        cap.release()
        return image_uuid, image_path
    cap.release()
    return None, None

# Function to save dot location and image uuid to CSV
def save_to_csv(dot_x, dot_y, image_uuid):
    with open('labels.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dot_x, dot_y, image_uuid])

# Function to show dot and capture image
def show_dot_and_capture():
    dot_x = 0.5  # 50% of the width
    dot_y = 0.5  # 50% of the height
    canvas_width = root.winfo_width()
    canvas_height = root.winfo_height()
    dot_px = int(dot_x * canvas_width)
    dot_py = int(dot_y * canvas_height)

    canvas.create_oval(dot_px-5, dot_py-5, dot_px+5, dot_py+5, fill='black')
    root.update()
    time.sleep(1)
    canvas.delete("all")

    image_uuid, image_path = capture_image()
    if image_uuid:
        save_to_csv(dot_x, dot_y, image_uuid)

# Create tkinter window
root = tk.Tk()
root.geometry("800x600")
root.title("Eye Tracking Mouse Cursor")

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# Show dot and capture image
show_dot_and_capture()

root.mainloop()