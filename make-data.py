from tkinter import messagebox as mb
import tkinter as tk
import random
import time
import uuid
import csv
import cv2

# Create tkinter window
root = tk.Tk()
root.title("Eye Tracking Mouse Cursor")
root.state('zoomed')

canvas = tk.Canvas(root, width=root.winfo_width(), height=root.winfo_height())
canvas.pack(expand=True, fill='both')
canvas.configure(bg='white')

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
def save_to_csv(image_uuid, dot_x, dot_y):
    with open('labels.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_uuid, dot_x, dot_y])

# Function to show dot and capture image
def show_dot_and_capture():
    dot_x = random.random()  # 50% of the width
    dot_y = random.random()  # 50% of the height
    canvas_width = root.winfo_width()
    canvas_height = root.winfo_height()
    dot_px = int(dot_x * canvas_width)
    dot_py = int(dot_y * canvas_height)

    canvas.create_oval(dot_px-5, dot_py-5, dot_px+5, dot_py+5, fill='black')
    canvas.update()
    time.sleep(1)
    canvas.delete("all")

    image_uuid, image_path = capture_image()
    if image_uuid:
        save_to_csv(image_uuid, dot_x, dot_y)


def start():
    # Start the process of showing dot and capturing image
    for i in range(10):
        show_dot_and_capture()
    mb.showinfo("Info", "Data collection complete!")
    if mb.askyesno("Again?", "Do you want to go again?"):
        start()
    else:
        root.quit()
root.after(250, start)  # Call the function after 1 second

root.mainloop()