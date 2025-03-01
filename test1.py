import cv2
import numpy as np
import os
import time
import base64
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Gemini AI
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Create folder for saving full-frame images
FRAME_FOLDER = "full_frames"
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Output text file
OUTPUT_FILE = "observations.txt"

# Timer settings
last_sent_time = 0
SEND_INTERVAL = 4  # Send an image every 4 seconds


def analyze_with_gemini(image_path, timestamp):
    """Sends the full frame to Gemini AI to check for money theft at the cash counter."""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": """
                Observe the **cash counter area** and respond in a structured format.
                If money theft is detected (**"Yes"**), provide details of the **suspect**.
                NO More Details  
                | Suspicious Activity at Cash Counter | Observed? (Yes/No) | Suspect Description (If Yes) |
                |--------------------------------------|--------------------|-----------------------------|
                | Money theft from cash counter?      |                    |                             |

                If theft is detected, describe the **clothing, appearance, and any identifiable features** of the suspect.
                Otherwise, leave the details column empty.
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = gemini_model.invoke([message])
        observation = response.content.strip()

        # Only save observations where theft is confirmed
        if "Yes" in observation:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
                file.write(f"{timestamp} - {observation}\n")
            print(f"✅ Observation Saved: {observation}")

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")


def process_frame(frame):
    """Saves the full frame and starts a thread for AI analysis every 4 seconds."""
    global last_sent_time

    if frame is None or frame.size == 0:
        print("⚠️ Warning: Empty frame received, skipping...")
        return

    current_time = time.time()
    if current_time - last_sent_time >= SEND_INTERVAL:
        last_sent_time = current_time

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(FRAME_FOLDER, f"frame_{timestamp}.jpg")
        cv2.imwrite(image_filename, frame)

        # Run Gemini AI analysis in a separate thread
        ai_thread = threading.Thread(target=analyze_with_gemini, args=(image_filename, timestamp))
        ai_thread.daemon = True
        ai_thread.start()


def on_mouse_move(event, x, y, flags, param):
    """Prints the x, y coordinates when the mouse moves."""
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Coordinates: X={x}, Y={y}")


def start_monitoring(video_file):
    """Reads video frames and monitors the cash counter area."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    cv2.namedWindow("Cash Counter Monitoring")
    cv2.setMouseCallback("Cash Counter Monitoring", on_mouse_move)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))  # Resize for better display
        process_frame(frame)

        cv2.imshow("Cash Counter Monitoring", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Monitoring Completed.")


if __name__ == "__main__":
    video_file = "vid3.mp4"
    start_monitoring(video_file)
