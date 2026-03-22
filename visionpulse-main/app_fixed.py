import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import sqlite3
from deepface import DeepFace
import pandas as pd
import time
from datetime import datetime
import json

# --- Page Config and Styling ---
st.set_page_config(page_title="OASIS Edge AI Unit", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #FFFFFF; }
    .stExpander { border-radius: 10px; border: 2px solid #4A4A4A; }
    .stExpander[aria-expanded="true"] { border-color: #4CAF50; }
    .stButton>button { width: 100%; border: 2px solid #4A4A4A; background-color: #262730; color: #FAFAFA; }
    .stButton>button:hover { border-color: #e63946; color: #e63946; }
    .stFileUploader { background-color: #262730; border-radius: 10px; padding: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡 Real-time Security & Surveillance")

# --- Paths ---
DB_PATH = "registered_faces"
SNAPSHOT_DIR = "snapshots"
SQLITE_DB = "vision_alerts.db"
DATA_DIR = "data"

# Create directories if they don't exist
for directory in [DB_PATH, SNAPSHOT_DIR, DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Database Connection (SQLite) ---
def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    return conn

# --- Ensure tables exist ---
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            object_type TEXT,
            camera_id TEXT,
            image_path TEXT
        )
    """)

    # Create email_user table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            report_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

init_db()

# --- Session State ---
if 'run_stream' not in st.session_state:
    st.session_state.run_stream = False
if 'live_history' not in st.session_state:
    st.session_state.live_history = []

# --- Load YOLO Models ---
@st.cache_resource
def load_models():
    vehicle_model = YOLO("yolov8n.pt")
    weapon_model = YOLO("lightingbest.pt")
    return vehicle_model, weapon_model

vehicle_model, weapon_model = load_models()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔴 Live Detection", "👤 Face Registration", "🚨 Full Alerts Log", "📧 Report"])


# --- 1️⃣ Live Detection ---
with tab1:
    st.header("Live Monitoring (Vehicles + Weapons + Face Recognition)")

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("▶ Start Webcam"):
        st.session_state.run_stream = True
    if btn_col2.button("⏹ Stop Webcam"):
        st.session_state.run_stream = False
        st.session_state.live_history = []

    frame_col, summary_col = st.columns([3, 1])
    frame_placeholder = frame_col.empty()
    summary_placeholder = summary_col.empty()

    if st.session_state.run_stream:
        # Try different backend APIs for webcam capture
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        
        for backend in backends:
            cap = cv2.VideoCapture(0 + backend)
            if cap is not None and cap.isOpened():
                # Set buffer size and timeout
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try to set resolution (1280x720 or fallback to 640x480)
                if not cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) or not cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                break
        
        if cap is None or not cap.isOpened():
            st.error("⚠ Unable to access the webcam. Please check your camera connection and permissions.")
        else:
            last_threat_alert_time = {}
            frame_count = 0
            # Add error recovery counter
            error_count = 0
            max_errors = 5

            while st.session_state.run_stream:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        error_count += 1
                        if error_count >= max_errors:
                            st.warning("⚠ Too many frame reading errors. Restarting camera...")
                            cap.release()
                            time.sleep(1)
                            cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
                            if not cap.isOpened():
                                st.error("⚠ Failed to recover camera connection.")
                                st.session_state.run_stream = False
                                break
                            error_count = 0
                        continue
                    error_count = 0  # Reset error count on successful frame read
                except Exception as e:
                    st.error(f"⚠ Camera error: {str(e)}")
                    time.sleep(0.1)  # Add small delay before retry
                    continue

                # ✅ Resize frame to speed up detection
                frame = cv2.resize(frame, (640, 480))
                annotated_frame = frame.copy()
                frame_count += 1

                if frame_count % 5 == 0:  # ✅ Process every 5th frame
                    # --- VEHICLE DETECTION ---
                    results_vehicle = vehicle_model(frame, verbose=False, conf=0.3)
                    for r in results_vehicle:
                        for box in r.boxes:
                            label = r.names[int(box.cls[0])].lower()
                            if label in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                st.session_state.live_history.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "object": label.title()
                                })

                    # --- WEAPON DETECTION with Extended Display ---
                    if not hasattr(st.session_state, 'weapon_detections'):
                        st.session_state.weapon_detections = {
                            'GUN': {'active': False, 'last_seen': 0, 'box': None, 'conf': 0},
                            'KNIFE': {'active': False, 'last_seen': 0, 'box': None, 'conf': 0}
                        }
                    
                    current_time = time.time()
                    results_weapon = weapon_model(frame, verbose=False, conf=0.35)
                    
                    # Process new detections
                    for r in results_weapon:
                        for box in r.boxes:
                            label = r.names[int(box.cls[0])].upper()
                            conf = float(box.conf[0])
                            
                            if label in ["GUN", "KNIFE"] and conf > 0.35:
                                st.session_state.weapon_detections[label] = {
                                    'active': True,
                                    'last_seen': current_time,
                                    'box': box.xyxy[0],
                                    'conf': conf
                                }
                                
                                # Add to history and save screenshot
                                now = datetime.now()
                                st.session_state.live_history.append({
                                    "time": now.strftime("%H:%M:%S"),
                                    "object": f"THREAT: {label}"
                                })
                                
                                last_time = last_threat_alert_time.get(label, datetime.min)
                                if (now - last_time).total_seconds() > 10:
                                    last_threat_alert_time[label] = now
                                    screenshot_path = os.path.join(
                                        SNAPSHOT_DIR, f"{label}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                                    )
                                    cv2.imwrite(screenshot_path, frame)

                                    conn = get_db_connection()
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "INSERT INTO alerts (timestamp, object_type, camera_id, image_path) VALUES (?, ?, ?, ?)",
                                        (now.strftime("%Y-%m-%d %H:%M:%S"), label, "cam0", screenshot_path)
                                    )
                                    conn.commit()
                                    cursor.close()
                                    conn.close()
                    
                    # Display active detections (including persistent display)
                    for label, detection in st.session_state.weapon_detections.items():
                        # Keep detection visible for 2 seconds after last seen
                        if detection['active'] and (current_time - detection['last_seen']) < 2.0:
                            x1, y1, x2, y2 = map(int, detection['box'])
                            color = (0, 0, 255) if label == "GUN" else (0, 255, 255)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, f"!! {label} !!", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            # Add alert text on top of frame
                            alert_text = f"ALERT: {label} DETECTED!"
                            cv2.putText(annotated_frame, alert_text, (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        elif current_time - detection['last_seen'] >= 2.0:
                            # Reset detection after display period
                            detection['active'] = False

                    # --- FACE RECOGNITION ---
                    try:
                        result_df_list = DeepFace.find(
                            img_path=frame, db_path=DB_PATH, enforce_detection=False,
                            silent=True, model_name='SFace', detector_backend='opencv'
                        )
                        if result_df_list and not result_df_list[0].empty:
                            for _, row in result_df_list[0].iterrows():
                                identity = os.path.basename(row['identity']).split('.')[0]
                                st.session_state.live_history.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "object": f"Face: {identity.title()}"
                                })
                                now = datetime.now()
                                last_time = last_threat_alert_time.get(identity, datetime.min)
                                if (now - last_time).total_seconds() > 10:
                                    last_threat_alert_time[identity] = now
                                    screenshot_path = os.path.join(
                                        SNAPSHOT_DIR, f"Face_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                                    )
                                    cv2.imwrite(screenshot_path, frame)

                                    conn = get_db_connection()
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "INSERT INTO alerts (timestamp, object_type, camera_id, image_path) VALUES (?, ?, ?, ?)",
                                        (now.strftime("%Y-%m-%d %H:%M:%S"), f"Face: {identity}", "cam0", screenshot_path)
                                    )
                                    conn.commit()
                                    cursor.close()
                                    conn.close()
                    except Exception as e:
                        pass

                # --- Show frame ---
                frame_placeholder.image(annotated_frame, channels="BGR")

                # --- Summary ---
                with summary_placeholder.container():
                    st.subheader("Recent Detections")
                    if st.session_state.live_history:
                        st.session_state.live_history = st.session_state.live_history[-10:]
                        df_summary = pd.DataFrame(st.session_state.live_history)
                        st.dataframe(df_summary.iloc[::-1], use_container_width=True, hide_index=True)
                    else:
                        st.write("Awaiting detections...")

                time.sleep(0.02)  # ✅ allow Streamlit UI refresh

            cap.release()

    else:
        frame_placeholder.markdown("### Webcam is stopped.")
        summary_placeholder.info("Start webcam to see live detections and alerts.")

# --- 2️⃣ Face Registration ---
with tab2:
    st.header("Register Known Individuals")
    st.info("Upload a clear photo. Name should have no spaces (e.g., 'john_doe').")
    col1, col2 = st.columns([1, 2])

    with col1:
        name_input = st.text_input("Enter person's name:", key="reg_name")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="reg_file")

        if st.button("Register Face", disabled=(not name_input or not uploaded_file)):
            if ' ' in name_input:
                st.error("❌ Please remove spaces from the name.")
            else:
                try:
                    file_ext = os.path.splitext(uploaded_file.name)[1]
                    file_path = os.path.join(DB_PATH, f"{name_input.lower()}{file_ext}")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    pkl_file = os.path.join(DB_PATH, "representations_sface.pkl")
                    if os.path.exists(pkl_file):
                        os.remove(pkl_file)
                    DeepFace.find(img_path=file_path, db_path=DB_PATH, enforce_detection=False,
                                  silent=True, model_name='SFace', detector_backend='opencv')
                    st.success(f"✅ Registered {name_input}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Registered Individuals")
        image_files = [f for f in os.listdir(DB_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            st.warning("No faces registered yet.")
        else:
            for file in image_files:
                file_path = os.path.join(DB_PATH, file)
                col_image, col_btn = st.columns([2, 1])
                with col_image:
                    st.image(file_path, caption=os.path.splitext(file)[0].title(), width=150)
                with col_btn:
                    if st.button(f"❌ Remove {file}", key=f"remove_{file}"):
                        os.remove(file_path)
                        pkl_file = os.path.join(DB_PATH, "representations_sface.pkl")
                        if os.path.exists(pkl_file):
                            os.remove(pkl_file)
                        st.success(f"✅ Removed {file}")
                        st.rerun()

# --- 3️⃣ Alerts Log ---
with tab3:
    st.header("📜 Full Security & Activity Alerts Log")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, object_type, camera_id, image_path FROM alerts ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        st.info("No alerts recorded yet.")
    else:
        for row in rows:
            timestamp, obj_type, cam_id, img_path = row['timestamp'], row['object_type'], row['camera_id'], row['image_path']
            with st.expander(f"🚨 {obj_type} at {timestamp} from {cam_id}"):
                if os.path.exists(img_path):
                    st.image(img_path)
                else:
                    st.warning("Image not found.")
                    
# --- 4️⃣ Report Tab ---                    
with tab4:
    st.header("📧 Generate & Send Security Report")
    email_input = st.text_input("Enter recipient's email:")

    if st.button("📤 Generate and Send Report", disabled=not email_input):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_path = f"Security_Report_{email_input.replace('@', '_')}.pdf"
            
            # ✅ Insert email into database
            cursor.execute("INSERT INTO email_user (email, report_path) VALUES (?, ?)", 
                          (email_input, report_path))
            conn.commit()

            # ✅ Fetch alerts
            cursor.execute("SELECT timestamp, object_type, camera_id, image_path FROM alerts ORDER BY timestamp DESC")
            alerts = cursor.fetchall()

            if not alerts:
                st.warning("No alerts to include in the report.")
            else:
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.units import inch
                import smtplib
                from email.message import EmailMessage
                import mimetypes
                import os

                # ✅ Generate PDF report
                doc = SimpleDocTemplate(report_path, pagesize=A4)
                styles = getSampleStyleSheet()
                story = [Paragraph("Security Alert Report", styles['Title']), Spacer(1, 0.2 * inch)]

                for alert in alerts:
                    story.append(Paragraph(f"<b>Time:</b> {alert['timestamp']}", styles['Normal']))
                    story.append(Paragraph(f"<b>Detected:</b> {alert['object_type']}", styles['Normal']))
                    story.append(Paragraph(f"<b>Camera:</b> {alert['camera_id']}", styles['Normal']))
                    if os.path.exists(alert['image_path']):
                        story.append(Image(alert['image_path'], width=4 * inch, height=3 * inch))
                    story.append(Spacer(1, 0.3 * inch))

                doc.build(story)

                # ✅ Send email
                msg = EmailMessage()
                msg["Subject"] = f"Security Alert Report - {current_time}"
                msg["From"] = "your_email@example.com"
                msg["To"] = email_input

                email_body = f"""
Hi,

Please find attached the latest security alert report.

Report Generated on: {current_time}
Total Alerts: {len(alerts)}

Stay safe,
Your Surveillance System
"""
                msg.set_content(email_body)

                with open(report_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(report_path)
                    mime_type, _ = mimetypes.guess_type(file_name)
                    main_type, sub_type = mime_type.split("/")
                    msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=file_name)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login("visionpluse615@gmail.com", "iyru fgos ahcg qgak")  # Replace securely
                    smtp.send_message(msg)

                st.success("✅ Report generated and sent successfully!")

            cursor.close()
            conn.close()

        except Exception as e:
            st.error(f"❌ Error: {e}")
