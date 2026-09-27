# MAJOR PROJECT VIVA & DEMONSTRATION CHEATSHEET
## AI-Based Smart Crowd Safety and Threat Detection System

> Keep this document open on your phone or printed on your desk during your presentation and viva.

---

## 1. THE 30-SECOND ELEVATOR PITCH
> *"Good morning respected evaluators. Our project is an end-to-end, real-time Computer Vision system designed to prevent crowd disasters and active weapon attacks before casualties occur.*  
> *Unlike conventional CCTV where human operators get fatigued, our system autonomously detects crowd congestion using custom YOLOv8 head detection, maps localized density with organic heatmaps, calculates sudden panic stampedes via Farnebäck optical flow, detects lethal weapons like guns and knives, and instantly dispatches automated Computer-Aided Dispatch (CAD) incident tickets to Police, Quick Response Teams, and Crowd Marshals in under 40 milliseconds."*

---

## 2. THE 3-MINUTE LIVE DEMO SEQUENCE

Follow this exact sequence during your live presentation:

| Step | Action on Screen | What to Say to the Evaluators |
| :--- | :--- | :--- |
| **1. Launch** | Double-click `run_system.bat` | *"The system launches with an automated one-click batch script, initializes our dual YOLOv8 models in background threads, and brings up our Security Operations Center (SOC) dashboard."* |
| **2. Safe Baseline** | Point to live webcam or click `🟢 Safe Baseline` in Modal | *"Here you see the normal operational baseline. Risk is LOW (Green), heads are detected, and because crowd density is low, the frame remains clear and uncolored with zero false alarms."* |
| **3. Crowd Heatmap** | Click `[ 📹 Source ]` -> `Continuous Crowd Walking` OR click `👥 77-Head Stampede` | *"Now we simulate dense crowd movement. Notice how the system identifies 77 heads, partitions them into a 4x6 spatial grid, and renders an organic Gaussian density heatmap showing congestion hotspots in real time."* |
| **4. Lethal Threat & CAD** | Click `🔫 Handgun Threat` or `🔪 Knife Threat` | *"When an armed threat appears, our custom multi-weapon model detects it. The system escalates immediately to CRITICAL (Red pulsing banner), emits an audible tactical radio chirp, and automatically issues a Police CAD Ticket (e.g. `DISPATCH-20260926...`) with target unit acknowledgments."* |
| **5. Drill Override** | Click `[ ⚡ Test Drill ]` in Emergency Dispatch panel | *"Security operators can also trigger instant tactical drills or override alerts from the command center console."* |
| **6. Compliance Audit** | Click `[ 📥 Export CSV ]` | *"Finally, operators can export the entire timestamped incident audit log into a CSV spreadsheet for police evidence and incident investigation."* |

---

## 3. KEY METRICS TO QUOTE FROM MEMORY
- **Head Detector Accuracy:** `Precision: 91.96%`, `Recall: 87.41%`, `mAP@50: 92.89%`
- **Weapon Detector Accuracy:** `Precision: 87.46%`, `Recall: 78.59%`, `mAP@50: 83.28%`
- **End-to-End Pipeline Throughput:** `28.4 FPS`
- **Head Model Inference Latency:** `14.2 ms` (YOLOv8n on SCUT-HEAD)
- **Weapon Model Inference Latency:** `15.6 ms` (YOLOv8n multi-weapon v4)
- **Weapon Confidence Threshold:** `0.48` (empirically balanced for zero video dropouts)
- **Emergency Dispatch CAD Latency:** `< 5 ms` (asynchronous non-blocking thread)
- **Spatial Density Grid:** `4 rows × 6 columns = 24 monitoring sectors`

---

## 4. TOP 10 TOUGH VIVA QUESTIONS & WINNING ANSWERS

### Q1: Why use YOLOv8 instead of pure density estimation networks like CSRNet or MCNN?
> **Answer:** *"CSRNet and MCNN regress a density map pixel-by-pixel, which provides a total count but does NOT output bounding boxes or individual head coordinates. Without bounding boxes, we cannot track velocity vectors for optical flow, nor can we calculate spatial proximity between people and weapons. YOLOv8 gives us both precise spatial localization and high throughput (over 60 FPS on GPU), enabling multi-modal risk fusion in real time."*

### Q2: Why train two separate YOLO models instead of one unified model with both heads and weapons?
> **Answer:** *"For three key engineering reasons:*  
> *1. **Dataset Specialization:** Head datasets (SCUT-HEAD / CrowdHuman) are captured from high-angle CCTV perspectives, whereas weapon datasets focus on handheld objects. Merging them creates severe class imbalance.*  
> *2. **Independent Tuning:** Crowd head detection requires a lower confidence threshold (0.30) to catch small, distant heads, while weapon detection requires a higher threshold (0.48) to prevent false positives.*  
> *3. **Modular Scalability:** In low-risk environments, the weapon detector can be paused to double processing FPS on edge devices."*

### Q3: How do you differentiate between uniform running (like a marathon or boarding a train) and a panic stampede?
> **Answer:** *"We use a dual-criterion anomaly metric: Farnebäck optical flow measures the mean magnitude of motion vectors ($\bar{M}$), while our spatial grid measures divergence and density deltas ($\Delta \bar{M}$). In uniform running, vectors are laminar and directional. In a stampede or panic crush, motion vectors exhibit high spatial turbulence, rapid direction divergence, and sudden velocity surges exceeding our calibrated threshold."*

### Q4: How does your system avoid the 'blue screen' or visual artifacts on video feeds?
> **Answer:** *"Earlier iterations applied standard OpenCV colormaps directly, which mapped zero values to dark blue and drew grid cages over the screen. We upgraded the rendering engine with an adaptive opacity mask: when density is zero, the alpha blend is zero, keeping empty regions 100% natural. When crowds are present, it applies bicubic scaling and a 51x51 Gaussian blur so only genuine crowd clusters glow with heat colors."*

### Q5: What happens if the network drops or the camera feed disconnects?
> **Answer:** *"The pipeline runs asynchronously in a background worker thread decoupled from the web server. If a video feed drops, the worker automatically attempts reconnection every 1.0 second, while the dashboard UI displays a 'Reconnecting...' state without crashing the server or losing past alert history."*

### Q6: How does the Emergency Dispatch System work? Is it connected to real services?
> **Answer:** *"The dispatch engine follows the APCO (Association of Public-Safety Communications Officials) Computer-Aided Dispatch (CAD) standard. When a Critical threat or High Density advisory occurs, it creates a structured incident record with a unique ticket ID, alerts 3 specific target channels (Police CAD, Tactical QRT radio, and Safety Marshals), and triggers an audible radio chirp. It also has a non-blocking webhook architecture ready to push alerts directly to Telegram or municipal CAD gateways."*

### Q7: What are the hardware requirements to deploy this in a real campus?
> **Answer:** *"For a single camera feed, the system runs on standard CPU at 15–20 FPS. For multi-camera production deployment (e.g. 8–16 CCTV streams), an edge server equipped with an NVIDIA RTX 3060 / 4060 GPU or NVIDIA Jetson Orin running TensorRT optimization can process all streams concurrently at 30 FPS."*

### Q8: What are the primary failure cases or limitations of this project?
> **Answer:** *"Honest scientific awareness is key: (1) Extreme night-time low-light conditions require infrared/thermal CCTV cameras; (2) Small pocket knives concealed inside clothing cannot be detected until brandished; and (3) Heavy camera perspective foreshortening requires camera calibration parameters for metric distance accuracy."*

### Q9: How does the system prevent alert spamming if an armed person stays in front of the camera for 5 minutes?
> **Answer:** *"The alert logger enforces a cooldown hysteresis window (e.g. 5 seconds for duplicate alerts) and maintains an in-memory lock. While the visual status remains 'Critical', dispatch tickets are deduplicated so responders receive one actionable incident notification rather than hundreds of duplicate pings."*

### Q10: What future work would you implement given another 6 months?
> **Answer:** *"Multi-camera re-identification (Re-ID) using feature embeddings so an armed suspect can be tracked continuously across 50 cameras, and acoustic gunshot/scream detection using edge micro-array microphones."*
