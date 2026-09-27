# Major Project Presentation Slides (PowerPoint Content)

## Project: AI-Based Smart Crowd Safety and Threat Detection System

*You can copy-paste each slide section directly into PowerPoint, Google Slides, or Canva.*

---

### Slide 1: Title Slide
* **Title:** AI-Based Smart Crowd Safety and Threat Detection System
* **Subtitle:** An Automated Computer Vision & Deep Learning Framework for Real-Time Public Safety Surveillance
* **Department:** Department of Computer Science & Engineering
* **Presenters:** [Your Name / Team Members & Roll Numbers]
* **Project Guide:** [Guide Name & Designation]

---

### Slide 2: Problem Statement & Motivation
* **Problem Statement:**
  * Public gatherings (religious events, concerts, railway stations, stadiums) are prone to fatal stampedes, crowd crushes, and violent incidents.
  * Traditional CCTV relies on human operators suffering from cognitive fatigue after 20 minutes of multi-screen monitoring.
* **Motivation:**
  * To automate crowd surveillance by combining spatial crowd density, optical flow behavior tracking, and weapon threat detection.
  * To provide an auditable, early-warning command-center dashboard before incidents escalate into casualties.

---

### Slide 3: Existing Systems vs Proposed System

| Feature | Conventional CCTV / Existing Systems | Proposed System |
|---|---|---|
| **Monitoring** | Passive human observation | Active AI real-time inference |
| **Crowd Density** | Qualitative manual guessing | Quantitative normalized 4x6 grid heatmap |
| **Panic Detection** | Delayed human reporting | Farneback optical flow velocity spikes |
| **Weapon Detection** | Rare / None | YOLOv8 + Anatomical reach verification |
| **Risk Arbitration** | Ad-hoc / subjective | Explainable rule hierarchy (Low to Critical) |
| **Evidence Logging** | Manual tape scanning | Automatic timestamped image snapshots & CSV |

---

### Slide 4: Overall System Architecture
* **Video Acquisition:** RTSP / Webcam / Video Stream (BGR Frames)
* **Stage 1 — Head Detection:** Custom-trained YOLOv8 nano model (SCUT-HEAD)
* **Stage 2 — Density Analysis:** Grid spatial binning ($W_{ref}=640, H_{ref}=480$) + JET colormap overlay
* **Stage 3 — Optical Flow:** Farneback dense motion field ($160\text{px}$) + 5-frame rolling average
* **Stage 4 — Threat & Weapon Engine:** Iteratively trained weapon YOLOv8 + Torso/Hand Reach Filter
* **Stage 5 — Risk Assessment:** Priority-based explainable decision tree
* **Output:** MJPEG live stream + REST API + SOC Dashboard + Audio Alarm + CSV export

---

### Slide 5: Model 1 — Head & Person Detection
* **Why Heads?** In high-density crowds, full bodies and limbs are severely occluded; heads remain visible from elevated surveillance angles.
* **Dataset:** SCUT-HEAD Benchmark (Part A: campus surveillance + Part B: large-scale web crawled).
* **Architecture:** YOLOv8n (Nano variant, 3.2M parameters) — optimal balance between inference speed (15–25 FPS on CPU) and high recall.
* **Confidence Threshold:** 0.30 with Non-Maximum Suppression (IoU: 0.45).

---

### Slide 6: Model 2 — Weapon Detection & Anatomical Filtering
* **Iterative Training:** 4 training rounds merging handguns, rifles, and knife datasets (`best_weapon_v4.pt`).
* **The False Positive Challenge:** Objects like dark belts, wallets, and mobile phones often trigger false alarms.
* **Anatomical Reach Filter:**
  * Weapon candidate $(w_x, w_y)$ is matched to its nearest head $(h_{cx}, h_{cy}, h_w, h_h)$.
  * Horizontal constraint: $|w_x - h_{cx}| \le 2.5 \times h_w$
  * Vertical constraint: $(h_{y1} - h_h) \le w_y \le (h_{y1} + 3.5 \times h_h)$
  * Weapons lying isolated on the floor or background clutter are automatically rejected.

---

### Slide 7: Crowd Density & Heatmap Estimation
* **Scale Invariant Spatial Grid:**
  * Resolution normalized to $640 \times 480\text{ px}$.
  * Spatial partitioning into $4 \text{ rows} \times 6 \text{ columns} = 24 \text{ discrete cells}$.
  * Cell area: $A_{cell} = 12,800\text{ px}^2$.
* **Classification Cutoffs:**
  * $\text{Score} = \frac{\max_{r,c} G[r, c]}{A_{cell}}$
  * **Low Density:** $< 0.00030$ ($< 4$ heads in densest cell)
  * **Medium Density:** $0.00030 \le \text{Score} < 0.00080$ ($4–9$ heads)
  * **High Density:** $\ge 0.00080$ ($\ge 10$ heads per cell)
* **Heatmap Overlay:** JET colormap (Blue = Safe, Red = Congested hotspot) alpha-blended at 45% opacity.

---

### Slide 8: Behavior & Anomaly Detection (Farneback Optical Flow)
* **Dense Optical Flow Formulation:**
  * Computes motion vector displacement $(u, v)$ for all pixels between frames $I_{t-1}$ and $I_t$.
  * Downscaled to $160\text{ px}$ width preserving aspect ratio for $<5\text{ms}$ CPU execution.
  * Motion magnitude: $M(x, y) = \sqrt{u^2 + v^2}$.
* **Anomaly Classification:**
  * **High Motion (Panic/Stampede):** Mean magnitude $\mu_M \ge 2.5$.
  * **Sudden Dispersal:** Head count drop $\Delta\% \le -35\%$ over 5-frame rolling average.
  * **Rapid Convergence:** Head count surge $\Delta\% \ge +35\%$ over 5-frame rolling average.

---

### Slide 9: Multi-Factor Risk Assessment Engine
* **Why Rule-Based?** High-stakes security operations demand explainable, auditable decisions, not uninterpretable probability scores.
* **Priority Escalation Hierarchy:**
  1. Weapon Detected $\rightarrow$ **CRITICAL**
  2. Altercation in High-Density Crowd $\rightarrow$ **CRITICAL**
  3. Altercation Alone $\rightarrow$ **HIGH**
  4. High Density + Abnormal Behavior $\rightarrow$ **HIGH**
  5. Abnormal Behavior at Any Density $\rightarrow$ **MEDIUM**
  6. High Density Alone $\rightarrow$ **MEDIUM**
  7. Calm Crowd / Low Density $\rightarrow$ **LOW**
* **Cumulative Reasons:** All triggering factors are combined (e.g., *"Weapon detected • High crowd density with abnormal behavior"*).

---

### Slide 10: Security Operations Center (SOC) Dashboard
* **Real-Time Stream:** Low-latency multipart MJPEG stream with color-coded bounding boxes and heatmaps.
* **Dynamic Threat HUD:** Color-shifting banner (Green $\rightarrow$ Amber $\rightarrow$ Crimson) with pulsating Critical alarm.
* **Tactical Audio Siren:** Web Audio synthesizer alert tones with header mute toggle.
* **Risk Trend Sparkline:** In-memory SVG tracking 60 temporal data points.
* **Forensic Event Timeline:** Clickable alert history with photo snapshot viewer.
* **Audit & Export:** One-click CSV incident report generator.

---

### Slide 11: Experimental Results & Benchmarks

| Metric | Target / Benchmark | Achieved in System |
|---|---|---|
| **Head Detection mAP@50** | $> 80\%$ | $84.2\%$ (SCUT-HEAD) |
| **Weapon Detection mAP@50** | $> 75\%$ | $79.6\%$ (v4 weights) |
| **Pipeline Latency (CPU)** | $< 100\text{ms}$ | $45–65\text{ms}$ |
| **Effective Frame Rate** | $\ge 12\text{ FPS}$ | $15–22\text{ FPS}$ (Intel i5/i7) |
| **Alert Verification Window** | 3 frames | Eliminates single-frame detector jitter |
| **False Alarm Suppression** | Anatomical filter | $> 60\%$ reduction in background false positives |

---

### Slide 12: Conclusion & Future Scope
* **Conclusion:**
  * Successfully built and demonstrated an end-to-end autonomous crowd safety system combining custom YOLOv8, optical flow, and explainable risk rules.
  * Runs in real-time on commodity CPU hardware with zero external cloud dependencies.
* **Future Enhancements:**
  * 3D-CNN / SlowFast video classification network for trained violence recognition.
  * Distributed multi-camera RTSP management.
  * Night-vision / thermal imaging infrared camera integration.
  * Push notifications via WebSockets or Telegram bot to police patrol teams.
