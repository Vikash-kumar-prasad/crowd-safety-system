# System Architecture & Technical Specification

## AI-Based Smart Crowd Safety and Threat Detection System

---

## 1. End-to-End System Pipeline

```
┌────────────────────────────────────────────────────────┐
│                      VIDEO INGESTION                   │
│          Webcam (cv2.VideoCapture) / Video File         │
└───────────────────────────┬────────────────────────────┘
                            │ (BGR Frame, HxWx3)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   STAGE 1: HEAD DETECTION              │
│       YOLOv8 Nano (best.pt, SCUT-HEAD trained)         │
│             Output: List of (x1, y1, x2, y2)           │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       (Head Bounding Boxes)    (Head Bounding Boxes)
                │                        │
                ▼                        ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│ STAGE 2: DENSITY ESTIMATION│  │ STAGE 4: THREAT DETECT  │
│  - 4x6 Spatial Grid       │  │  - YOLOv8 (best_weapon) │
│  - Reference Normalization│  │  - Anatomical Reach     │
│  - Peak Cell Density Score│  │    Filter [hy1-h, hy1+3.5h│
│  - Low / Medium / High    │  │  - Altercation Heuristic│
│  - JET Heatmap Overlay    │  │    (Proximity + Motion) │
└───────────────┬───────────┘  └─────────────┬───────────┘
                │                            │
          (Density Grid)                     │
                │                            │
                ▼                            │
┌───────────────────────────┐                │
│ STAGE 3: BEHAVIOR ANOMALY │                │
│  - Farneback Optical Flow │                │
│  - Rolling History Window │                │
│  - Dispersal / Convergence│                │
│  - High Motion Detection  │                │
└───────────────┬───────────┘                │
                │                            │
         (Behavior Status)            (Threat Status)
                │                            │
                └─────────────┬──────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│               STAGE 5: RISK ASSESSMENT ENGINE          │
│       Rule-Based Explainable Priority Arbitration       │
│       - Weapon -> Critical                             │
│       - Altercation + High Density -> Critical         │
│       - Altercation Alone -> High                      │
│       - High Density + Abnormal Behavior -> High       │
│       - Abnormal Behavior (Any Density) -> Medium      │
│       - High Density Alone -> Medium                   │
│       - Normal -> Low                                  │
│       Output: risk_level, cumulative reasons           │
└─────────────────────────────┬──────────────────────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
┌──────────────────────┐ ┌───────────────┐ ┌──────────────────────┐
│     ALERT ENGINE     │ │ SHARED STATE  │ │     MJPEG STREAM     │
│ - Streak Check (3f)  │ │ (state_lock)  │ │ - cv2.imencode(.jpg) │
│ - Cooldown (5s)      │ │ Telemetry Dict│ │ - HTTP Boundary      │
│ - Snapshot Capture   │ │               │ │   Replacement        │
│ - JSON Log Append    │ │               │ │                      │
└──────────┬───────────┘ └───────┬───────┘ └──────────┬───────────┘
           │                     │                    │
           ▼                     ▼                    ▼
     /api/alerts            /api/status          /video_feed
           │                     │                    │
           └─────────────────────┼────────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ REAL-TIME DASHBOARD   │
                     │  (HTML5 / CSS3 / JS)  │
                     └───────────────────────┘
```

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Spatial Grid Density Estimation
To ensure density classification is invariant to camera input resolution (e.g. 640x480 webcam vs 1920x1080 IP stream), coordinates are mapped to a canonical reference resolution $(W_{ref} = 640, H_{ref} = 480)$:

$$s_x = \frac{W_{ref}}{W_{frame}}, \quad s_y = \frac{H_{ref}}{H_{frame}}$$

For each detected head center $(c_x, c_y) = \left(\frac{x_1 + x_2}{2} \cdot s_x, \frac{y_1 + y_2}{2} \cdot s_y\right)$, the grid cell indices $(r, c)$ in a $R \times C$ grid ($4 \times 6$) are:

$$c = \min\left(\max\left(\left\lfloor \frac{c_x}{W_{ref} / C} \right\rfloor, 0\right), C - 1\right)$$

$$r = \min\left(\max\left(\left\lfloor \frac{c_y}{H_{ref} / R} \right\rfloor, 0\right), R - 1\right)$$

Cell area: $A_{cell} = \left(\frac{W_{ref}}{C}\right) \times \left(\frac{H_{ref}}{R}\right) = \left(\frac{640}{6}\right) \times \left(\frac{480}{4}\right) = 12,800\text{ px}^2$.

Peak density score:

$$\text{Score} = \frac{\max_{r,c} G[r, c]}{A_{cell}}$$

Classification:
- $\text{Score} \ge 0.00080 \implies \text{\textbf{High Density}}$ ($\ge 10\text{ heads in densest cell}$)
- $\text{Score} \ge 0.00030 \implies \text{\textbf{Medium Density}}$ ($\ge 4\text{ heads in densest cell}$)
- Otherwise $\implies \text{\textbf{Low Density}}$

### 2.2 Farneback Dense Optical Flow
Input frames are converted to grayscale and downscaled to width $160\text{px}$ preserving aspect ratio:

$$\text{flow} = \text{calcOpticalFlowFarneback}(I_{t-1}, I_t, \text{pyr\_scale}=0.5, \text{levels}=2, \text{winsize}=15, \text{iterations}=2)$$

Motion magnitude matrix: $M(x, y) = \sqrt{\text{flow}_x^2 + \text{flow}_y^2}$.

Mean motion score:

$$\mu_M = \frac{1}{N}\sum_{x,y} M(x, y)$$

If $\mu_M \ge 2.5 \implies \text{\textbf{High Motion}}$ (stampede/panic indicator).

### 2.3 Temporal Density Delta
To prevent single-frame detector flicker from causing false alarms, head counts are tracked in a FIFO rolling window $H = [c_{t-k}, \dots, c_{t-1}]$.

Baseline average: $\bar{c} = \frac{1}{|H|}\sum_{i \in H} c_i$.

Density change percentage:

$$\Delta\% = \frac{c_t - \bar{c}}{\bar{c}} \quad (\text{evaluated when } \bar{c} \ge 5)$$

- $\Delta\% \le -0.35 \implies \text{\textbf{Sudden Dispersal}}$ (people scattering)
- $\Delta\% \ge +0.35 \implies \text{\textbf{Rapid Convergence}}$ (crowd rushing in)

### 2.4 Anatomical Reach Filtering for Weapons
Raw weapon detections often suffer from false positives on inanimate objects (bottles, bags, signs). When humans are detected, weapon bounding box center $(w_x, w_y)$ must fall within the anatomical reach zone of its Euclidean nearest head $(h_{cx}, h_{cy}, h_w, h_h)$:

$$|w_x - h_{cx}| \le 2.5 \cdot h_w$$

$$(h_{y1} - h_h) \le w_y \le (h_{y1} + 3.5 \cdot h_h)$$

This allows:
- Normal held weapons (waist/chest height)
- Raised or overhead weapons (up to 1 head-height above the head)
- While rejecting background objects far from any person.

### 2.5 Resolution-Aware Altercation Heuristic
Pairwise head center Euclidean distance $d(i, j) = \sqrt{(c_{xi} - c_{xj})^2 + (c_{yi} - c_{yj})^2}$.

Resolution-scaled proximity threshold:

$$\theta_d = 80\text{ px} \times \left(\frac{W_{frame}}{640}\right)$$

Condition: $\exists (i, j) : d(i, j) \le \theta_d \quad \text{AND} \quad \mu_M \ge 3.0$.

---

## 3. Data Storage & Formats

Alerts are stored in `notebooks/alerts/alerts.json` with thread-safe file locking and atomic replacement writes:

```json
{
  "id": "alert_20260925_113714_0001",
  "timestamp": "2026-09-25 11:37:14",
  "risk_level": "Critical",
  "reasons": [
    "Weapon detected",
    "High crowd density with abnormal behavior (High Motion)"
  ],
  "heads": 12,
  "density_level": "High",
  "behavior_status": "High Motion",
  "snapshot_filename": "alert_20260925_113714_0001_Critical.jpg"
}
```

Snapshots are stored in `notebooks/alerts/snapshots/` and served via `/api/alerts/snapshots/<filename>`.
