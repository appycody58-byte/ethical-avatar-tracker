# Ethical Avatar Tracker

**Production-ready ethical virtual avatar facial tracking system** powered by MediaPipe Face Landmarker.

No celebrity face cloning. No non-consensual deepfakes. Only original / synthetic / consented avatars.

## Features

- Real-time 478 3D face landmarks + 52 ARKit-style blendshapes
- One-Euro filter smoothing for smooth, professional motion
- Built-in FPS counter
- Browser (JavaScript) version optimized for WebRTC video calls
- Python production class ready for Ubuntu servers / desktops
- Clear blendshape mapping for 3D avatars (Ready Player Me, VRoid, custom, etc.)
- Architecture ready for full-body tracking (Face + Pose + Hands)

## Quick Start (Ubuntu)

```bash
git clone https://github.com/appycody58-byte/ethical-avatar-tracker.git
cd ethical-avatar-tracker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the official model
mkdir -p models
wget -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

python scripts/run_demo.py
```

## Project Structure

```
ethical-avatar-tracker/
├── src/
│   ├── face_tracker.py          # Full production Python class
│   └── blendshape_mapper.py
├── web/
│   └── tracker.js               # Browser / WebRTC version
├── scripts/
│   └── run_demo.py
├── models/                   # Place face_landmarker.task here
├── requirements.txt
└── README.md
```

## Ethical Guidelines

This project is intentionally designed for **ethical** avatar systems only:
- Use original / synthetic characters
- Or user-consented digital twins
- Always disclose when an avatar is AI-driven
- Never clone real living people without explicit permission

## License

Apache 2.0 (compatible with MediaPipe)

Built for powerful, clean, ethical AI avatar experiences.
