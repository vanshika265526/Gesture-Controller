# Advanced Gesture Controller

An AI-powered hand gesture recognition system that enables touchless computer control using real-time computer vision. Built with Python, OpenCV, MediaPipe, PyAutoGUI, Pycaw, and Screen Brightness Control, the application allows users to control their computer through intuitive hand gestures captured by a webcam.

The system supports multiple operating modes including virtual mouse control, volume adjustment, brightness management, scrolling, presentation navigation, and custom gesture training. By leveraging MediaPipe's 21-point hand landmark detection and advanced gesture recognition techniques, the application delivers smooth and responsive interaction without requiring physical input devices.

## Features

### Virtual Mouse

* Move cursor using index finger tracking
* Left-click using thumb-index pinch gesture
* Right-click using thumb-middle pinch gesture
* Smooth cursor movement with motion filtering

### Volume Control

* Adjust system volume by changing the distance between thumb and index finger
* Real-time volume percentage visualization
* Automatic mute detection

### Brightness Control

* Control screen brightness through hand gestures
* Live brightness adjustment with visual feedback

### Scroll Control

* Scroll webpages and documents using vertical finger movement
* Smooth up/down scrolling experience

### Presentation Mode

* Navigate presentation slides using open-palm swipe gestures
* Swipe left for next slide
* Swipe right for previous slide

### Custom Gesture Training

* Record and save custom hand gestures
* Persistent gesture storage using JSON
* Real-time gesture matching and recognition

### Multi-Hand Support

* Simultaneous tracking of two hands
* Independent control for volume and brightness
* Enhanced interaction capabilities

## Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* PyAutoGUI
* Pycaw
* Screen Brightness Control
* JSON

## Installation

```bash
pip install opencv-python mediapipe numpy pyautogui pycaw comtypes screen-brightness-control
```

## Usage

```bash
python gesture_controller.py
```

### Controls

| Key     | Action                  |
| ------- | ----------------------- |
| M / Tab | Switch mode             |
| R       | Start gesture recording |
| S       | Save custom gesture     |
| D       | Delete last gesture     |
| Esc     | Cancel recording        |
| Q       | Quit application        |

## System Architecture

```text
Webcam Input
      ↓
OpenCV Frame Processing
      ↓
MediaPipe Hand Detection
      ↓
21 Landmark Extraction
      ↓
Gesture Recognition Engine
      ↓
Mode Controller
      ↓
System Automation Actions
```

## Applications

* Touchless computer interaction
* Smart presentations
* Accessibility solutions
* Human-Computer Interaction (HCI) research
* AI and Computer Vision learning projects
* Productivity enhancement tools

## Future Enhancements

* Gesture-based application launcher
* Virtual keyboard
* AI-powered dynamic gesture recognition
* Voice and gesture hybrid control
* Cross-platform support
* Gesture analytics dashboard

## License

This project is open-source and available for educational and personal use.
