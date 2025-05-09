# Smart Parking Management System 🚗🅿️

A real-time web-based parking management system that utilizes AI and embedded hardware to detect vehicle presence and manage entry/exit operations.

## 🧠 Project Overview

This project aims to optimize parking space usage through automation and intelligent monitoring.

### 🔍 Problem
Manual monitoring of parking lots is inefficient, error-prone, and time-consuming.

### 💡 Proposed Solution
A system that automatically:
- Detects cars in each parking spot using computer vision (OpenCV).
- Tracks car entries using an ESP32 sensor to control a barrier.
- Updates a real-time dashboard using a web app (Flask + Tailwind CSS + AJAX).

---

## 🧰 Technologies Used

### Backend
- **Python (Flask)** — Web framework
- **OpenCV** — Car detection in live camera feeds
- **ESP32** — Embedded device for car entry detection

### Frontend
- **Tailwind CSS** — UI styling
- **JavaScript + AJAX** — Real-time UI updates without page refresh

### Others
- **JSON** — Communication between scripts
- **Git** — Version control
- **GitHub** — Project hosting

---

## ⚙️ Features

- Real-time availability of each parking spot
- Automatic entry detection with ESP32 and barrier control
- Dynamic and responsive web interface
- Live updates without full-page reloads
