# 🖼️ Ultrawide Wallpaper Cutter

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> A smart wallpaper generator that automatically crops or extends your images for any screen ratio — from 16:9 up to 32:9 ultrawide monitors.

---

## 🎯 Overview

**Ultrawide Wallpaper Cutter** is a simple yet powerful Streamlit app that helps you turn any image into the perfect wallpaper for your monitor setup.  
No AI dependencies — just clever cropping, edge detection, and seamless background extension.

It supports all popular aspect ratios:
- 16:9 – Standard widescreen  
- 21:9 – Ultrawide  
- 32:9 – Super ultrawide  
- 16:10, 4:3, 3:2, and custom ratios  

You can choose between **Fill**, **Fit**, and **Outpaint-Lite** modes to achieve the best composition for your display.

---

## 🚀 Features

### ✂️ Smart Image Modes
| Mode | Description |
|------|--------------|
| **Fill (crop)** | Edge-aware, face-preserving crop that keeps subjects centered |
| **Fit (letterbox)** | Maintains full image by adding borders to match ratio |
| **Outpaint-Lite (extend)** | Automatically fills sidebars with blurred or mirrored backgrounds for ultrawide output |

### 🧠 Intelligent Cropping
- Gradient-based **edge detection** (keeps the most detailed areas).
- **Face-aware framing** via OpenCV Haar cascades.
- Optional **gravity** control (center, top, bottom, left, right).

### 📦 Batch Processing
- Upload multiple images at once  
- Preview each processed wallpaper  
- Download all results as a single ZIP file

---

## 🧰 Tech Stack

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/)

---

## 💻 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/ultrawide-wallpaper-cutter.git
cd ultrawide-wallpaper-cutter
