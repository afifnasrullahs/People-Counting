# Entrance Detection - People Counter

Sistem pendeteksi dan penghitung orang yang masuk/keluar menggunakan YOLO dan MQTT.

## 📋 Fitur

- **Deteksi Real-time**: Menggunakan YOLOv8 untuk mendeteksi orang
- **Tracking**: Menggunakan BotSORT untuk melacak pergerakan setiap orang
- **Counting**: Menghitung orang yang masuk (MASUK) dan keluar (KELUAR)
- **MQTT Integration**: Mengirim data occupancy ke broker MQTT
- **RTSP/RTMP Support**: Mendukung stream video dari IP camera

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- Paho MQTT Client
- NumPy

## 📦 Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd EntranceDetection
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Atau manual:

4. **Download model weights**
   
   Model YOLOv8 akan otomatis didownload saat pertama kali dijalankan, atau download manual:
   - [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) - Nano (tercepat)
   - [yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) - Small (default)
   - [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) - Medium
   - [yolov8l.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) - Large (paling akurat)

5. **Konfigurasi environment**
   
   Copy `.env.example` ke `.env` dan sesuaikan konfigurasi:
   ```bash
   cp .env.example .env
   ```

## ⚙️ Konfigurasi

Edit file `final.py` atau buat file `.env` untuk mengubah konfigurasi:

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `VIDEO_SOURCE` | RTSP URL | Sumber video (RTSP/RTMP/file/webcam) |
| `MODEL_WEIGHTS` | `yolov8s.pt` | Model YOLO yang digunakan |
| `CONF_THRESHOLD` | `0.35` | Confidence threshold deteksi |
| `LINE_POSITION` | `0.5` | Posisi garis vertikal (0-1) |
| `RESIZE_TO` | `(960, 540)` | Resolusi frame untuk processing |

### MQTT Configuration

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `broker` | `` | Alamat MQTT broker |
| `port` | `1883` | Port MQTT |
| `username` | `` | Username MQTT |
| `password` | `` | Password MQTT |
| `topic` | `` | Topic untuk publish data |

## 🚀 Penggunaan

**Cara baru (modular):**
```bash
python main.py
```

**Cara lama (monolithic):**
```bash
python final.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit program |
| `R` | Reset semua counter |

## 📊 Output

### Display
- **Garis Hijau Vertikal**: Garis penghitung
- **Box Hijau**: Orang di sisi kiri garis
- **Box Merah**: Orang di sisi kanan garis
- **Trail Kuning**: Jejak pergerakan orang

### MQTT Payload
```json
{
  "occupancy": 5
}
```

## 🔧 Troubleshooting

### Stream tidak terbuka
- Pastikan URL RTSP/RTMP benar
- Cek koneksi jaringan ke IP camera
- Pastikan firewall tidak memblokir koneksi

### FPS rendah
- Gunakan model yang lebih ringan (`yolov8n.pt`)
- Kurangi resolusi (`RESIZE_TO`)
- Pastikan GPU tersedia dan CUDA terinstall

### Counting tidak akurat
- Sesuaikan `LINE_POSITION` sesuai posisi pintu
- Tingkatkan `CONF_THRESHOLD` jika banyak false positive
- Kurangi `MIN_MOVEMENT_PIXELS` jika pergerakan lambat

### MQTT tidak terkoneksi
- Verifikasi credentials MQTT
- Cek konektivitas ke broker
- Pastikan port tidak diblokir firewall

## 📁 Struktur Project

```
EntranceDetection/
├── main.py                 # Entry point aplikasi
├── final.py                # Legacy monolithic version
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (tidak di-commit)
├── .gitignore              # Git ignore rules
├── README.md               # Dokumentasi
│
├── src/                    # Source code utama
│   ├── __init__.py
│   ├── config/             # Konfigurasi
│   │   ├── __init__.py
│   │   └── settings.py     # Semua settings dari .env
│   │
│   ├── core/               # Logic inti
│   │   ├── __init__.py
│   │   ├── counter.py      # PeopleCounter class
│   │   └── detector.py     # YOLO detector wrapper
│   │
│   ├── mqtt/               # MQTT handling
│   │   ├── __init__.py
│   │   └── manager.py      # MQTTManager class
│   │
│   ├── stream/             # Video streaming
│   │   ├── __init__.py
│   │   └── reader.py       # RTMPReader class
│   │
│   └── utils/              # Utilities
│       ├── __init__.py
│       └── logger.py       # Logging configuration
│
├── models/                 # YOLO model weights
│   └── .gitkeep
│
├── logs/                   # Log files
│   └── .gitkeep
│
└── tests/                  # Unit tests
    ├── __init__.py
    └── test_mqtt.py
```

## 📝 License

MIT License

## 👥 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request
