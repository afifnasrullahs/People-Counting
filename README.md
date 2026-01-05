# Entrance Detection - People Counter

Sistem pendeteksi dan penghitung orang yang masuk/keluar menggunakan MobileNet-SSD dan MQTT.

## 📋 Fitur

- **Deteksi Real-time**: Menggunakan MobileNet-SSD untuk mendeteksi orang
- **Tracking**: Menggunakan centroid-based tracker untuk melacak pergerakan setiap orang
- **Counting**: Menghitung orang yang masuk (MASUK) dan keluar (KELUAR)
- **MQTT Integration**: Mengirim data occupancy ke broker MQTT
- **RTSP/RTMP Support**: Mendukung stream video dari IP camera
- **Lightweight**: Tidak memerlukan GPU, berjalan efisien di CPU

## 🛠️ Requirements

- Python 3.8+
- OpenCV (dengan modul DNN)
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

4. **Download model MobileNet-SSD**
   
   Jalankan script untuk mendownload model:
   ```bash
   python download_model.py
   ```
   
   Atau download manual:
   - [MobileNetSSD_deploy.prototxt](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt) - Config file
   - [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel) - Weights file
   
   Letakkan file di folder `models/`

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
| `MODEL_CONFIG` | `MobileNetSSD_deploy.prototxt` | Config file MobileNet-SSD |
| `MODEL_WEIGHTS` | `MobileNetSSD_deploy.caffemodel` | Weights file MobileNet-SSD |
| `CONF_THRESHOLD` | `0.5` | Confidence threshold deteksi |
| `LINE_POSITION` | `0.5` | Posisi garis vertikal (0-1) |
| `RESIZE_TO` | `(960, 540)` | Resolusi frame untuk processing |

### MQTT Configuration

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `broker` | `206.237.97.19` | Alamat MQTT broker |
| `port` | `1883` | Port MQTT |
| `username` | `urbansolv` | Username MQTT |
| `password` | `letsgosolv` | Password MQTT |
| `topic` | `entrance/device-1/data` | Topic untuk publish data |

## 🚀 Penggunaan

```bash
python main.py
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
- Kurangi resolusi (`RESIZE_WIDTH`, `RESIZE_HEIGHT`)
- Pastikan tidak ada proses berat lain berjalan

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
├── download_model.py       # Script download model
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (tidak di-commit)
├── .env.example            # Template environment variables
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
│   │   └── detector.py     # MobileNet-SSD detector
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
├── models/                 # Model files
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
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
