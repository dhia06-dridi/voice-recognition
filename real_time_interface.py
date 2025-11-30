# real_time_voice_detector_clean_design.py
import sys
import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import json
import warnings
from collections import deque
import time
import traceback

print("=" * 60)
print("🚀 STARTING REAL-TIME VOICE DETECTOR")
print("=" * 60)

try:
    import pyaudio
    print("✅ PyAudio imported")
except ImportError as e:
    print(f"❌ Missing PyAudio: pip install pyaudio")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    print("✅ PyQt6 imported")
except ImportError as e:
    print(f"❌ Missing PyQt6: pip install PyQt6")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime imported")
    print(f"   Providers: {ort.get_available_providers()}")
except ImportError as e:
    print(f"❌ Missing ONNX Runtime: pip install onnxruntime")
    input("Press Enter to exit...")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Check required files
print("\n📁 Checking required files...")
REQUIRED_FILES = {
    'speech_recognition_model_v1.onnx': 'ONNX model',
    'class_labels.json': 'Class labels'
}

missing_files = []
for filename, description in REQUIRED_FILES.items():
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"   ✅ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ {filename} - NOT FOUND")
        missing_files.append(filename)

if missing_files:
    print(f"\n❌ Missing files: {', '.join(missing_files)}")
    input("\nPress Enter to exit...")
    sys.exit(1)

print("\n" + "=" * 60)


class AudioRecorderThread(QThread):
    """Real-time audio recording thread"""
    audio_data_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self, sample_rate=16000, chunk_size=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_recording = False
        self.audio_buffer = deque(maxlen=sample_rate * 3)
        self.emit_interval = sample_rate // 4
        self.samples_since_emit = 0
        
    def run(self):
        self.is_recording = True
        p = None
        stream = None
        
        try:
            p = pyaudio.PyAudio()
            
            print("\n🎤 Available audio devices:")
            default_device = None
            try:
                default_device = p.get_default_input_device_info()
                print(f"   Default: {default_device['name']}")
            except Exception as e:
                print(f"   ⚠️ No default input device")
            
            device_index = default_device['index'] if default_device else None
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index
            )
            
            print(f"✅ Audio stream started (16kHz, mono)")
            
            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    self.audio_buffer.extend(audio_array)
                    self.samples_since_emit += len(audio_array)
                    
                    if self.samples_since_emit >= self.emit_interval:
                        if len(self.audio_buffer) >= self.sample_rate:
                            self.audio_data_signal.emit(np.array(self.audio_buffer))
                        self.samples_since_emit = 0
                        
                except Exception as e:
                    time.sleep(0.1)
                    
        except Exception as e:
            error_msg = f"Audio error: {e}"
            print(f"❌ {error_msg}")
            self.error_signal.emit(error_msg)
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            if p:
                try:
                    p.terminate()
                except:
                    pass
    
    def stop(self):
        self.is_recording = False
        self.wait(2000)


class GPUVoiceDetector:
    """GPU-accelerated voice detector"""
    
    def __init__(self, onnx_path, labels_path, confidence_threshold=0.6):
        print(f"\n🔧 Initializing detector...")
        
        self.confidence_threshold = confidence_threshold
        
        # Load labels
        try:
            with open(labels_path, 'r') as f:
                self.idx_to_label = json.load(f)
            self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
            self.num_classes = len(self.idx_to_label)
            print(f"✅ {self.num_classes} classes loaded")
            print(f"   Examples: {list(self.idx_to_label.values())[:10]}")
        except Exception as e:
            raise Exception(f"Failed to load labels: {e}")
        
        # Audio config
        self.sample_rate = 16000
        self.target_length = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        self.f_min = 50
        self.f_max = 8000
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  PyTorch device: {self.device}")
        
        # Transforms
        try:
            self.transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max
            ).to(self.device)
            
            self.to_db = T.AmplitudeToDB().to(self.device)
            print("✅ Audio transforms created")
        except Exception as e:
            raise Exception(f"Failed to create transforms: {e}")
        
        # Setup ONNX
        self._setup_onnx(onnx_path)
        
        # State
        self.smoothing_factor = 0.7  # Increased smoothing for smaller variations
        self.previous_probs = np.zeros(self.num_classes)
        self.vad_threshold = 0.005
        self.vad_history = deque(maxlen=5)
        self.noise_floor = 0.001
        self.last_detection_time = 0
        self.min_detection_interval = 0.5
        
        print("✅ Detector initialized!\n")
    
    def _setup_onnx(self, onnx_path):
        try:
            providers = []
            available = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available and torch.cuda.is_available():
                providers.append('CUDAExecutionProvider')
                print("✅ CUDA provider available")
            
            if 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')
                print("✅ DirectML provider available")
            
            providers.append('CPUExecutionProvider')
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self.session = ort.InferenceSession(
                onnx_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            actual = self.session.get_providers()
            print(f"🚀 ONNX using: {actual[0]}")
            
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            print(f"📐 Input shape: {self.input_shape}")
            
        except Exception as e:
            raise Exception(f"ONNX setup failed: {e}")
    
    def has_voice_activity(self, audio_data):
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        if len(self.vad_history) > 0:
            self.noise_floor = min(np.mean(self.vad_history), self.noise_floor * 1.01)
        
        self.vad_history.append(energy)
        return energy > (self.noise_floor + self.vad_threshold)
    
    def preprocess_audio(self, audio_data):
        try:
            waveform = torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)
            
            # Pre-emphasis
            pre_emphasis = 0.97
            waveform = torch.cat([
                waveform[:, :1], 
                waveform[:, 1:] - pre_emphasis * waveform[:, :-1]
            ], dim=1)
            
            waveform = waveform / (waveform.abs().max() + 1e-8)
            
            # Adjust length
            current_length = waveform.shape[1]
            if current_length < self.target_length:
                pad_amount = self.target_length - current_length
                waveform = F.pad(waveform, (0, pad_amount), mode='reflect')
            elif current_length > self.target_length:
                start = (current_length - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
            
            # Mel spectrogram
            mel_spec = self.transform(waveform)
            mel_spec_db = self.to_db(mel_spec)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Adjust to 32 frames
            current_frames = mel_spec_db.shape[2]
            if current_frames < 32:
                repeat_count = (32 + current_frames - 1) // current_frames
                mel_spec_db = mel_spec_db.repeat(1, 1, repeat_count)[:, :, :32]
            elif current_frames > 32:
                start = (current_frames - 32) // 2
                mel_spec_db = mel_spec_db[:, :, start:start + 32]
            
            # 3 channels
            mel_spec_3ch = mel_spec_db.repeat(3, 1, 1)
            final_tensor = mel_spec_3ch.unsqueeze(0)
            
            return final_tensor.cpu().numpy()
            
        except Exception as e:
            return np.zeros((1, 3, 128, 32), dtype=np.float32)
    
    def apply_temporal_smoothing(self, current_probs):
        if np.sum(self.previous_probs) == 0:
            self.previous_probs = current_probs
            return current_probs
        
        # Stronger smoothing for smaller variations
        smoothed = (self.smoothing_factor * self.previous_probs + 
                   (1 - self.smoothing_factor) * current_probs)
        self.previous_probs = smoothed
        return smoothed
    
    def predict(self, audio_data):
        try:
            if not self.has_voice_activity(audio_data):
                return None, 0.0, self.previous_probs, []
            
            input_tensor = self.preprocess_audio(audio_data)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            output = torch.tensor(outputs[0])
            
            probabilities = F.softmax(output, dim=1)
            current_probs = probabilities.numpy()[0]
            smoothed_probs = self.apply_temporal_smoothing(current_probs)
            
            probs_tensor = torch.tensor(smoothed_probs).unsqueeze(0)
            top_probs, top_indices = probs_tensor.topk(min(5, self.num_classes), dim=1)
            top_probs = top_probs.numpy()[0]
            top_indices = top_indices.numpy()[0]
            
            best_prob = top_probs[0]
            best_idx = top_indices[0]
            best_label = self.idx_to_label[best_idx]
            
            current_time = time.time()
            if current_time - self.last_detection_time < self.min_detection_interval:
                return None, best_prob, smoothed_probs, list(zip(top_indices, top_probs))
            
            if best_prob >= self.confidence_threshold:
                self.last_detection_time = current_time
                return best_label, best_prob, smoothed_probs, list(zip(top_indices, top_probs))
            else:
                return None, best_prob, smoothed_probs, list(zip(top_indices, top_probs))
                
        except Exception as e:
            return None, 0.0, self.previous_probs, []


class CompactClassWidget(QWidget):
    """Compact widget for a single class with small vertical progress bar"""
    
    COLORS = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
        "#27ae60", "#2980b9", "#8e44ad", "#f1c40f", "#d35400"
    ]
    
    def __init__(self, class_name, class_idx, probability=0.0):
        super().__init__()
        self.class_name = class_name
        self.class_idx = class_idx
        self.probability = probability
        self.is_detected = False
        self.color = self.COLORS[class_idx % len(self.COLORS)]
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        
        # Class name - compact
        self.name_label = QLabel(self.class_name)
        self.name_label.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("color: #ecf0f1; padding: 1px;")
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(30)
        
        # Compact VERTICAL Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # Hide text for compact size
        self.progress_bar.setFixedWidth(25)  # Very narrow
        self.progress_bar.setFixedHeight(60)  # Compact height
        self.progress_bar.setOrientation(Qt.Orientation.Vertical)
        
        # Small percentage label
        self.percent_label = QLabel("0%")
        self.percent_label.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percent_label.setStyleSheet("color: #ecf0f1; padding: 0px;")
        self.percent_label.setMaximumHeight(12)
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.percent_label)
        
        self.setLayout(layout)
        self.setFixedWidth(70)  # Very compact width
        self.setFixedHeight(110)  # Fixed height for all
        self.update_display()
        
    def update_probability(self, probability, is_detected=False):
        self.probability = probability
        self.is_detected = is_detected
        self.update_display()
        
    def update_display(self):
        percent = int(self.probability * 100)
        self.progress_bar.setValue(percent)
        self.percent_label.setText(f"{percent}%")
        
        if self.is_detected:
            self.setStyleSheet(f"""
                QWidget {{
                    background-color: rgba(46, 204, 113, 0.4);
                    border: 2px solid #2ecc71;
                    border-radius: 6px;
                }}
            """)
            self.name_label.setStyleSheet("color: #2ecc71; font-weight: bold; background: transparent;")
            self.percent_label.setStyleSheet("color: #2ecc71; font-weight: bold; background: transparent;")
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #2ecc71;
                    border-radius: 3px;
                    background-color: #1a1a1a;
                }}
                QProgressBar::chunk {{
                    background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
                        stop:0 #2ecc71, stop:1 #27ae60);
                    border-radius: 2px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QWidget {{
                    background-color: #2c3e50;
                    border: 1px solid #34495e;
                    border-radius: 6px;
                }}
            """)
            self.name_label.setStyleSheet("color: #ecf0f1; font-weight: normal; background: transparent;")
            self.percent_label.setStyleSheet("color: #ecf0f1; font-weight: normal; background: transparent;")
            
            # Simpler color based on percentage
            if percent >= 70:
                bar_color = "#2ecc71"
            elif percent >= 40:
                bar_color = "#f39c12"
            elif percent >= 20:
                bar_color = "#e74c3c"
            else:
                bar_color = "#7f8c8d"
            
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #34495e;
                    border-radius: 3px;
                    background-color: #1a1a1a;
                }}
                QProgressBar::chunk {{
                    background-color: {bar_color};
                    border-radius: 2px;
                }}
            """)


class NotificationWidget(QWidget):
    """Floating notification widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        self.icon_label = QLabel("🎯")
        self.icon_label.setFont(QFont("Arial", 18))
        
        self.text_label = QLabel()
        self.text_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.text_label.setStyleSheet("color: white;")
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(46, 204, 113, 0.95), 
                    stop:1 rgba(39, 174, 96, 0.95));
                border: 2px solid #27ae60;
                border-radius: 8px;
            }
        """)
        
    def show_detection(self, word, confidence):
        self.text_label.setText(f"  {word.upper()}  ({confidence:.0f}%)")
        
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - self.width() - 20, 20)
        
        self.show()
        self.raise_()
        QTimer.singleShot(2000, self.hide)


class RealTimeVoiceDetectorUI(QMainWindow):
    """Main UI with compact layout showing all classes together"""
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.recorder = None
        self.is_monitoring = False
        
        self.last_prediction = None
        self.last_confidence = 0
        self.last_all_probabilities = np.array([])
        self.last_detected_class_idx = None
        
        self.class_widgets = {}
        self.notification_widget = NotificationWidget()
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_audio_level = 0
        
        print("\n🎨 Initializing UI...")
        self.init_ui()
        self.load_model()
        print("✅ UI initialized\n")
        
    def init_ui(self):
        self.setWindowTitle("🎤 Real-Time Voice Detector — Compact Display")
        self.setGeometry(50, 50, 1000, 700)  # Smaller window
        
        # Dark professional theme
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #0f3460;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: rgba(15, 52, 96, 0.3);
                color: #e94560;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #e94560;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #d63447);
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5577, stop:1 #e94560);
            }
            QPushButton:pressed {
                background: #d63447;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 10px;
            }
            QTextEdit {
                background-color: #0f0f1e;
                color: #2ecc71;
                border: 1px solid #0f3460;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New';
                font-size: 9px;
                padding: 5px;
            }
            QSpinBox {
                background-color: #16213e;
                color: #ecf0f1;
                border: 1px solid #0f3460;
                border-radius: 4px;
                padding: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QScrollArea {
                background-color: transparent;
                border: 1px solid #0f3460;
                border-radius: 8px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:0.5 #0f3460, stop:1 #e94560);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        header_layout = QHBoxLayout(header)
        
        title = QLabel("🎤 VOICE DETECTOR")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        main_layout.addWidget(header)
        
        # Status bar - compact
        status_group = QGroupBox("📊 STATUS")
        status_layout = QGridLayout()
        status_layout.setSpacing(6)
        
        self.status_label = QLabel("🔴 Stopped")
        self.status_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #e74c3c;")
        status_layout.addWidget(self.status_label, 0, 0)
        
        self.audio_level_label = QLabel("🔊: 0%")
        self.audio_level_label.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.audio_level_label, 0, 1)
        
        self.fps_label = QLabel("📊: 0")
        self.fps_label.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.fps_label, 0, 2)
        
        self.gpu_label = QLabel("🖥️: CPU")
        self.gpu_label.setFont(QFont("Segoe UI", 9))
        status_layout.addWidget(self.gpu_label, 0, 3)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Controls - compact
        control_group = QGroupBox("🎛️ CONTROLS")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        self.start_btn = QPushButton("🚀 START")
        self.start_btn.setMinimumHeight(35)
        self.start_btn.clicked.connect(self.toggle_monitoring)
        control_layout.addWidget(self.start_btn)
        
        control_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(10, 95)
        self.threshold_spin.setValue(60)
        self.threshold_spin.setSuffix("%")
        self.threshold_spin.setMinimumWidth(60)
        self.threshold_spin.valueChanged.connect(self.update_threshold)
        control_layout.addWidget(self.threshold_spin)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Current detection - compact
        detection_group = QGroupBox("🎯 DETECTION")
        detection_layout = QVBoxLayout()
        
        self.current_prediction = QLabel("Click START")
        self.current_prediction.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.current_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_prediction.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(15, 52, 96, 0.5), 
                    stop:1 rgba(26, 26, 46, 0.5));
                color: #3498db;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #0f3460;
            }
        """)
        self.current_prediction.setMinimumHeight(60)
        detection_layout.addWidget(self.current_prediction)
        
        detection_group.setLayout(detection_layout)
        main_layout.addWidget(detection_group)
        
        # All classes — COMPACT GRID LAYOUT
        classes_group = QGroupBox("📈 ALL CLASSES")
        classes_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(250)  # Smaller height
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #0f3460; 
                border-radius: 6px;
                background-color: rgba(15, 52, 96, 0.1);
            }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #e94560;
                border-radius: 5px;
                min-height: 15px;
            }
        """)

        self.classes_container = QWidget()
        self.classes_container.setStyleSheet("background-color: transparent;")
        
        # Use grid layout for compact arrangement
        self.classes_layout = QGridLayout(self.classes_container)
        self.classes_layout.setSpacing(8)
        self.classes_layout.setContentsMargins(10, 10, 10, 10)

        scroll_area.setWidget(self.classes_container)
        classes_layout.addWidget(scroll_area)
        classes_group.setLayout(classes_layout)
        main_layout.addWidget(classes_group)
        
        # Detection log - compact
        log_group = QGroupBox("📝 LOG")
        log_layout = QVBoxLayout()
        
        self.detection_log = QTextEdit()
        self.detection_log.setMaximumHeight(80)
        self.detection_log.setReadOnly(True)
        log_layout.addWidget(self.detection_log)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # UI update timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(33)
    
    def create_class_widgets(self):
        """Create compact grid of class widgets"""
        while self.classes_layout.count():
            child = self.classes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.class_widgets = {}

        if not self.detector:
            return

        # Calculate grid dimensions
        num_classes = self.detector.num_classes
        cols = 8  # Fixed number of columns for compact layout
        rows = (num_classes + cols - 1) // cols  # Calculate rows needed

        # Create class widgets in grid
        for class_idx in range(num_classes):
            class_name = self.detector.idx_to_label[class_idx]
            widget = CompactClassWidget(class_name, class_idx, 0.0)
            self.class_widgets[class_idx] = widget
            
            # Calculate grid position
            row = class_idx // cols
            col = class_idx % cols
            self.classes_layout.addWidget(widget, row, col)

        print(f"✅ {len(self.class_widgets)} class widgets created (compact grid layout)")
    
    def load_model(self):
        """Load ONNX model"""
        try:
            self.detector = GPUVoiceDetector(
                onnx_path='speech_recognition_model_v1.onnx',
                labels_path='class_labels.json',
                confidence_threshold=0.6
            )
            
            self.log_message("✅ Model loaded successfully!")
            self.log_message(f"📊 Classes: {self.detector.num_classes}")
            
            if torch.cuda.is_available():
                self.gpu_label.setText("🖥️: CUDA ⚡")
                self.gpu_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            else:
                self.gpu_label.setText("🖥️: CPU")
                self.gpu_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            
            self.create_class_widgets()
            
        except Exception as e:
            self.log_message(f"❌ Model loading error: {e}")
            print(f"\n❌ Full error:\n{traceback.format_exc()}")
    
    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        if not self.detector:
            self.log_message("❌ Model not loaded!")
            return
        
        try:
            self.recorder = AudioRecorderThread()
            self.recorder.audio_data_signal.connect(self.process_audio)
            self.recorder.error_signal.connect(self.handle_audio_error)
            self.recorder.start()
            
            self.is_monitoring = True
            self.start_btn.setText("⏹️ STOP")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #c0392b, stop:1 #8e44ad);
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 6px;
                }
            """)
            
            self.status_label.setText("🟢 LISTENING")
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.log_message("🎙️ Detection started")
            
        except Exception as e:
            self.log_message(f"❌ Start error: {e}")
            print(traceback.format_exc())
    
    def stop_monitoring(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        
        self.is_monitoring = False
        self.start_btn.setText("🚀 START")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("🔴 Stopped")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.log_message("⏹️ Detection stopped")
    
    def handle_audio_error(self, error_msg):
        self.log_message(f"❌ Audio: {error_msg}")
        self.stop_monitoring()
    
    def process_audio(self, audio_data):
        if self.detector:
            self.last_audio_level = np.sqrt(np.mean(audio_data ** 2))
            prediction, confidence, all_probs, top_predictions = self.detector.predict(audio_data)
            self.last_all_probabilities = all_probs
            self.last_prediction = prediction
            self.last_confidence = confidence
            
            if prediction:
                for idx, label in self.detector.idx_to_label.items():
                    if label == prediction:
                        self.last_detected_class_idx = idx
                        break
                
                # Show notification for new detection
                if prediction != getattr(self, 'last_notified_prediction', None):
                    self.last_notified_prediction = prediction
                    self.show_notification(prediction, confidence)
    
    def update_ui(self):
        # FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_label.setText(f"📊: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Audio level
        audio_level_percent = min(int(self.last_audio_level * 500), 100)
        self.audio_level_label.setText(f"🔊: {audio_level_percent}%")
        
        # Current prediction
        if self.last_prediction:
            self.current_prediction.setText(
                f"{self.last_prediction.upper()}\n"
                f"{self.last_confidence*100:.1f}%"
            )
            self.current_prediction.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(46, 204, 113, 0.4), 
                        stop:1 rgba(39, 174, 96, 0.4));
                    color: #2ecc71;
                    padding: 15px;
                    border-radius: 8px;
                    border: 2px solid #2ecc71;
                    font-size: 14px;
                }
            """)
        else:
            if self.is_monitoring:
                self.current_prediction.setText("LISTENING...\nSpeak clearly")
            else:
                self.current_prediction.setText("Click START")
            self.current_prediction.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(15, 52, 96, 0.5), 
                        stop:1 rgba(26, 26, 46, 0.5));
                    color: #3498db;
                    padding: 15px;
                    border-radius: 8px;
                    border: 2px solid #0f3460;
                }
            """)
        
        # Update all class probabilities
        if len(self.last_all_probabilities) > 0:
            for class_idx, widget in self.class_widgets.items():
                if class_idx < len(self.last_all_probabilities):
                    prob = self.last_all_probabilities[class_idx]
                    is_detected = (class_idx == self.last_detected_class_idx)
                    widget.update_probability(prob, is_detected)
        
        # Clear highlight after delay
        if self.last_detected_class_idx is not None:
            QTimer.singleShot(1500, lambda: setattr(self, 'last_detected_class_idx', None))
    
    def show_notification(self, word, confidence):
        timestamp = time.strftime("%H:%M:%S")
        message = f"[{timestamp}] 🎯 {word.upper()} → {confidence*100:.1f}%"
        self.log_message(message)
        self.notification_widget.show_detection(word, confidence * 100)
        
        # Flash border
        original_style = self.styleSheet()
        self.setStyleSheet(original_style + " QMainWindow { border: 3px solid #2ecc71; }")
        QTimer.singleShot(500, lambda: self.setStyleSheet(original_style))
        
        QApplication.beep()
    
    def update_threshold(self, value):
        if self.detector:
            self.detector.confidence_threshold = value / 100.0
            self.log_message(f"🔧 Threshold: {value}%")
    
    def log_message(self, message):
        self.detection_log.append(message)
        scrollbar = self.detection_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        print("\n🛑 Shutting down...")
        if self.is_monitoring:
            self.stop_monitoring()
        event.accept()


def main():
    try:
        print("\n🚀 Launching application...")
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 46))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(236, 240, 241))
        palette.setColor(QPalette.ColorRole.Base, QColor(15, 15, 30))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(22, 33, 62))
        palette.setColor(QPalette.ColorRole.Text, QColor(236, 240, 241))
        palette.setColor(QPalette.ColorRole.Button, QColor(15, 52, 96))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(236, 240, 241))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(233, 69, 96))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        app.setPalette(palette)
        
        print("✅ QApplication created")
        
        window = RealTimeVoiceDetectorUI()
        window.show()
        
        print("✅ Window displayed")
        print("\n" + "=" * 60)
        print("🎤 APPLICATION READY!")
        print("=" * 60)
        print("💡 Click 'START' and speak command words")
        print("📊 Compact vertical bars - all classes visible")
        print("🎨 Small variations with strong smoothing")
        print("🔔 Visual & audio alerts on detection")
        print("=" * 60 + "\n")
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        print(traceback.format_exc())
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()