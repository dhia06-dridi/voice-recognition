# real_time_voice_detector_mobilenetv3.py
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
print("🚀 DÉMARRAGE DU DÉTECTEUR VOCAL - MobileNetV3")
print("=" * 60)

try:
    import pyaudio
    print("✅ PyAudio importé")
except ImportError as e:
    print(f"❌ PyAudio manquant: pip install pyaudio")
    input("Appuyez sur Entrée...")
    sys.exit(1)

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    print("✅ PyQt6 importé")
except ImportError as e:
    print(f"❌ PyQt6 manquant: pip install PyQt6")
    input("Appuyez sur Entrée...")
    sys.exit(1)

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime importé")
    print(f"   Providers: {ort.get_available_providers()}")
except ImportError as e:
    print(f"❌ ONNX Runtime manquant: pip install onnxruntime")
    input("Appuyez sur Entrée...")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Vérifier les fichiers requis
print("\n📁 Vérification des fichiers...")
REQUIRED_FILES = {
    'D:\\voice_processing\\mobilnet_v3\\mobilnet_100epoch\\speech_recognition_model.onnx': 'Modèle ONNX MobileNetV3',
    'class_labels.json': 'Labels des classes'
}

missing_files = []
for filename, description in REQUIRED_FILES.items():
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"   ✅ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ {filename} - NON TROUVÉ")
        missing_files.append(filename)

if missing_files:
    print(f"\n❌ Fichiers manquants: {', '.join(missing_files)}")
    input("\nAppuyez sur Entrée...")
    sys.exit(1)

print("\n" + "=" * 60)


class AudioRecorderThread(QThread):
    """Thread pour l'enregistrement audio en temps réel"""
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
            
            print("\n🎤 Périphériques audio disponibles:")
            default_device = None
            try:
                default_device = p.get_default_input_device_info()
                print(f"   Par défaut: {default_device['name']}")
            except Exception as e:
                print(f"   ⚠️ Pas de périphérique par défaut")
            
            device_index = default_device['index'] if default_device else None
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index
            )
            
            print(f"✅ Stream audio démarré (16kHz, mono)")
            
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
            error_msg = f"Erreur audio: {e}"
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
    """Détecteur vocal avec accélération GPU - MobileNetV3"""
    
    def __init__(self, onnx_path, labels_path, confidence_threshold=0.6):
        print(f"\n🔧 Initialisation du détecteur MobileNetV3...")
        
        self.confidence_threshold = confidence_threshold
        
        # Charger les labels
        try:
            with open(labels_path, 'r') as f:
                self.idx_to_label = json.load(f)
            self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
            self.num_classes = len(self.idx_to_label)
            print(f"✅ {self.num_classes} classes chargées")
            print(f"   Exemples: {list(self.idx_to_label.values())[:10]}")
        except Exception as e:
            raise Exception(f"Échec chargement labels: {e}")
        
        # Configuration audio
        self.sample_rate = 16000
        self.target_length = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        self.f_min = 50
        self.f_max = 8000
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device PyTorch: {self.device}")
        
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
            print("✅ Transforms audio créés")
        except Exception as e:
            raise Exception(f"Échec création transforms: {e}")
        
        # Setup ONNX
        self._setup_onnx(onnx_path)
        
        # État
        self.smoothing_factor = 0.4
        self.previous_probs = np.zeros(self.num_classes)
        self.vad_threshold = 0.005
        self.vad_history = deque(maxlen=5)
        self.noise_floor = 0.001
        self.last_detection_time = 0
        self.min_detection_interval = 0.5
        
        print("✅ Détecteur MobileNetV3 initialisé!\n")
    
    def _setup_onnx(self, onnx_path):
        try:
            providers = []
            available = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available and torch.cuda.is_available():
                providers.append('CUDAExecutionProvider')
                print("✅ Provider CUDA disponible")
            
            if 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')
                print("✅ Provider DirectML disponible")
            
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
            print(f"🚀 ONNX utilise: {actual[0]}")
            
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            print(f"📐 Shape d'entrée MobileNetV3: {self.input_shape}")
            
            # Adapter le prétraitement selon la shape d'entrée
            if len(self.input_shape) == 4:
                self.target_height = self.input_shape[2]
                self.target_width = self.input_shape[3]
                print(f"🎯 Dimensions cible: {self.target_height}x{self.target_width}")
            else:
                # Dimensions par défaut pour MobileNetV3
                self.target_height = 224
                self.target_width = 224
                print(f"🎯 Dimensions par défaut: {self.target_height}x{self.target_width}")
            
        except Exception as e:
            raise Exception(f"Échec setup ONNX: {e}")
    
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
            
            # Ajuster longueur
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
            
            # REDIMENSIONNEMENT POUR MobileNetV3
            # MobileNetV3 attend généralement 224x224 (comme ImageNet)
            mel_spec_db_resized = F.interpolate(
                mel_spec_db.unsqueeze(0), 
                size=(self.target_height, self.target_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # 3 canaux pour MobileNetV3 (RGB)
            mel_spec_3ch = mel_spec_db_resized.repeat(3, 1, 1)
            final_tensor = mel_spec_3ch.unsqueeze(0)
            
            return final_tensor.cpu().numpy()
            
        except Exception as e:
            print(f"❌ Erreur prétraitement: {e}")
            return np.zeros((1, 3, self.target_height, self.target_width), dtype=np.float32)
    
    def apply_temporal_smoothing(self, current_probs):
        if np.sum(self.previous_probs) == 0:
            self.previous_probs = current_probs
            return current_probs
        
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
            print(f"❌ Erreur prédiction: {e}")
            return None, 0.0, self.previous_probs, []


class ModernClassWidget(QWidget):
    """Widget moderne pour une classe avec graphique en barres coloré"""
    
    # Palette de couleurs professionnelles
    COLORS = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
        "#27ae60", "#2980b9", "#8e44ad", "#f1c40f", "#d35400",
        "#7f8c8d", "#1f618d", "#148f77", "#b03a2e", "#6c3483"
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
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Nom de la classe
        self.name_label = QLabel(self.class_name)
        self.name_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("color: #ecf0f1; padding: 2px;")
        
        # Barre de progression moderne
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setFormat("%p%")
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        self.update_display()
        
    def update_probability(self, probability, is_detected=False):
        self.probability = probability
        self.is_detected = is_detected
        self.update_display()
        
    def update_display(self):
        percent = int(self.probability * 100)
        self.progress_bar.setValue(percent)
        
        if self.is_detected:
            # Animation de détection - vert vif avec glow
            self.setStyleSheet(f"""
                QWidget {{
                    background-color: rgba(46, 204, 113, 0.3);
                    border: 2px solid #2ecc71;
                    border-radius: 6px;
                    padding: 4px;
                }}
            """)
            self.name_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid #2ecc71;
                    border-radius: 4px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 9px;
                    background-color: #1a1a1a;
                }}
                QProgressBar::chunk {{
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #2ecc71, stop:1 #27ae60);
                    border-radius: 3px;
                }}
            """)
        else:
            # Couleur normale selon le pourcentage
            self.setStyleSheet(f"""
                QWidget {{
                    background-color: #2c3e50;
                    border: 1px solid #34495e;
                    border-radius: 6px;
                    padding: 4px;
                }}
            """)
            self.name_label.setStyleSheet("color: #ecf0f1;")
            
            # Gradient de couleur basé sur la probabilité
            if percent >= 50:
                gradient_color = self.color
                opacity = 1.0
            elif percent >= 20:
                gradient_color = self.color
                opacity = 0.7
            else:
                gradient_color = "#7f8c8d"
                opacity = 0.4
            
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #34495e;
                    border-radius: 4px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 9px;
                    background-color: #1a1a1a;
                }}
                QProgressBar::chunk {{
                    background-color: {gradient_color};
                    opacity: {opacity};
                    border-radius: 3px;
                }}
            """)


class NotificationWidget(QWidget):
    """Widget de notification flottante moderne"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)
        
        self.icon_label = QLabel("🎯")
        self.icon_label.setFont(QFont("Arial", 24))
        
        self.text_label = QLabel()
        self.text_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
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
                border-radius: 12px;
            }
        """)
        
    def show_detection(self, word, confidence):
        self.text_label.setText(f"  {word.upper()}  ({confidence:.0f}%)")
        
        # Position en haut à droite
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - self.width() - 20, 20)
        
        self.show()
        self.raise_()
        
        # Animation de disparition après 2 secondes
        QTimer.singleShot(2000, self.hide)


class RealTimeVoiceDetectorUI(QMainWindow):
    """Interface principale avec design moderne"""
    
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
        
        print("\n🎨 Initialisation de l'interface...")
        self.init_ui()
        self.load_model()
        print("✅ Interface initialisée\n")
        
    def init_ui(self):
        self.setWindowTitle("🎤 Détecteur Vocal Temps Réel - MobileNetV3")
        self.setGeometry(50, 50, 1400, 900)
        
        # Thème moderne sombre
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #0f3460;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: rgba(15, 52, 96, 0.3);
                color: #e94560;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #e94560;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #d63447);
                border: none;
                color: white;
                padding: 14px 28px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 8px;
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
                font-size: 12px;
            }
            QTextEdit {
                background-color: #0f0f1e;
                color: #2ecc71;
                border: 2px solid #0f3460;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New';
                font-size: 11px;
                padding: 8px;
            }
            QSpinBox {
                background-color: #16213e;
                color: #ecf0f1;
                border: 2px solid #0f3460;
                border-radius: 6px;
                padding: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #0f3460;
                border-radius: 3px;
            }
            QScrollArea {
                background-color: transparent;
                border: 2px solid #0f3460;
                border-radius: 10px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # En-tête avec gradient
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:0.5 #0f3460, stop:1 #e94560);
                border-radius: 12px;
                padding: 10px;
            }
        """)
        header_layout = QHBoxLayout(header)
        
        title = QLabel("🎤 DÉTECTEUR VOCAL TEMPS RÉEL - MobileNetV3")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        main_layout.addWidget(header)
        
        # Barre de statut
        status_group = QGroupBox("📊 STATUT DU SYSTÈME")
        status_layout = QGridLayout()
        status_layout.setSpacing(10)
        
        self.status_label = QLabel("🔴 Arrêté")
        self.status_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #e74c3c;")
        status_layout.addWidget(self.status_label, 0, 0)
        
        self.audio_level_label = QLabel("🔊 Audio: 0%")
        self.audio_level_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.audio_level_label, 0, 1)
        
        self.fps_label = QLabel("📊 FPS: 0")
        self.fps_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.fps_label, 0, 2)
        
        self.gpu_label = QLabel("🖥️ Device: CPU")
        self.gpu_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.gpu_label, 0, 3)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Contrôles
        control_group = QGroupBox("🎛️ CONTRÔLES")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)
        
        self.start_btn = QPushButton("🚀 DÉMARRER LA DÉTECTION")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self.toggle_monitoring)
        control_layout.addWidget(self.start_btn)
        
        control_layout.addWidget(QLabel("Seuil de confiance:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(10, 95)
        self.threshold_spin.setValue(60)
        self.threshold_spin.setSuffix("%")
        self.threshold_spin.setMinimumWidth(80)
        self.threshold_spin.valueChanged.connect(self.update_threshold)
        control_layout.addWidget(self.threshold_spin)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Zone de détection actuelle
        detection_group = QGroupBox("🎯 DÉTECTION EN COURS")
        detection_layout = QVBoxLayout()
        
        self.current_prediction = QLabel("Cliquez sur DÉMARRER et parlez")
        self.current_prediction.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.current_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_prediction.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(15, 52, 96, 0.5), 
                    stop:1 rgba(26, 26, 46, 0.5));
                color: #3498db;
                padding: 30px;
                border-radius: 15px;
                border: 3px solid #0f3460;
            }
        """)
        self.current_prediction.setMinimumHeight(100)
        detection_layout.addWidget(self.current_prediction)
        
        detection_group.setLayout(detection_layout)
        main_layout.addWidget(detection_group)
        
        # Toutes les classes - Grid moderne
        classes_group = QGroupBox("📈 TOUTES LES CLASSES - PROBABILITÉS EN TEMPS RÉEL")
        classes_layout = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(380)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: transparent;")
        self.classes_grid = QGridLayout(scroll_widget)
        self.classes_grid.setSpacing(8)
        self.classes_grid.setContentsMargins(10, 10, 10, 10)
        
        scroll_area.setWidget(scroll_widget)
        classes_layout.addWidget(scroll_area)
        classes_group.setLayout(classes_layout)
        main_layout.addWidget(classes_group)
        
        # Journal de détection
        log_group = QGroupBox("📝 JOURNAL DES DÉTECTIONS")
        log_layout = QVBoxLayout()
        
        self.detection_log = QTextEdit()
        self.detection_log.setMaximumHeight(110)
        self.detection_log.setReadOnly(True)
        log_layout.addWidget(self.detection_log)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Timer d'interface (30 FPS pour des animations fluides)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(33)
        
    def create_class_widgets(self):
        """Créer les widgets pour toutes les classes"""
        # Nettoyer
        for i in reversed(range(self.classes_grid.count())): 
            widget = self.classes_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.class_widgets = {}
        
        if not self.detector:
            return
        
        # Créer grid (5 colonnes pour plus de classes visibles)
        num_columns = 5
        row = 0
        col = 0
        
        for class_idx in range(self.detector.num_classes):
            class_name = self.detector.idx_to_label[class_idx]
            widget = ModernClassWidget(class_name, class_idx, 0.0)
            self.class_widgets[class_idx] = widget
            self.classes_grid.addWidget(widget, row, col)
            
            col += 1
            if col >= num_columns:
                col = 0
                row += 1
        
        print(f"✅ {len(self.class_widgets)} widgets de classe créés")
    
    def load_model(self):
        """Charger le modèle ONNX MobileNetV3"""
        try:
            self.detector = GPUVoiceDetector(
                onnx_path='D:/voice_processing/mobilnet_v3/speech_recognition_model.onnx',
                labels_path='class_labels.json',
                confidence_threshold=0.6
            )
            
            self.log_message("✅ Modèle MobileNetV3 chargé avec succès!")
            self.log_message(f"📊 Nombre de classes: {self.detector.num_classes}")
            self.log_message(f"🎯 Dimensions d'entrée: {self.detector.target_height}x{self.detector.target_width}")
            
            # Mettre à jour l'indicateur GPU
            if torch.cuda.is_available():
                self.gpu_label.setText("🖥️ Device: CUDA GPU ⚡")
                self.gpu_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            else:
                self.gpu_label.setText("🖥️ Device: CPU")
                self.gpu_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            
            self.create_class_widgets()
            
        except Exception as e:
            self.log_message(f"❌ Erreur de chargement: {e}")
            print(f"\n❌ Erreur complète:\n{traceback.format_exc()}")
    
    def toggle_monitoring(self):
        """Basculer la surveillance"""
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Démarrer la surveillance"""
        if not self.detector:
            self.log_message("❌ Modèle non chargé!")
            return
        
        try:
            self.recorder = AudioRecorderThread()
            self.recorder.audio_data_signal.connect(self.process_audio)
            self.recorder.error_signal.connect(self.handle_audio_error)
            self.recorder.start()
            
            self.is_monitoring = True
            self.start_btn.setText("⏹️ ARRÊTER LA DÉTECTION")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #c0392b, stop:1 #8e44ad);
                    border: none;
                    color: white;
                    padding: 14px 28px;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #e74c3c, stop:1 #9b59b6);
                }
            """)
            
            self.status_label.setText("🟢 EN ÉCOUTE...")
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.log_message("🎙️ Détection démarrée - Parlez maintenant!")
            
        except Exception as e:
            self.log_message(f"❌ Erreur démarrage: {e}")
            print(traceback.format_exc())
    
    def stop_monitoring(self):
        """Arrêter la surveillance"""
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        
        self.is_monitoring = False
        self.start_btn.setText("🚀 DÉMARRER LA DÉTECTION")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("🔴 Arrêté")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.log_message("⏹️ Détection arrêtée")
    
    def handle_audio_error(self, error_msg):
        """Gérer les erreurs audio"""
        self.log_message(f"❌ Audio: {error_msg}")
        self.stop_monitoring()
    
    def process_audio(self, audio_data):
        """Traiter les données audio"""
        if self.detector:
            self.last_audio_level = np.sqrt(np.mean(audio_data ** 2))
            
            start_time = time.time()
            prediction, confidence, all_probs, top_predictions = self.detector.predict(audio_data)
            
            self.last_all_probabilities = all_probs
            self.last_prediction = prediction
            self.last_confidence = confidence
            
            if prediction:
                self.show_notification(prediction, confidence)
                # Trouver l'index de la classe détectée
                for idx, label in self.detector.idx_to_label.items():
                    if label == prediction:
                        self.last_detected_class_idx = idx
                        break
    
    def update_ui(self):
        """Mettre à jour l'interface"""
        # FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_label.setText(f"📊 FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Niveau audio
        audio_level_percent = min(int(self.last_audio_level * 500), 100)
        self.audio_level_label.setText(f"🔊 Audio: {audio_level_percent}%")
        
        # Affichage de la détection
        if self.last_prediction:
            self.current_prediction.setText(
                f"🎯 {self.last_prediction.upper()}\n"
                f"Confiance: {self.last_confidence*100:.1f}%"
            )
            self.current_prediction.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(46, 204, 113, 0.4), 
                        stop:1 rgba(39, 174, 96, 0.4));
                    color: #2ecc71;
                    padding: 30px;
                    border-radius: 15px;
                    border: 3px solid #2ecc71;
                    font-size: 24px;
                }
            """)
        else:
            if self.is_monitoring:
                self.current_prediction.setText("🔍 EN ÉCOUTE...\nParlez clairement")
            else:
                self.current_prediction.setText("Cliquez sur DÉMARRER et parlez")
            self.current_prediction.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 rgba(15, 52, 96, 0.5), 
                        stop:1 rgba(26, 26, 46, 0.5));
                    color: #3498db;
                    padding: 30px;
                    border-radius: 15px;
                    border: 3px solid #0f3460;
                }
            """)
        
        # Mettre à jour toutes les probabilités des classes
        if len(self.last_all_probabilities) > 0:
            for class_idx, widget in self.class_widgets.items():
                if class_idx < len(self.last_all_probabilities):
                    probability = self.last_all_probabilities[class_idx]
                    is_detected = (class_idx == self.last_detected_class_idx)
                    widget.update_probability(probability, is_detected)
        
        # Réinitialiser la surbrillance après un délai
        if self.last_detected_class_idx is not None:
            QTimer.singleShot(1500, lambda: setattr(self, 'last_detected_class_idx', None))
    
    def show_notification(self, word, confidence):
        """Afficher une notification moderne"""
        timestamp = time.strftime("%H:%M:%S")
        message = f"[{timestamp}] 🎯 MOT DÉTECTÉ: '{word.upper()}' → {confidence*100:.1f}%"
        self.log_message(message)
        
        # Notification flottante
        self.notification_widget.show_detection(word, confidence * 100)
        
        # Animation de flash de fenêtre
        original_style = self.styleSheet()
        flash_style = original_style + """
            QMainWindow {
                border: 4px solid #2ecc71;
            }
        """
        self.setStyleSheet(flash_style)
        QTimer.singleShot(500, lambda: self.setStyleSheet(original_style))
        
        # Effet sonore système (beep)
        QApplication.beep()
    
    def update_threshold(self, value):
        """Mettre à jour le seuil de confiance"""
        if self.detector:
            self.detector.confidence_threshold = value / 100.0
            self.log_message(f"🔧 Seuil de confiance: {value}%")
    
    def log_message(self, message):
        """Ajouter un message au journal"""
        self.detection_log.append(message)
        scrollbar = self.detection_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Gérer la fermeture de la fenêtre"""
        print("\n🛑 Arrêt en cours...")
        if self.is_monitoring:
            self.stop_monitoring()
        event.accept()


def main():
    """Point d'entrée principal"""
    try:
        print("\n🚀 Démarrage de l'application MobileNetV3...")
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Palette sombre personnalisée
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
        
        print("✅ QApplication créée")
        
        window = RealTimeVoiceDetectorUI()
        window.show()
        
        print("✅ Fenêtre affichée")
        print("\n" + "=" * 60)
        print("🎤 APPLICATION MOBILENETV3 PRÊTE!")
        print("=" * 60)
        print("💡 Cliquez sur 'DÉMARRER' et prononcez des mots de commande")
        print("🎨 Design moderne avec couleurs distinctes par classe")
        print("🔔 Notifications visuelles et sonores à chaque détection")
        print("🤖 Modèle: MobileNetV3 optimisé pour la reconnaissance vocale")
        print("=" * 60 + "\n")
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        print(traceback.format_exc())
        input("\nAppuyez sur Entrée pour quitter...")
        sys.exit(1)


if __name__ == "__main__":
    main()