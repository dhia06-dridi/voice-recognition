import importlib

libs = [
    "torch",
    "torchaudio",
    "torchvision",
    "timm",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "tqdm",
    "onnxruntime",
    "PyQt6",
    "pyaudio"
]

for lib in libs:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "Version non disponible")
        print(f"{lib} : {version}")
    except Exception as e:
        print(f"{lib} : non installé")
