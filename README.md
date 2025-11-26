# IsoFace 🧑‍🤝‍🧑

**CPU-optimized face clustering using ArcFace embeddings and DBSCAN**

IsoFace automatically groups photos of the same person together using state-of-the-art face recognition technology. It runs entirely on your CPU with ONNX Runtime optimization - no GPU required!

## How It Works

1. **Detect** - Find faces using RetinaFace (best-in-class detector)
2. **Embed** - Extract 512-dimensional face embeddings using ArcFace
3. **Cluster** - Group similar faces using DBSCAN (no need to specify number of people!)

## Features

- **CPU Optimized** - Uses ONNX Runtime for 2-5x faster inference
- **High Accuracy** - Uses `buffalo_l` model (ArcFace-ResNet50)
- **Auto Clustering** - DBSCAN automatically discovers the number of people
- **Auto Organize** - Automatically sorts photos into folders by person
- **100% Local** - No cloud, no API calls, everything runs on your machine

## Installation

```bash
# Clone and setup
git clone https://github.com/990aa/isoface
cd isoface

# Create virtual environment with uv
uv venv --python 3.12
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv sync
```

## Quick Start

### Test with LFW Dataset (Kaggle)

```bash
# Run with default settings (downloads LFW dataset automatically)
uv run python main.py

# Process more people
uv run python main.py --max-people 50

# Process entire dataset
uv run python main.py --all
```

### Use Your Own Photos

```bash
# Process a folder of photos
uv run python main.py --input /path/to/photos --output organized_photos
```

## CLI Options

```
usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--eps EPS]
               [--min-samples MIN_SAMPLES] [--max-people MAX_PEOPLE] [--all]

options:
  -h, --help            show this help message and exit
  --input, -i           Input directory with images (downloads LFW if not specified)
  --output, -o          Output directory for clustered photos (default: clustered_faces)
  --eps                 DBSCAN eps threshold (default: 0.5)
                        - Lower (0.4) = stricter matching (may split same person)
                        - Higher (0.6) = looser matching (may merge different people)
  --min-samples         Minimum photos to form a group (default: 3)
  --max-people          Limit to first N people for demo (default: 10)
  --all                 Process all images (ignore --max-people)
```

## Tuning Tips

### The `eps` Parameter

This is the most important tuning parameter:

| Problem | Solution |
|---------|----------|
| Different people merged into one folder | **Decrease** eps (try 0.45, 0.40) |
| Same person split into multiple folders | **Increase** eps (try 0.55, 0.60) |

### The `min_samples` Parameter

- Lower values (2) = More clusters, including singletons
- Higher values (5+) = Only well-represented people get clusters

## Output Structure

```
clustered_faces/
├── Person_000/        # First identified person
│   ├── photo1.jpg
│   └── photo2.jpg
├── Person_001/        # Second identified person
│   └── photo3.jpg
└── Uncategorized/     # Faces that couldn't be grouped
    └── random.jpg
```

## Python API

```python
from main import FaceClusterer

# Initialize
clusterer = FaceClusterer(model_name="buffalo_l")

# Process images
clusterer.process_directory("/path/to/photos")

# Cluster faces
labels = clusterer.cluster(eps=0.5, min_samples=3)

# Organize into folders
clusterer.organize_photos(labels, "output_folder")

# Get statistics
stats = clusterer.get_statistics(labels)
print(f"Found {stats['total_clusters']} distinct people")
```

## Performance

| Operation | Speed (CPU) |
|-----------|-------------|
| Face detection + embedding | ~50-100ms per face |
| Clustering 1000 faces | < 1 second |

## Tech Stack

- **InsightFace** - Pre-trained models (RetinaFace + ArcFace)
- **ONNX Runtime** - CPU-optimized inference engine
- **scikit-learn** - DBSCAN clustering
- **OpenCV** - Image I/O

## License

MIT
