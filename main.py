"""
IsoFace: Face Clustering using ArcFace Embeddings and DBSCAN

CPU-optimized face recognition and clustering using ONNX Runtime.
Uses RetinaFace for detection and ArcFace for embedding extraction.
"""

import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


class FaceClusterer:
    """
    Face clustering pipeline using InsightFace (ArcFace + RetinaFace).
    Optimized for CPU inference using ONNX Runtime.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: tuple = (640, 640)):
        """
        Initialize the face analysis model.

        Args:
            model_name: InsightFace model pack ('buffalo_l' for highest accuracy)
            det_size: Detection size (higher = more accurate but slower)
        """
        print(f"Loading InsightFace model '{model_name}'...")
        print("Using ONNX Runtime for CPU-optimized inference")

        # Initialize with CPU execution provider for ONNX optimization
        self.app = FaceAnalysis(
            name=model_name, providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

        # Storage for embeddings and metadata
        self.embeddings: list[np.ndarray] = []
        self.image_paths: list[str] = []
        self.face_locations: list[np.ndarray] = []

        print("Model loaded successfully!\n")

    def clear(self):
        """Clear all stored embeddings and metadata."""
        self.embeddings.clear()
        self.image_paths.clear()
        self.face_locations.clear()

    def process_image(self, image_path: str) -> int:
        """
        Process a single image and extract face embeddings.

        Args:
            image_path: Path to the image file

        Returns:
            Number of faces detected in the image
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Warning: Could not read image: {image_path}")
            return 0

        # Detect faces and extract embeddings
        faces = self.app.get(img)

        for face in faces:
            self.embeddings.append(face.embedding)
            self.image_paths.append(image_path)
            self.face_locations.append(face.bbox)

        return len(faces)

    def process_directory(
        self, directory: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")
    ) -> tuple[int, int]:
        """
        Process all images in a directory recursively.

        Args:
            directory: Path to the directory containing images
            extensions: Tuple of valid image file extensions

        Returns:
            Tuple of (total_images_processed, total_faces_found)
        """
        directory = Path(directory)
        image_files = []

        # Collect all image files recursively
        for ext in extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))
        total_images = len(image_files)
        total_faces = 0

        print(f"Found {total_images} images to process\n")

        for i, img_path in enumerate(image_files, 1):
            if i % 50 == 0 or i == total_images:
                print(f"Processing: {i}/{total_images} images...")

            faces_found = self.process_image(str(img_path))
            total_faces += faces_found

        print(f"\nProcessed {total_images} images, found {total_faces} faces\n")
        return total_images, total_faces

    def cluster(self, eps: float = 0.5, min_samples: int = 3) -> np.ndarray:
        """
        Cluster face embeddings using DBSCAN.

        Args:
            eps: Maximum distance between samples (lower = stricter matching)
            min_samples: Minimum samples to form a cluster

        Returns:
            Array of cluster labels (-1 = uncategorized/noise)
        """
        if len(self.embeddings) == 0:
            print("No embeddings to cluster!")
            return np.array([])

        embeddings_array = np.array(self.embeddings)
        print(f"Clustering {len(embeddings_array)} face embeddings...")
        print(f"Parameters: eps={eps}, min_samples={min_samples}, metric=cosine\n")

        # DBSCAN with cosine distance (optimal for Siamese-trained embeddings)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clusterer.fit_predict(embeddings_array)

        # Statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        print("Clustering Results:")
        print(f"  - Identified {n_clusters} distinct people")
        print(f"  - {n_noise} faces marked as uncategorized (noise)")
        print()

        return labels

    def organize_photos(
        self, labels: np.ndarray, output_dir: str, copy_files: bool = True
    ) -> dict:
        """
        Organize photos into folders based on cluster labels.

        Args:
            labels: Cluster labels from DBSCAN
            output_dir: Output directory for organized photos
            copy_files: If True, copy files; if False, just return the mapping

        Returns:
            Dictionary mapping cluster labels to list of image paths
        """
        output_dir = Path(output_dir)
        clusters = defaultdict(list)

        # Group images by cluster
        for label, img_path in zip(labels, self.image_paths):
            clusters[label].append(img_path)

        if copy_files:
            print(f"Organizing photos into: {output_dir}\n")

            for label, paths in sorted(clusters.items()):
                if label == -1:
                    folder_name = "Uncategorized"
                else:
                    folder_name = f"Person_{label:03d}"

                folder_path = output_dir / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)

                # Copy unique images (same image might have multiple faces)
                unique_paths = set(paths)
                for img_path in unique_paths:
                    dest = folder_path / Path(img_path).name
                    # Handle duplicate filenames
                    if dest.exists():
                        stem = dest.stem
                        suffix = dest.suffix
                        counter = 1
                        while dest.exists():
                            dest = folder_path / f"{stem}_{counter}{suffix}"
                            counter += 1
                    shutil.copy2(img_path, dest)

                print(f"  {folder_name}: {len(unique_paths)} photos")

        return dict(clusters)

    def get_statistics(self, labels: np.ndarray) -> dict:
        """Get clustering statistics."""
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        stats = {
            "total_faces": len(labels),
            "total_clusters": n_clusters,
            "uncategorized": list(labels).count(-1),
            "cluster_sizes": {},
        }

        for label in unique_labels:
            if label != -1:
                stats["cluster_sizes"][f"Person_{label}"] = list(labels).count(label)

        return stats

    def auto_tune(
        self,
        eps_values: list[float] | None = None,
        min_samples: int = 2,
    ) -> tuple[float, list[dict]]:
        """
        Auto-tune the eps parameter by testing multiple values.
        
        Helps find the optimal eps by showing how many groups are found
        vs how many faces are left uncategorized for each value.

        Args:
            eps_values: List of eps values to test (default: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
            min_samples: Minimum samples to form a cluster (kept at 2 to catch small groups)

        Returns:
            Tuple of (recommended_eps, list of results for each eps)
        """
        if len(self.embeddings) == 0:
            print("No embeddings to tune! Process images first.")
            return 0.5, []

        if eps_values is None:
            eps_values = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

        embeddings_array = np.array(self.embeddings)
        results = []

        print("\n" + "=" * 60)
        print("AUTO-TUNER: Finding Optimal EPS Value")
        print("=" * 60)
        print(f"\nTesting {len(eps_values)} different eps values...")
        print(f"Total faces to cluster: {len(embeddings_array)}")
        print(f"min_samples fixed at: {min_samples}\n")

        print(f"{'EPS':<10} | {'Found Groups':<15} | {'Uncategorized':<15} | {'Recommendation'}")
        print("-" * 70)

        best_eps = eps_values[0]
        best_score = -1

        for eps in eps_values:
            clt = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            clt.fit(embeddings_array)

            # Count groups (excluding -1 which is noise)
            num_groups = len(set(clt.labels_)) - (1 if -1 in clt.labels_ else 0)
            num_noise = list(clt.labels_).count(-1)

            # Score: balance between groups found and noise reduced
            # Higher groups + lower noise = better
            noise_ratio = num_noise / len(embeddings_array) if len(embeddings_array) > 0 else 1
            score = num_groups * (1 - noise_ratio)

            # Determine recommendation
            if noise_ratio > 0.7:
                recommendation = "Too strict"
            elif noise_ratio < 0.1 and num_groups < 3:
                recommendation = "Too loose (merged)"
            elif 0.1 <= noise_ratio <= 0.4:
                recommendation = "★ Good balance"
            else:
                recommendation = "Acceptable"

            if score > best_score:
                best_score = score
                best_eps = eps

            result = {
                "eps": eps,
                "num_groups": num_groups,
                "num_noise": num_noise,
                "noise_ratio": noise_ratio,
                "score": score,
                "recommendation": recommendation,
            }
            results.append(result)

            print(f"{eps:<10} | {num_groups:<15} | {num_noise:<15} | {recommendation}")

        print("-" * 70)
        print(f"\n★ Recommended EPS: {best_eps}")
        print("  (Highest groups with lowest uncategorized ratio)\n")

        return best_eps, results


def download_lfw_dataset() -> str:
    """
    Download the LFW dataset from Kaggle using kagglehub.

    Returns:
        Path to the downloaded dataset
    """
    print("Downloading LFW dataset from Kaggle...")
    print("(This may take a while on first run)\n")

    import kagglehub

    # Download the dataset
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
    print(f"Dataset downloaded to: {path}\n")

    return path


def find_lfw_images_dir(base_path: str) -> str:
    """
    Find the actual images directory within the LFW dataset.

    Args:
        base_path: Base path of the downloaded dataset

    Returns:
        Path to the directory containing face images
    """
    base_path = Path(base_path)

    # Common locations for LFW images
    possible_paths = [
        base_path / "lfw-deepfunneled" / "lfw-deepfunneled",
        base_path / "lfw-deepfunneled",
        base_path / "lfw",
        base_path,
    ]

    for path in possible_paths:
        if path.exists():
            # Check if this directory has subdirectories with images
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if subdirs:
                # Check if subdirs contain images
                for subdir in subdirs[:3]:  # Check first 3
                    images = list(subdir.glob("*.jpg"))
                    if images:
                        print(f"Found LFW images at: {path}")
                        return str(path)

    # Fallback to base path
    print(f"Using base path: {base_path}")
    return str(base_path)


def run_demo(
    dataset_path: str | None = None,
    output_dir: str = "clustered_faces",
    eps: float = 0.5,
    min_samples: int = 3,
    max_people: int | None = 10,
    auto_tune: bool = False,
):
    """
    Run the face clustering demo.

    Args:
        dataset_path: Path to image directory (downloads LFW if None)
        output_dir: Output directory for clustered photos
        eps: DBSCAN eps parameter (distance threshold)
        min_samples: Minimum samples to form a cluster
        max_people: Limit processing to first N people (for demo speed)
        auto_tune: If True, run auto-tuner to find optimal eps
    """
    print("=" * 60)
    print("IsoFace: Face Clustering Demo")
    print("=" * 60 + "\n")

    # Get dataset
    if dataset_path is None:
        dataset_path = download_lfw_dataset()
        dataset_path = find_lfw_images_dir(dataset_path)

    # For demo purposes, limit to subset of people
    if max_people is not None:
        dataset_path = Path(dataset_path)
        subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])[:max_people]
        if subdirs:
            # Create a temporary view with limited people
            print(f"Demo mode: Processing first {len(subdirs)} people from dataset\n")
            demo_paths = []
            for subdir in subdirs:
                demo_paths.extend(subdir.glob("*.jpg"))
        else:
            demo_paths = list(dataset_path.rglob("*.jpg"))[:100]

    # Initialize clusterer
    clusterer = FaceClusterer(model_name="buffalo_l")

    # Process images
    print("-" * 40)
    print("Phase 1: Face Detection & Embedding")
    print("-" * 40)

    if max_people is not None and 'demo_paths' in locals():
        for i, img_path in enumerate(demo_paths, 1):
            if i % 20 == 0 or i == len(demo_paths):
                print(f"Processing: {i}/{len(demo_paths)} images...")
            clusterer.process_image(str(img_path))
        print(f"\nProcessed {len(demo_paths)} images, found {len(clusterer.embeddings)} faces\n")
    else:
        clusterer.process_directory(str(dataset_path))

    if len(clusterer.embeddings) == 0:
        print("No faces found in the dataset!")
        return

    # Auto-tune if requested
    if auto_tune:
        print("-" * 40)
        print("Phase 2: Auto-Tuning EPS Parameter")
        print("-" * 40)
        recommended_eps, tune_results = clusterer.auto_tune(min_samples=min_samples)
        eps = recommended_eps
        print(f"Using auto-tuned eps={eps}\n")

    # Cluster faces
    print("-" * 40)
    print("Phase 3: Clustering with DBSCAN" if auto_tune else "Phase 2: Clustering with DBSCAN")
    print("-" * 40)

    labels = clusterer.cluster(eps=eps, min_samples=min_samples)

    # Organize into folders
    print("-" * 40)
    print("Phase 4: Organizing Photos" if auto_tune else "Phase 3: Organizing Photos")
    print("-" * 40)

    clusters = clusterer.organize_photos(labels, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    stats = clusterer.get_statistics(labels)
    print(f"Total faces processed: {stats['total_faces']}")
    print(f"Distinct people identified: {stats['total_clusters']}")
    print(f"Uncategorized faces: {stats['uncategorized']}")
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 60)

    return clusterer, labels, clusters


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Face clustering using ArcFace embeddings and DBSCAN"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input directory with images (downloads LFW dataset if not specified)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="clustered_faces",
        help="Output directory for clustered photos",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps (distance threshold). Lower=stricter, Higher=looser",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples to form a cluster",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=10,
        help="Limit to first N people for demo (None for all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all images (ignore --max-people)",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Auto-tune eps parameter by testing multiple values",
    )

    args = parser.parse_args()

    max_people = None if args.all else args.max_people

    run_demo(
        dataset_path=args.input,
        output_dir=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        max_people=max_people,
        auto_tune=args.auto_tune,
    )


if __name__ == "__main__":
    main()
