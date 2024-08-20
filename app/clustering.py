import os
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from PIL import Image
import logging
import numpy as np
from flask import jsonify
import json
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)

# Path to the clusters file
CLUSTER_FILE = 'clusters.json'

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Load clusters from file if it exists
def load_clusters_from_file():
    if os.path.exists(CLUSTER_FILE):
        with open(CLUSTER_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save clusters to file
def save_clusters_to_file():
    with open(CLUSTER_FILE, 'w') as f:
        json.dump(clusters, f, indent=4)

# Global clusters variable, initially loaded from file
clusters = load_clusters_from_file()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
    # Normalize the embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds


def add_cluster():
    # Calculate the new cluster ID based on the existing clusters
    if clusters:
        new_cluster_id = max(clusters.keys()) + 1
    else:
        new_cluster_id = 0

    # Add the new cluster
    clusters[new_cluster_id] = {'name': f'Cluster {new_cluster_id}', 'images': []}
    save_clusters_to_file()  # Save the state

    return jsonify({'status': 'success', 'cluster_id': new_cluster_id})



def delete_cluster(cluster_id, target_cluster_id=None):
    cluster_id = int(cluster_id)

    # Find the next available cluster if no target_cluster_id is provided
    if target_cluster_id is None:
        cluster_ids = sorted(clusters.keys())
        next_cluster_id = None
        for cid in cluster_ids:
            if cid != cluster_id:
                next_cluster_id = cid
                break
    else:
        next_cluster_id = int(target_cluster_id)

    if next_cluster_id is None:
        return jsonify({'status': 'error', 'message': 'No available cluster to transfer images'}), 400

    # Transfer images to the next cluster
    if cluster_id in clusters and next_cluster_id in clusters:
        clusters[next_cluster_id]['images'].extend(clusters[cluster_id]['images'])
        del clusters[cluster_id]
        save_clusters_to_file()  # Save the state after deletion and transfer
        return jsonify({'status': 'success', 'new_cluster_data': clusters[next_cluster_id]})
    else:
        return jsonify({'status': 'error', 'message': 'Cluster ID not found'}), 400



def rename_cluster(cluster_id, new_name):
    cluster_id = int(cluster_id)

    # Debugging output to ensure the cluster ID and name are correct
    print(f"Renaming Cluster ID: {cluster_id} to '{new_name}' (Type: {type(cluster_id)})")

    # Ensure the cluster exists and has the correct structure
    if cluster_id in clusters and isinstance(clusters[cluster_id], dict):
        print(f"Cluster {cluster_id} exists and is a dictionary. Proceeding to rename.")
        clusters[cluster_id]['name'] = new_name
        save_clusters_to_file()  # Save the state
        return jsonify({'status': 'success'})
    else:
        print(f"Error: clusters[{cluster_id}] is not a dictionary or doesn't contain 'name'.")
        return jsonify({'status': 'error', 'message': 'Cluster does not exist or is improperly structured'}), 500


def move_image(image, old_cluster_id, new_cluster_id):
    old_cluster_id = int(old_cluster_id)
    new_cluster_id = int(new_cluster_id)

    if old_cluster_id in clusters and new_cluster_id in clusters:
        if image in clusters[old_cluster_id]['images']:
            clusters[old_cluster_id]['images'].remove(image)
            clusters[new_cluster_id]['images'].append(image)
            save_clusters_to_file()

            return jsonify({
                'status': 'success',
                'old_cluster': clusters[old_cluster_id],
                'new_cluster': clusters[new_cluster_id]
            })
        else:
            return jsonify({'status': 'error', 'message': 'Image not found in the old cluster'}), 400
    else:
        return jsonify({'status': 'error', 'message': 'Cluster ID not found'}), 400


def perform_clustering(image_dir, num_clusters=5):
    embeddings = {}

    # Path to the static/images/ directory inside the app folder
    static_images_dir = os.path.join('app', 'static', 'images')

    # Ensure the static/images/ directory exists
    if not os.path.exists(static_images_dir):
        os.makedirs(static_images_dir)

    # Check for discrepancies between raw_images and static/images
    raw_images = set(os.listdir(image_dir))
    static_images = set(os.listdir(static_images_dir))

    # If there is a discrepancy, clear the static/images/ directory
    if raw_images != static_images:
        logging.info("Discrepancy found between raw_images and static_images. Clearing static/images/ directory.")
        for filename in os.listdir(static_images_dir):
            file_path = os.path.join(static_images_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Generate embeddings and copy images to the static/images/ directory
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            logging.info(f"Generating embedding for {filename}")
            embeddings[filename] = get_image_embedding(image_path)

            # Copy the image to static/images/ only if it doesn't already exist there
            static_image_path = os.path.join(static_images_dir, filename)
            if not os.path.exists(static_image_path):
                shutil.copy(image_path, static_image_path)
    
    logging.info(f"Generated embeddings for {len(embeddings)} images.")
    
    embedding_matrix = torch.cat(list(embeddings.values())).cpu().numpy()
    
    n_components = min(50, embedding_matrix.shape[0])

    if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] == 0:
        raise ValueError(f"Invalid embedding matrix shape: {embedding_matrix.shape}")

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embedding_matrix)
    
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
    cluster_labels = clustering_model.fit_predict(reduced_embeddings)
    logging.info(f"Clustering complete. Cluster labels: {cluster_labels}")

    clustered_images = {}
    for i in range(num_clusters):
        clustered_images[i] = {'name': f'Cluster {i}', 'images': []}

    for label, filename in zip(cluster_labels, embeddings.keys()):
        clustered_images[label]['images'].append(filename)

    global clusters
    clusters = clustered_images

    logging.info(f"Clustered images into {num_clusters} clusters.")
    return clusters


def get_clusters():
    """
    Return the current clusters.
    
    :return: Dictionary of clusters.
    """
    global clusters
    return clusters
