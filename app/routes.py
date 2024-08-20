from flask import Blueprint, render_template, request, jsonify
from .clustering import perform_clustering, get_clusters, rename_cluster, move_image, save_clusters_to_file, add_cluster, delete_cluster

bp = Blueprint('main', __name__)

clusters = {}  # This will store the clusters globally

@bp.route('/clusters')
def clusters_view():
    global clusters
    if not clusters: 
        clusters = perform_clustering('data/raw_images', num_clusters=5)
        save_clusters_to_file()
    return render_template('cluster_view.html', clusters=clusters)


@bp.route('/recluster', methods=['POST'])
def recluster():
    clusters = perform_clustering('data/raw_images', num_clusters=5)
    save_clusters_to_file()
    return jsonify({'status': 'success', 'clusters': clusters})

@bp.route('/add_cluster', methods=['POST'])
def add_cluster_route():
    return add_cluster()




@bp.route('/rename_cluster', methods=['POST'])
def rename_cluster_route():
    data = request.get_json()
    return rename_cluster(data['cluster_id'], data['new_name'])

@bp.route('/move_image', methods=['POST'])
def move_image_route():
    data = request.json
    return move_image(data['image'], data['old_cluster_id'], data['new_cluster_id'])

@bp.route('/delete_cluster', methods=['POST'])
def delete_cluster_route():
    data = request.json
    cluster_id = data['cluster_id']
    target_cluster_id = data.get('target_cluster_id', None)
    return delete_cluster(cluster_id, target_cluster_id)
