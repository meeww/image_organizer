# Image Organizer

Image Organizer is a work‑in‑progress tool that helps you explore and organize large sets of images by grouping them into meaningful clusters. It leverages OpenAI’s CLIP model to compute embeddings for your images, reduces the embedding dimensionality with PCA and then groups similar images together using agglomerative clustering. The result is presented through a simple web interface where you can view clusters, rename them, add or delete clusters, and drag images between clusters.

## Features

- **AI‑powered image clustering:** uses the `openai/clip‑vit‑base‑patch32` model via the HuggingFace transformers library to generate semantic embeddings for your images and groups them with agglomerative clustering.
- **Interactive web UI:** clusters are displayed in an interactive interface built with Flask, jQuery and jQuery UI. Each cluster panel displays its images and provides controls to add, rename or delete clusters.
- **Drag‑and‑drop reorganisation:** you can move images between clusters simply by dragging and dropping thumbnails, add new empty clusters, or delete clusters (with automatic transfer of their images).
- **Persistent state:** clusters are saved to `clusters.json` so you can continue working where you left off.
- **Minimal dependencies:** runs as a local Flask application with Python. All dependencies are listed in `requirements.txt`.

## Installation

1. Clone this repository and change into it:

   ```bash
   git clone https://github.com/meeww/image_organizer.git
   cd image_organizer
   ```

2. (Optional but recommended) Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the images you want to organise into the `data/raw_images` directory. Any `.jpg` or `.jpeg` files in this folder will be embedded and clustered.

## Running the app

Start the Flask development server with:

```bash
python run.py
```

Then open your browser to `http://localhost:5000/clusters`. The application will embed your images, perform PCA and agglomerative clustering (defaults to 5 clusters) and display an initial grouping.

### Using the UI

- To re‑run clustering from scratch, click the *Recluster* button.
- To create a new empty cluster, click the plus icon.
- To rename a cluster, click on its title and type a new name.
- To delete a cluster, click the trash icon; any images in the deleted cluster will be moved into the next cluster.
- Drag and drop images between clusters to manually refine the grouping.
- When you add, move or delete clusters/images, changes are saved to `clusters.json` automatically.

## Configuration

- The default number of clusters is set in `app/routes.py` (`num_clusters=5`). Adjust this argument when calling `perform_clustering()` to suit your dataset.
- If you need to change Flask settings (e.g. secret key), edit `config.py`.

## Roadmap

This project is a work in progress. Potential improvements include:
- Supporting other embedding models or feature extractors.
- Allowing dynamic adjustment of the number of clusters from the UI.
- Persisting user edits to the image order across sessions.
- Packaging the app into a standalone desktop tool.

## Contributing

Pull requests and issues are welcome! Feel free to fork the repository, improve the clustering logic or UI and open a pull request.
