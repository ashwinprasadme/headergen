
import requests
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))



def download_models_from_github_release(repo_owner="secure-software-engineering", 
                                       repo_name="HeaderGen", 
                                       release_tag="models", 
                                       asset_names=["rf_code_scaled.pkl", "scaler.pkl",  "selector.pkl",  "tfidf_vectorizer.pkl"], 
                                       download_path=f"{SCRIPT_DIR}/models/"):
    """Downloads specific files from a GitHub release.
    Args:
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        release_tag (str): The tag name of the release (e.g., 'v1.0.0').
        asset_names (list): The names of the asset files to download.
        download_path (str): The local path where the files should be saved.
    """

    # first check if the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    # check if files already exist and remove from the list
    assets_to_download = []
    for asset_name in asset_names:
        if not os.path.exists(os.path.join(download_path, asset_name)):
            assets_to_download.append(asset_name)

    if not assets_to_download:
        return "Models already exist" 

    # API endpoint to get release info
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{release_tag}"

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad response status codes

    release_data = response.json()

    print("Assets in this release:")  # Add this line
    for asset in release_data['assets']:
        print(asset['name'])  # Add this line

    for asset_name in assets_to_download:
        # Find the download URL of the asset
        asset_url = None
        for asset in release_data['assets']:
            if asset['name'] == asset_name:
                asset_url = asset['browser_download_url']
                break

        if not asset_url:
            raise ValueError(f"Asset '{asset_name}' not found in the release.")

        # Download the file
        print(f"Downloading model file: {asset['name']}")
        response = requests.get(asset_url, stream=True)
        response.raise_for_status()

        file_path = os.path.join(download_path, asset_name)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: 
                    f.write(chunk)

        print(f"File downloaded to: {file_path}")
