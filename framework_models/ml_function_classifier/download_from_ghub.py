import requests
import os

def download_file_from_github_release(repo_owner="suvanshchawla", 
                                      repo_name="cell-classifer", 
                                      release_tag="rel-cell-classifier-ver-0.0.1", 
                                      asset_name="code_classifier.pkl", 
                                      download_path="./cell_classifier/"):
    """Downloads a specific file from a GitHub release.

    Args:
        repo_owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        release_tag (str): The tag name of the release (e.g., 'v1.0.0').
        asset_name (str): The name of the asset file to download.
        download_path (str): The local path where the file should be saved.
    """

    # API endpoint to get release info
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{release_tag}"

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad response status codes

    release_data = response.json()
    
    print("Assets in this release:")  # Add this line
    for asset in release_data['assets']:
        print(asset['name'])  # Add this line


    # Find the download URL of the asset
    asset_url = None
    for asset in release_data['assets']:
        if asset['name'] == asset_name:
            asset_url = asset['browser_download_url']
            break

    if not asset_url:
        raise ValueError(f"Asset '{asset_name}' not found in the release.")

    # Download the file
    response = requests.get(asset_url, stream=True)
    response.raise_for_status()

    file_path = os.path.join(download_path, asset_name)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk: 
                f.write(chunk)

    print(f"File downloaded to: {file_path}")

