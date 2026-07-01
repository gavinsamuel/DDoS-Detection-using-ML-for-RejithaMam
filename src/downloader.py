import os
import urllib.request

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded {destination}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise e

def main():
    # Setup directories
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # File mappings
    files = {
        "KDDTrain+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
        "KDDTest+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"
    }

    for filename, url in files.items():
        dest_path = os.path.join(data_dir, filename)
        if os.path.exists(dest_path):
            print(f"File {filename} already exists at {dest_path}. Skipping download.")
        else:
            download_file(url, dest_path)

if __name__ == "__main__":
    main()
