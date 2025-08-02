import os
import requests

def download_file(url, dest_folder="downloads"):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = url.split("/")[-1].split("?")[0]
    local_path = os.path.join(dest_folder, local_filename)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

def is_docx(file_path):
    return file_path.lower().endswith('.docx')
