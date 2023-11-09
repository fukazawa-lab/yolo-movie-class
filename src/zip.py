import zipfile
import os

def zip_folder_with_subfolder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, os.path.join("JPEGImages", arcname))

if __name__ == "__main__":
    folder_to_zip = "yolo-movie-class/JPEGImages"  # ZIPにするフォルダのパス
    zip_file_path = "yolo-movie-class/JPEGImages.zip"  # 保存するZIPファイルのパス

    zip_folder_with_subfolder(folder_to_zip, zip_file_path)
    print(f"{zip_file_path} が作成されました。yolo-movie-class/JPEGImages.zipをダウンロードしてローカルに保存してください。")