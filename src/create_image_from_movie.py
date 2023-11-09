import random
import os
from PIL import Image, ImageSequence


def split_and_save(input_gif, output_train, output_test, output_images, num_images=50):

    # GIFを読み込み
    gif = Image.open(input_gif)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]

    # 前半と後半に分割
    half = len(frames) // 2
    train_frames = frames[:half]
    test_frames = frames[half:]

    # 分割したフレームをGIFに保存
    train_gif = Image.new("RGB", gif.size)
    train_gif.save(output_train, save_all=True, append_images=train_frames)

    test_gif = Image.new("RGB", gif.size)
    test_gif.save(output_test, save_all=True, append_images=test_frames)

    # ランダムに画像を選択して保存
    selected_frames = random.sample(train_frames, k=min(num_images, len(train_frames)))
    for i, frame in enumerate(selected_frames):
        image_path = os.path.join(output_images, f"image_{i}.jpg")
        frame.save(image_path, "JPEG")


if __name__ == "__main__":
    input_gif = "yolo-movie-class/movie_resource/movie.gif"
    output_train = "yolo-movie-class/movie_resource/movie_train.gif"
    output_test = "yolo-movie-class/movie_resource/movie_test.gif"
    output_images = "yolo-movie-class/JPEGImages"

    split_and_save(input_gif, output_train, output_test, output_images, num_images=50)
