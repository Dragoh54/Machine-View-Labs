from PIL import Image
import numpy as np
import math
from collections import deque

# Параметры изображения
img_width = 320
img_height = 320
src = "./img/2.jpg"

# Загрузка изображения
img = Image.open(src).resize((img_width, img_height))
img_data = np.array(img)


def get_image_data(image):
    # Возвращаем данные изображения как одномерный массив
    return image.flatten()


def split_on_pixels(image_data):
    # Разделение данных изображения на пиксели
    pixels = []
    for i in range(0, len(image_data), 4):
        pixels.append(
            {
                "r": image_data[i],
                "g": image_data[i + 1],
                "b": image_data[i + 2],
                "a": image_data[i + 3],
            }
        )
    return pixels


def get_region_growing_segments(image_data):
    visited_pixels = {}
    observing_queue = deque()
    image_data = list(image_data)
    pixels_amount = len(image_data) // 4
    width = img_width * 4
    threshold = 110  # Порог для роста области
    possible_pixel_positions = list(range(len(image_data)))

    def is_pixel_not_checked(index):
        return (
            index % 4 == 0
            and index > 0
            and index <= len(image_data) - width - 4
            and index not in visited_pixels
        )

    def find_available_pixel():
        return next(
            (
                index
                for index in possible_pixel_positions
                if is_pixel_not_checked(index)
            ),
            None,
        )

    def get_neighbour_pixel_to_check(cur_index):
        neighbours = [
            cur_index + 4,
            cur_index + width,
            cur_index - width,
            cur_index - 4,
        ]
        return next(
            (index for index in neighbours if is_pixel_not_checked(index)), None
        )

    def distance(r1, r2, g1, g2, b1, b2):
        return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)

    def run(current_pixel):
        visited_pixels[current_pixel] = True
        observing_queue.append(current_pixel)

        while observing_queue:
            current_pixel = observing_queue[0]
            neighbour_pixel = get_neighbour_pixel_to_check(current_pixel)
            if neighbour_pixel is not None:
                r1, g1, b1 = (
                    image_data[current_pixel],
                    image_data[current_pixel + 1],
                    image_data[current_pixel + 2],
                )
                r2, g2, b2 = (
                    image_data[neighbour_pixel],
                    image_data[neighbour_pixel + 1],
                    image_data[neighbour_pixel + 2],
                )
                if distance(r1, r2, g1, g2, b1, b2) < threshold:
                    image_data[neighbour_pixel] = 255 - image_data[neighbour_pixel]
                    image_data[neighbour_pixel + 1] = (
                        255 - image_data[neighbour_pixel + 1]
                    )
                    image_data[neighbour_pixel + 2] = (
                        255 - image_data[neighbour_pixel + 2]
                    )
                    observing_queue.append(neighbour_pixel)
                visited_pixels[neighbour_pixel] = True
            else:
                observing_queue.popleft()

    while pixels_amount - len(visited_pixels) > 0.05 * pixels_amount:
        available_pixel = find_available_pixel()
        if available_pixel is not None:
            run(available_pixel)

    return image_data


def get_split_and_merge_segments(image_data):
    # Реализация разделения и слияния
    def should_area_be_splitted(pixels):
        threshold = 60
        gray_pixels = [get_gray_color(pix) for pix in pixels]
        return max(gray_pixels) - min(gray_pixels) >= threshold

    def get_gray_color(pixel):
        return 0.2125 * pixel["r"] + 0.7154 * pixel["g"] + 0.0721 * pixel["b"]

    def split_on_4_areas(area):
        rows_and_cols = int(math.sqrt(len(area["pixels"])) / 2)
        area1_pixels = area["pixels"][: rows_and_cols**2]
        area2_pixels = area["pixels"][rows_and_cols**2 : rows_and_cols**2 * 2]
        area3_pixels = area["pixels"][rows_and_cols**2 * 2 : rows_and_cols**2 * 3]
        area4_pixels = area["pixels"][rows_and_cols**2 * 3 :]
        return [
            {"pixels": area1_pixels},
            {"pixels": area2_pixels},
            {"pixels": area3_pixels},
            {"pixels": area4_pixels},
        ]

    def split(areas):
        splitted = []
        for area in areas:
            if should_area_be_splitted(area["pixels"]):
                splitted += split_on_4_areas(area)
            else:
                splitted.append(area)
        return splitted

    return split([{"pixels": image_data}])


def draw_on_image(image_data):
    # Возвращаем изображение после сегментации
    img_data = np.array(image_data, dtype=np.uint8).reshape((img_height, img_width, 4))
    return Image.fromarray(img_data)


# Процесс сегментации
image_data = get_image_data(img)
split_and_merge_segments = get_split_and_merge_segments(split_on_pixels(image_data))
region_growing_segments = get_region_growing_segments(image_data)

# Конвертируем обратно в изображение
region_growing_image = draw_on_image(region_growing_segments)
split_and_merge_image = draw_on_image(split_and_merge_segments)

# Сохраняем результат
region_growing_image.save("./res/region_growing_segmented.png")
split_and_merge_image.save("./res/split_and_merge_segmented.png")
