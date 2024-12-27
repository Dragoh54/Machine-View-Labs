import numpy as np
from PIL import Image

# Параметры изображения
IMG_PATH = "./img/2.jpg"
IMG_WIDTH, IMG_HEIGHT = 320, 320
THRESHOLD_REGION_GROWING = 110
THRESHOLD_SPLIT_AND_MERGE = 60


def load_image(path):
    img = Image.open(path).resize((IMG_WIDTH, IMG_HEIGHT))
    return np.array(img)


def save_image(data, path):
    # Нормализация данных в диапазон 0-255
    data = np.clip(data, 0, 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(path)


def distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))


def region_growing_segmentation(image):
    visited = np.zeros(image.shape[:2], dtype=bool)
    result = image.copy()

    def grow_region(seed):
        queue = [seed]
        while queue:
            x, y = queue.pop(0)
            visited[x, y] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < image.shape[0]
                    and 0 <= ny < image.shape[1]
                    and not visited[nx, ny]
                ):
                    if distance(image[x, y], image[nx, ny]) < THRESHOLD_REGION_GROWING:
                        # result[nx, ny] = 255 - result[nx, ny]
                        queue.append((nx, ny))
                        visited[nx, ny] = True

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if not visited[x, y]:
                grow_region((x, y))

    return result


def should_split_area(pixels):
    gray_pixels = (
        0.2125 * pixels[..., 0] + 0.7154 * pixels[..., 1] + 0.0721 * pixels[..., 2]
    )
    return np.max(gray_pixels) - np.min(gray_pixels) >= THRESHOLD_SPLIT_AND_MERGE


def split_and_merge_segmentation(image):
    h, w = image.shape[:2]
    areas = []

    def split(area, x, y):
        if area.size == 0:
            return []

        # Если область нужно разделить
        if should_split_area(area):
            h, w = area.shape[:2]
            h2, w2 = h // 2, w // 2
            return (
                split(area[:h2, :w2], x, y)  # Левая верхняя область
                + split(area[:h2, w2:], x, y + w2)  # Правая верхняя область
                + split(area[h2:, :w2], x + h2, y)  # Левая нижняя область
                + split(area[h2:, w2:], x + h2, y + w2)  # Правая нижняя область
            )
        else:
            # Если область не нужно разделять, сохраняем её координаты
            areas.append((x, y, area))
            return [(x, y, area)]

    split(image, 0, 0)
    return areas


def save_segmented_image(image, areas, path):
    """
    Сохраняет изображение с объединенными областями.

    :param image: Исходное изображение (numpy массив)
    :param areas: Список областей [(x, y, область)]
    :param path: Путь для сохранения результата
    """
    # Итоговое изображение такого же размера, как оригинал
    result = np.zeros_like(image)

    for x, y, area in areas:
        h, w = area.shape[:2]
        mean_color = np.mean(area, axis=(0, 1)).astype(np.uint8)  # Средний цвет области
        result[x : x + h, y : y + w] = mean_color  # Заполняем область средним цветом

    # Сохраняем итоговое изображение
    save_image(result, path)


def main():
    image = load_image(IMG_PATH)
    print(f"Loaded image size: {image.shape}")

    print("Processing Region Growing Segmentation...")
    region_growing_result = region_growing_segmentation(image)
    save_image(region_growing_result, "./res/region_growing_result.jpg")
    print("Saved Region Growing Segmentation...")

    print("Processing Split and Merge Segmentation...")
    areas = split_and_merge_segmentation(image)
    save_segmented_image(image, areas, "./res/split_and_merge_result.jpg")
    print("Saved Split and Merge Segmentation...")


if __name__ == "__main__":
    main()
