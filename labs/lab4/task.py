from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Добавляем tqdm для отображения прогресса


def plot_histogram(image):
    r, g, b = image.split()
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(r).flatten(), bins=256, color="red", alpha=0.5, label="Red")
    plt.hist(np.array(g).flatten(), bins=256, color="green", alpha=0.5, label="Green")
    plt.hist(np.array(b).flatten(), bins=256, color="blue", alpha=0.5, label="Blue")
    plt.title("Color Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def to_grayscale(image):
    grayscale_image = image.convert("L")
    grayscale_image.save("./img/grayscale.jpg")
    return grayscale_image


def binarize_image(image, threshold=128):
    binarized_image = image.point(lambda p: 255 if p > threshold else 0)
    binarized_image.save("./img/binarized.jpg")
    return binarized_image


def remove_noise_binary(image):
    binary_array = np.array(image)
    height, width = binary_array.shape

    filtered_array = np.copy(binary_array)
    for x in tqdm(range(1, height - 1), desc="Removing binary noise", unit="row"):
        for y in range(1, width - 1):
            neighbors = binary_array[x - 1 : x + 2, y - 1 : y + 2]
            if np.sum(neighbors) < 4 * 255:
                filtered_array[x, y] = 0
            else:
                filtered_array[x, y] = 255

    filtered_image = Image.fromarray(filtered_array)
    filtered_image.save("./img/binary_noise_removed.jpg")
    return filtered_image


def average_filter(image, kernel_size=3):
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape
    filtered_array = np.copy(img_array)

    pad = kernel_size // 2
    padded_array = np.pad(img_array, pad, mode="edge")

    for x in tqdm(range(height), desc="Applying average filter", unit="row"):
        for y in range(width):
            neighbors = padded_array[x : x + kernel_size, y : y + kernel_size]
            filtered_array[x, y] = np.mean(neighbors)

    filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
    filtered_image.save("./img/average_filtered.jpg")
    return filtered_image


def median_filter(image, kernel_size=3):
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape
    filtered_array = np.copy(img_array)

    pad = kernel_size // 2
    padded_array = np.pad(img_array, pad, mode="edge")

    for x in tqdm(range(height), desc="Applying median filter", unit="row"):  # Прогресс
        for y in range(width):
            neighbors = padded_array[x : x + kernel_size, y : y + kernel_size]
            filtered_array[x, y] = np.median(neighbors)

    filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
    filtered_image.save("./img/median_filtered.jpg")
    return filtered_image


def edge_detection_binary(image):
    binary_array = np.array(image)
    height, width = binary_array.shape

    edges_array = np.zeros_like(binary_array)
    for x in tqdm(
        range(1, height - 1), desc="Edge detection on binary image", unit="row"
    ):  # Прогресс
        for y in range(1, width - 1):
            gx = abs(int(binary_array[x + 1, y]) - int(binary_array[x - 1, y]))
            gy = abs(int(binary_array[x, y + 1]) - int(binary_array[x, y - 1]))
            edges_array[x, y] = 255 if gx + gy > 0 else 0

    edges_image = Image.fromarray(edges_array)
    edges_image.save("./img/binary_edges.jpg")
    return edges_image


def edge_detection_grayscale(image):
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape

    edges_array = np.zeros_like(img_array)
    for x in tqdm(
        range(1, height - 1), desc="Edge detection on grayscale image", unit="row"
    ):  # Прогресс
        for y in range(1, width - 1):
            gx = abs(img_array[x + 1, y] - img_array[x - 1, y])
            gy = abs(img_array[x, y + 1] - img_array[x, y - 1])
            edges_array[x, y] = min(255, gx + gy)

    edges_image = Image.fromarray(edges_array.astype(np.uint8))
    edges_image.save("./img/grayscale_edges.jpg")
    return edges_image


if __name__ == "__main__":
    # Укажите путь к вашему изображению
    img_path = "./img/image.png"
    img = Image.open(img_path).convert("RGB")

    print("1. Построение гистограммы изображения...")
    plot_histogram(img)

    print("2. Преобразование в полутоновое изображение...")
    grayscale_img = to_grayscale(img)

    print("3. Бинаризация изображения...")
    binarized_img = binarize_image(grayscale_img)

    print("4. Устранение шумов на бинарном изображении...")
    binary_denoised = remove_noise_binary(binarized_img)

    print("5. Усредняющий фильтр на полутоновом изображении...")
    average_filtered_img = average_filter(grayscale_img)

    print("6. Медианный фильтр на полутоновом изображении...")
    median_filtered_img = median_filter(grayscale_img)

    print("7. Выделение границ на бинарном изображении...")
    binary_edges = edge_detection_binary(binary_denoised)

    print("8. Выделение границ на полутоновом изображении...")
    grayscale_edges = edge_detection_grayscale(grayscale_img)

    print("Все операции завершены!")


# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm


# # Построение гистограммы изображения
# def plot_histogram(image):
#     r, g, b = image.split()
#     plt.figure(figsize=(10, 6))
#     plt.hist(np.array(r).flatten(), bins=256, color="red", alpha=0.5, label="Red")
#     plt.hist(np.array(g).flatten(), bins=256, color="green", alpha=0.5, label="Green")
#     plt.hist(np.array(b).flatten(), bins=256, color="blue", alpha=0.5, label="Blue")
#     plt.title("Color Histogram")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()


# # Преобразование цветного изображения в полутоновое
# def to_grayscale(image):
#     grayscale_image = image.convert("L")  # Конвертация в оттенки серого
#     grayscale_image.show()
#     grayscale_image.save("./img/grayscale.jpg")
#     return grayscale_image


# # Бинаризация полутоновых изображений
# def binarize_image(image, threshold=128):
#     binarized_image = image.point(lambda p: 255 if p > threshold else 0)
#     binarized_image.show()
#     binarized_image.save("./img/binarized.jpg")
#     return binarized_image


# # Устранение шумов на бинарном изображении
# def remove_noise_binary(image):
#     binary_array = np.array(image)
#     height, width = binary_array.shape

#     # Усреднение по соседям
#     filtered_array = np.copy(binary_array)
#     for x in range(1, height - 1):
#         for y in range(1, width - 1):
#             neighbors = binary_array[x - 1 : x + 2, y - 1 : y + 2]
#             if (
#                 np.sum(neighbors) < 4 * 255
#             ):  # Если большинство соседей черные, делаем пиксель черным
#                 filtered_array[x, y] = 0
#             else:
#                 filtered_array[x, y] = 255

#     filtered_image = Image.fromarray(filtered_array)
#     filtered_image.show()
#     filtered_image.save("./img/binary_noise_removed.jpg")
#     return filtered_image


# # Устранение шумов на полутоновом изображении
# # Усредняющий фильтр:
# def average_filter(image, kernel_size=3):
#     img_array = np.array(image, dtype=np.float32)
#     height, width = img_array.shape
#     filtered_array = np.copy(img_array)

#     pad = kernel_size // 2
#     padded_array = np.pad(img_array, pad, mode="edge")

#     for x in range(height):
#         for y in range(width):
#             neighbors = padded_array[x : x + kernel_size, y : y + kernel_size]
#             filtered_array[x, y] = np.mean(neighbors)

#     filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
#     filtered_image.show()
#     filtered_image.save("./img/average_filtered.jpg")
#     return filtered_image


# # Медианный фильтр:
# def median_filter(image, kernel_size=3):
#     img_array = np.array(image, dtype=np.float32)
#     height, width = img_array.shape
#     filtered_array = np.copy(img_array)

#     pad = kernel_size // 2
#     padded_array = np.pad(img_array, pad, mode="edge")

#     for x in range(height):
#         for y in range(width):
#             neighbors = padded_array[x : x + kernel_size, y : y + kernel_size]
#             filtered_array[x, y] = np.median(neighbors)

#     filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
#     filtered_image.show()
#     filtered_image.save("./img/median_filtered.jpg")
#     return filtered_image


# # Выделение границ объектов на бинарном изображении
# def edge_detection_binary(image):
#     binary_array = np.array(image)
#     height, width = binary_array.shape

#     edges_array = np.zeros_like(binary_array)
#     for x in range(1, height - 1):
#         for y in range(1, width - 1):
#             gx = abs(int(binary_array[x + 1, y]) - int(binary_array[x - 1, y]))
#             gy = abs(int(binary_array[x, y + 1]) - int(binary_array[x, y - 1]))
#             edges_array[x, y] = 255 if gx + gy > 0 else 0

#     edges_image = Image.fromarray(edges_array)
#     edges_image.show()
#     edges_image.save("./img/binary_edges.jpg")
#     return edges_image


# # Выделение границ объектов на полутоновом изображении
# def edge_detection_grayscale(image):
#     img_array = np.array(image, dtype=np.float32)
#     height, width = img_array.shape

#     edges_array = np.zeros_like(img_array)
#     for x in range(1, height - 1):
#         for y in range(1, width - 1):
#             gx = abs(img_array[x + 1, y] - img_array[x - 1, y])
#             gy = abs(img_array[x, y + 1] - img_array[x, y - 1])
#             edges_array[x, y] = min(255, gx + gy)

#     edges_image = Image.fromarray(edges_array.astype(np.uint8))
#     edges_image.show()
#     edges_image.save("./img/grayscale_edges.jpg")
#     return edges_image


# if __name__ == "__main__":
#     # Укажите путь к вашему изображению
#     img_path = "./img/image.jpg"
#     img = Image.open(img_path).convert("RGB")

#     print("1. Построение гистограммы изображения...")
#     plot_histogram(img)

#     print("2. Преобразование в полутоновое изображение...")
#     grayscale_img = to_grayscale(img)

#     print("3. Бинаризация изображения...")
#     binarized_img = binarize_image(grayscale_img)

#     print("4. Устранение шумов на бинарном изображении...")
#     binary_denoised = remove_noise_binary(binarized_img)

#     print("5. Усредняющий фильтр на полутоновом изображении...")
#     average_filtered_img = average_filter(grayscale_img)

#     print("6. Медианный фильтр на полутоновом изображении...")
#     median_filtered_img = median_filter(grayscale_img)

#     print("7. Выделение границ на бинарном изображении...")
#     binary_edges = edge_detection_binary(binary_denoised)

#     print("8. Выделение границ на полутоновом изображении...")
#     grayscale_edges = edge_detection_grayscale(grayscale_img)

#     print("Все операции завершены!")
