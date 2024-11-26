import matplotlib.pyplot as plt
import numpy as np


def inside(point, edge_start, edge_end):
    dx = edge_end[0] - edge_start[0]
    dy = edge_end[1] - edge_start[1]
    return (point[0] - edge_start[0]) * dy - (point[1] - edge_start[1]) * dx >= 0


def intersection(s, p, edge_start, edge_end):
    dx = p[0] - s[0]
    dy = p[1] - s[1]
    edge_dx = edge_end[0] - edge_start[0]
    edge_dy = edge_end[1] - edge_start[1]

    determinant = dx * edge_dy - dy * edge_dx
    if determinant == 0:
        return None

    t = (
        (edge_start[0] - s[0]) * edge_dy - (edge_start[1] - s[1]) * edge_dx
    ) / determinant
    return [s[0] + t * dx, s[1] + t * dy]


def sutherland_hodgman(polygon, clipper):
    output_list = polygon
    for i in range(len(clipper)):
        input_list = output_list
        output_list = []

        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]

        for j in range(len(input_list)):
            s = input_list[j - 1]
            p = input_list[j]

            if inside(p, edge_start, edge_end):
                if not inside(s, edge_start, edge_end):
                    output_list.append(intersection(s, p, edge_start, edge_end))
                output_list.append(p)
            elif inside(s, edge_start, edge_end):
                output_list.append(intersection(s, p, edge_start, edge_end))

    return output_list


def plot_polygon(polygon, color, label):
    if polygon:  # Проверка на пустой список
        polygon = np.array(polygon + [polygon[0]])  # Замыкаем контур
        plt.plot(polygon[:, 0], polygon[:, 1], color=color, label=label)
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3)


def generate_star(center, radius1, radius2, points):
    """
    Генерация координат звезды.
    :param center: Центр звезды (x, y).
    :param radius1: Радиус для вершин на концах лучей.
    :param radius2: Радиус для внутренних вершин.
    :param points: Количество концов звезды.
    """
    angle_step = np.pi / points
    star = []
    for i in range(2 * points):
        radius = radius1 if i % 2 == 0 else radius2
        angle = i * angle_step
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        star.append([x, y])
    return star


# Задаем многоугольник (звезда) и отсекающую границу
polygon = generate_star(center=(3, 3), radius1=2, radius2=1, points=5)
clipper = [[2, 1.5], [2, 4.5], [4, 4], [4, 2]]

# Выполняем отсечение
clipped_polygon = sutherland_hodgman(polygon, clipper)

# Визуализация
plt.figure(figsize=(8, 8))
plot_polygon(polygon, "blue", "Исходный многоугольник")
plot_polygon(clipper, "red", "Отсекающая граница")
plot_polygon(clipped_polygon, "green", "Усеченный многоугольник")

plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.grid(True)
plt.show()
