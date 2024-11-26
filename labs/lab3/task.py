# var 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def compute_control_points(x, y):
    """
    Рассчитывает опорные точки (контрольные) для кривых Безье.
    :param x: Список координат x.
    :param y: Список координат y.
    :return: Координаты опорных точек для каждой кривой.
    """
    n = len(x) - 1
    control_points = []
    for i in range(1, n):
        prev_x, prev_y = x[i - 1], y[i - 1]
        cur_x, cur_y = x[i], y[i]
        next_x, next_y = x[i + 1], y[i + 1]

        # Вычисление длины шага
        step1 = (cur_x - prev_x) / 3
        step2 = (next_x - cur_x) / 3

        # Координаты контрольных точек
        control_point1 = (
            cur_x - step1,
            cur_y - step1 * (cur_y - prev_y) / (cur_x - prev_x),
        )
        control_point2 = (
            cur_x + step2,
            cur_y + step2 * (next_y - cur_y) / (next_x - cur_x),
        )

        control_points.append((control_point1, control_point2))

    return control_points


def plot_bezier_curve(x, y):
    """
    Строит кривые Безье для заданного набора точек.
    :param x: Список координат x.
    :param y: Список координат y.
    """
    control_points = compute_control_points(x, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, "o", label="Исходные точки", markersize=8, color="red")

    # Построение кривых Безье
    for i in range(1, len(x) - 1):
        prev_x, prev_y = x[i - 1], y[i - 1]
        cur_x, cur_y = x[i], y[i]
        next_x, next_y = x[i + 1], y[i + 1]

        (cp1_x, cp1_y), (cp2_x, cp2_y) = control_points[i - 1]

        # Генерация кривой с помощью функции scipy
        bezier_x = [prev_x, cp1_x, cp2_x, next_x]
        bezier_y = [prev_y, cp1_y, cp2_y, next_y]

        spline_x = make_interp_spline(bezier_x, bezier_y, k=3)
        fine_x = np.linspace(prev_x, next_x, 100)
        fine_y = spline_x(fine_x)

        ax.plot(fine_x, fine_y, label=f"Сегмент {i}", color="blue")
        ax.plot(
            [prev_x, cp1_x], [prev_y, cp1_y], "--", color="gray", alpha=0.6
        )  # Линия к первой опорной точке
        ax.plot(
            [next_x, cp2_x], [next_y, cp2_y], "--", color="gray", alpha=0.6
        )  # Линия ко второй опорной точке

    ax.legend()
    ax.set_title("Кривые Безье")
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()


# Пример входных данных
x_points = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
y_points = [
    200,
    300,
    100,
    170,
    120,
    250,
    300,
    50,
    100,
    200,
    150,
    300,
    120,
    200,
    230,
    200,
]

plot_bezier_curve(x_points, y_points)


# var 2

# import matplotlib.pyplot as plt
# import numpy as np

# # Исходные точки графика
# graphic_points = [
#     {"x": 0, "y": 200},
#     {"x": 50, "y": 300},
#     {"x": 100, "y": 100},
#     {"x": 150, "y": 170},
#     {"x": 200, "y": 120},
#     {"x": 250, "y": 250},
#     {"x": 300, "y": 300},
#     {"x": 350, "y": 50},
#     {"x": 400, "y": 100},
#     {"x": 450, "y": 200},
#     {"x": 500, "y": 150},
#     {"x": 550, "y": 300},
#     {"x": 600, "y": 120},
#     {"x": 650, "y": 200},
#     {"x": 700, "y": 230},
#     {"x": 750, "y": 200},
# ]


# # Вспомогательная функция для расчета контрольных точек
# def get_control_points(p0, p1, p2, tension=0.4):
#     d01 = np.sqrt((p1["x"] - p0["x"]) ** 2 + (p1["y"] - p0["y"]) ** 2)
#     d12 = np.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)

#     fa = tension * d01 / (d01 + d12)
#     fb = tension * d12 / (d01 + d12)

#     cp1 = {
#         "x": p1["x"] - fa * (p2["x"] - p0["x"]),
#         "y": p1["y"] - fa * (p2["y"] - p0["y"]),
#     }
#     cp2 = {
#         "x": p1["x"] + fb * (p2["x"] - p0["x"]),
#         "y": p1["y"] + fb * (p2["y"] - p0["y"]),
#     }

#     return cp1, cp2


# # Построение кривых Безье
# def draw_bezier_curve(ax):
#     x_vals = []
#     y_vals = []

#     for i in range(1, len(graphic_points) - 1):
#         p0 = graphic_points[i - 1]
#         p1 = graphic_points[i]
#         p2 = graphic_points[i + 1]

#         # Контрольные точки
#         cp1, cp2 = get_control_points(p0, p1, p2)

#         # Генерация кривой Безье
#         t = np.linspace(0, 1, 100)
#         x = (
#             (1 - t) ** 3 * p0["x"]
#             + 3 * (1 - t) ** 2 * t * cp1["x"]
#             + 3 * (1 - t) * t**2 * cp2["x"]
#             + t**3 * p2["x"]
#         )
#         y = (
#             (1 - t) ** 3 * p0["y"]
#             + 3 * (1 - t) ** 2 * t * cp1["y"]
#             + 3 * (1 - t) * t**2 * cp2["y"]
#             + t**3 * p2["y"]
#         )

#         x_vals.extend(x)
#         y_vals.extend(y)

#     ax.plot(x_vals, y_vals, linewidth=2, label="Кривая Безье")


# # Рисование исходных данных и кривых
# def plot_chart():
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # Исходные точки
#     x_points = [p["x"] for p in graphic_points]
#     y_points = [p["y"] for p in graphic_points]

#     ax.plot(x_points, y_points, "o--", label="Исходные точки", color="black")

#     # Кривая Безье
#     draw_bezier_curve(ax)

#     ax.axhline(200, color="black", linewidth=1)  # Горизонтальная ось
#     ax.axvline(0, color="black", linewidth=1)  # Вертикальная ось
#     ax.set_xlim(-50, 800)
#     ax.set_ylim(0, 400)
#     ax.legend()
#     ax.set_aspect("equal")
#     plt.show()


# # Построение графика
# plot_chart()
