import matplotlib.pyplot as plt
import numpy as np

# Исходные точки графика
graphic_points = [
    {"x": 0, "y": 200},
    {"x": 50, "y": 300},
    {"x": 100, "y": 100},
    {"x": 150, "y": 170},
    {"x": 200, "y": 120},
    {"x": 250, "y": 250},
    {"x": 300, "y": 300},
    {"x": 350, "y": 50},
    {"x": 400, "y": 100},
    {"x": 450, "y": 200},
    {"x": 500, "y": 150},
    {"x": 550, "y": 300},
    {"x": 600, "y": 120},
    {"x": 650, "y": 200},
    {"x": 700, "y": 230},
    {"x": 750, "y": 200},
]


# Функция для рисования осей
def draw_axis(ax):
    ax.axhline(200, color="black", linewidth=2)  # Горизонтальная ось
    ax.axvline(0, color="black", linewidth=2)  # Вертикальная ось
    for point in graphic_points:
        ax.plot(
            [point["x"], point["x"]], [190, 210], color="black", linewidth=1
        )  # Засечки по x


# Функция для рисования изначального графика
def draw_init_chart(ax):
    x = [p["x"] for p in graphic_points]
    y = [p["y"] for p in graphic_points]
    ax.plot(x, y, linestyle="--", color="black", label="Изначальный график")


# Вспомогательные функции для вычисления контрольных точек кривых Безье
def get_k(prev, current, next_, xStretch, xStretchSqr):
    left_distance = current["y"] - prev["y"]
    right_distance = current["y"] - next_["y"]
    if next_["y"] - prev["y"] != 0:
        return (
            np.sqrt(
                (xStretchSqr + left_distance**2) * (xStretchSqr + right_distance**2)
            )
            - xStretchSqr
            - left_distance * right_distance
        ) / (xStretch * (next_["y"] - prev["y"]))
    return 0


def get_delta_x(k, xStretch):
    return xStretch / 2 * np.sqrt(1 / (1 + k**2))


def get_left_control_point(current, k, xStretch):
    delta_x = get_delta_x(k, xStretch)
    return {"x": current["x"] - delta_x, "y": current["y"] - k * delta_x}


def get_right_control_point(current, k, xStretch):
    delta_x = get_delta_x(k, xStretch)
    return {"x": current["x"] + delta_x, "y": current["y"] + k * delta_x}


# Функция для рисования графика с использованием кривых Безье
def draw_graphic_using_bezier(ax):
    xStretch = 50
    xStretchSqr = xStretch**2

    current = graphic_points[0]
    right_control_point = None
    x_bezier = [current["x"]]
    y_bezier = [current["y"]]

    for i in range(1, len(graphic_points)):
        prev = current
        current = graphic_points[i]
        is_last_turn = i == len(graphic_points) - 1

        if not is_last_turn:
            k = get_k(prev, current, graphic_points[i + 1], xStretch, xStretchSqr)
            left_control_point = get_left_control_point(current, k, xStretch)

        if i == 1:
            x_bezier.append(left_control_point["x"])
            y_bezier.append(left_control_point["y"])
        elif is_last_turn:
            x_bezier.append(right_control_point["x"])
            y_bezier.append(right_control_point["y"])
        else:
            x_bezier.append(right_control_point["x"])
            y_bezier.append(right_control_point["y"])
            x_bezier.append(left_control_point["x"])
            y_bezier.append(left_control_point["y"])

        right_control_point = get_right_control_point(current, k, xStretch)

    ax.plot(
        x_bezier,
        y_bezier,
        color="red",
        linewidth=2,
        label="Аппроксимация кривыми Безье",
    )


# Основная программа
fig, ax = plt.subplots(figsize=(10, 5))
draw_axis(ax)
draw_init_chart(ax)
draw_graphic_using_bezier(ax)

ax.set_xlim(-50, 800)
ax.set_ylim(0, 400)
ax.legend()
ax.set_aspect("equal")
plt.show()
