from __future__ import annotations

from functools import lru_cache
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle

from poc.config import ActionTimingConfig
from poc.grid_map import DEFAULT_LAYOUT_PATH, GridOccupancyMap

ROBOT_VISUAL_RADIUS_M = 0.18


def save_animation_media(
    anim_obj: animation.FuncAnimation,
    output_path: str | Path,
    fps: int = 12,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".html":
        path.write_text(anim_obj.to_jshtml(), encoding="utf-8")
        return path
    if suffix == ".gif":
        anim_obj.save(str(path), writer="pillow", fps=fps)
        return path
    if suffix == ".mp4":
        _save_animation_mp4(anim_obj, path, fps=fps)
        return path
    raise ValueError(f"Unsupported animation format: {suffix}")


def plot_match_overview(
    result: Any,
    figsize: tuple[int, int] = (14, 8),
    field_ylim: tuple[float, float] = (-1.0, 1.0),
    show_grid_map: bool = True,
    grid_alpha: float = 0.35,
):
    data = _result_to_dict(result)
    fig, ax_field, ax_score, ax_actions = _build_overview_figure(figsize)

    _plot_field(
        ax_field,
        data,
        field_ylim=field_ylim,
        show_grid_map=show_grid_map,
        grid_alpha=grid_alpha,
    )
    _plot_score(ax_score, data)
    _plot_action_timeline(ax_actions, data)
    fig.tight_layout()
    return fig


def animate_match_overview(
    result: Any,
    figsize: tuple[int, int] = (14, 8),
    field_ylim: tuple[float, float] = (-1.0, 1.0),
    interval: int = 120,
    frame_stride: int = 2,
    show_grid_map: bool = True,
    grid_alpha: float = 0.35,
):
    data = _result_to_dict(result)
    fig, ax_field, ax_score, ax_actions = _build_overview_figure(figsize)

    history = data["history"]
    initial_history = history[0] if history else None
    initial_state = _materialize_field_state(data, initial_history)
    grid_overlay = _plot_field_background(
        ax_field,
        data,
        field_ylim=field_ylim,
        grid_state=initial_state if show_grid_map else None,
        enemy_position=tuple(initial_history["enemy_position"]) if initial_history and show_grid_map else None,
        grid_alpha=grid_alpha,
    )
    _plot_score_background(ax_score, data)
    _plot_action_timeline(ax_actions, data)

    frame_indices = list(range(0, len(history), max(frame_stride, 1)))
    if not frame_indices or frame_indices[-1] != len(history) - 1:
        frame_indices.append(len(history) - 1)

    field_artists = _init_field_state_artists(ax_field, initial_state)

    our_line, = ax_field.plot([], [], color="#118ab2", linewidth=2, label="our robot")
    enemy_line, = ax_field.plot([], [], color="#ef476f", linewidth=2, label="enemy robot")
    our_marker, = ax_field.plot([], [], marker="o", markersize=10, linestyle="None", color="#118ab2")
    enemy_marker, = ax_field.plot([], [], marker="o", markersize=10, linestyle="None", color="#ef476f")
    our_radius = _add_robot_radius_circle(ax_field, color="#118ab2")
    enemy_radius = _add_robot_radius_circle(ax_field, color="#ef476f")
    if initial_history is not None:
        _update_robot_radius_circle(our_radius, tuple(initial_history["our_position"]))
        _update_robot_radius_circle(enemy_radius, tuple(initial_history["enemy_position"]))
    time_text = ax_field.text(
        0.02,
        0.02,
        "",
        transform=ax_field.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax_field.legend(loc="upper center", ncol=3)

    score_our_line, = ax_score.plot([], [], color="#118ab2", linewidth=2, label="our score")
    score_enemy_line, = ax_score.plot([], [], color="#ef476f", linewidth=2, label="enemy score")
    score_cursor = ax_score.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_score.legend()

    timeline_cursor = ax_actions.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    def update(frame_number: int):
        history_index = frame_indices[frame_number]
        current_history = history[: history_index + 1]
        current = current_history[-1]

        current_state = _materialize_field_state(data, current)
        if grid_overlay is not None:
            _update_grid_map_overlay(
                grid_overlay,
                current_state,
                tuple(current["enemy_position"]),
                alpha=grid_alpha,
            )
        _update_field_state_artists(field_artists, current_state)

        our_traj = [point["our_position"] for point in current_history]
        enemy_traj = [point["enemy_position"] for point in current_history]
        our_line.set_data([point[0] for point in our_traj], [point[1] for point in our_traj])
        enemy_line.set_data([point[0] for point in enemy_traj], [point[1] for point in enemy_traj])
        our_marker.set_data([current["our_position"][0]], [current["our_position"][1]])
        enemy_marker.set_data([current["enemy_position"][0]], [current["enemy_position"][1]])
        _update_robot_radius_circle(our_radius, tuple(current["our_position"]))
        _update_robot_radius_circle(enemy_radius, tuple(current["enemy_position"]))

        times = [entry["time"] for entry in current_history]
        our_scores = [entry["our_score"] for entry in current_history]
        enemy_scores = [entry["enemy_score"] for entry in current_history]
        score_our_line.set_data(times, our_scores)
        score_enemy_line.set_data(times, enemy_scores)

        current_time = current["time"]
        score_cursor.set_xdata([current_time, current_time])
        timeline_cursor.set_xdata([current_time, current_time])
        time_text.set_text(f"t = {current_time:.1f}s")

        return (
            *((grid_overlay,) if grid_overlay is not None else ()),
            our_line,
            enemy_line,
            our_marker,
            enemy_marker,
            our_radius,
            enemy_radius,
            score_our_line,
            score_enemy_line,
            score_cursor,
            timeline_cursor,
            time_text,
        )

    return animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        blit=False,
        repeat=True,
    )


def _save_animation_mp4(
    anim_obj: animation.FuncAnimation,
    output_path: Path,
    fps: int = 12,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MP4 export requires opencv-python (cv2).") from exc

    figure = anim_obj._fig
    writer = None
    codecs = ("mp4v", "avc1", "H264")

    try:
        anim_obj._init_draw()
        for frame_data in anim_obj.new_saved_frame_seq():
            anim_obj._draw_next_frame(frame_data, blit=False)
            figure.canvas.draw()
            width, height = figure.canvas.get_width_height()
            rgb = _canvas_to_rgb_array(figure.canvas, width=width, height=height)
            bgr = rgb[:, :, ::-1]

            if writer is None:
                for codec in codecs:
                    candidate = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*codec),
                        float(fps),
                        (width, height),
                    )
                    if candidate.isOpened():
                        writer = candidate
                        break
                    candidate.release()
                if writer is None:
                    raise RuntimeError(
                        "Failed to open MP4 writer via OpenCV. Tried codecs: "
                        + ", ".join(codecs)
                    )

            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()


def _canvas_to_rgb_array(canvas, width: int, height: int) -> np.ndarray:
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])

    if hasattr(canvas, "tostring_rgb"):
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        return rgb.reshape((height, width, 3))

    renderer = getattr(canvas, "renderer", None)
    if renderer is not None and hasattr(renderer, "buffer_rgba"):
        rgba = np.asarray(renderer.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])

    raise RuntimeError(
        "Matplotlib canvas does not expose a compatible pixel buffer for MP4 export."
    )


def _build_overview_figure(figsize: tuple[int, int]):
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_field = fig.add_subplot(grid[:, 0])
    ax_score = fig.add_subplot(grid[0, 1])
    ax_actions = fig.add_subplot(grid[1, 1])
    return fig, ax_field, ax_score, ax_actions


def _plot_field(
    ax,
    data: dict[str, Any],
    field_ylim: tuple[float, float],
    show_grid_map: bool = True,
    grid_alpha: float = 0.35,
) -> None:
    final_history = data["history"][-1] if data["history"] else None
    final_state = _materialize_field_state(data, final_history)
    _plot_field_background(
        ax,
        data,
        field_ylim=field_ylim,
        grid_state=final_state if show_grid_map else None,
        enemy_position=tuple(final_history["enemy_position"]) if final_history and show_grid_map else None,
        grid_alpha=grid_alpha,
    )
    _init_field_state_artists(ax, final_state)

    our_traj = [entry["our_position"] for entry in data["history"]]
    enemy_traj = [entry["enemy_position"] for entry in data["history"]]
    ax.plot([p[0] for p in our_traj], [p[1] for p in our_traj], color="#118ab2", linewidth=2, label="our robot")
    ax.plot([p[0] for p in enemy_traj], [p[1] for p in enemy_traj], color="#ef476f", linewidth=2, label="enemy robot")
    if final_history is not None:
        our_position = tuple(final_history["our_position"])
        enemy_position = tuple(final_history["enemy_position"])
        ax.plot([our_position[0]], [our_position[1]], marker="o", markersize=10, linestyle="None", color="#118ab2")
        ax.plot([enemy_position[0]], [enemy_position[1]], marker="o", markersize=10, linestyle="None", color="#ef476f")
        _update_robot_radius_circle(_add_robot_radius_circle(ax, color="#118ab2"), our_position)
        _update_robot_radius_circle(_add_robot_radius_circle(ax, color="#ef476f"), enemy_position)
    ax.legend(loc="upper center", ncol=3)


def _plot_field_background(
    ax,
    data: dict[str, Any],
    field_ylim: tuple[float, float],
    grid_state: dict[str, Any] | None = None,
    enemy_position: tuple[float, float] | None = None,
    grid_alpha: float = 0.35,
):
    width, _height = data["field_size"]
    ax.set_title("Field and Robot Trajectories")
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(*field_ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.add_patch(
        plt.Rectangle(
            (-width / 2, field_ylim[0]),
            width,
            field_ylim[1] - field_ylim[0],
            fill=False,
            linewidth=2,
        )
    )

    overlay_artist = None
    if grid_state is not None:
        overlay_artist = _add_grid_map_overlay(
            ax,
            grid_state,
            enemy_position=enemy_position,
            alpha=grid_alpha,
        )

    _draw_route_guides(ax, data)

    endgame = data["endgame"][data["our_side"]]
    chill_x, chill_y = endgame["chill_point"]
    ax.scatter(chill_x, chill_y, s=150, marker="P", color="#8338ec", label="our chill point", zorder=3)
    home_points = endgame["home_waypoints"]
    ax.plot([p[0] for p in home_points], [p[1] for p in home_points], "--", color="#8338ec", alpha=0.9, zorder=1)
    return overlay_artist


@lru_cache(maxsize=4)
def _grid_map_template_cached(layout_path: str, layout_mtime_ns: int) -> GridOccupancyMap:
    del layout_mtime_ns
    return GridOccupancyMap.from_layout(layout_path, team_color="all")


def _grid_map_template() -> GridOccupancyMap | None:
    if not DEFAULT_LAYOUT_PATH.exists():
        return None
    stat = DEFAULT_LAYOUT_PATH.stat()
    return _grid_map_template_cached(str(DEFAULT_LAYOUT_PATH), stat.st_mtime_ns)


def _add_grid_map_overlay(
    ax,
    field_state: dict[str, Any],
    enemy_position: tuple[float, float] | None,
    alpha: float,
):
    rgba, extent = _grid_overlay_rgba(field_state, enemy_position=enemy_position, alpha=alpha)
    if rgba is None:
        return None
    return ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        zorder=0.4,
    )


def _update_grid_map_overlay(
    overlay_artist,
    field_state: dict[str, Any],
    enemy_position: tuple[float, float] | None,
    alpha: float,
) -> None:
    rgba, extent = _grid_overlay_rgba(field_state, enemy_position=enemy_position, alpha=alpha)
    if rgba is None:
        return
    overlay_artist.set_data(rgba)
    overlay_artist.set_extent(extent)


def _grid_overlay_rgba(
    field_state: dict[str, Any],
    enemy_position: tuple[float, float] | None,
    alpha: float,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    template = _grid_map_template()
    if template is None:
        return None, None

    overlay_map = template.clone()
    overlay_map.active_start_ids = {
        str(source.get("map_obstacle_id") or key)
        for key, source in field_state["sources"].items()
        if int(source.get("available_items", 0)) > 0
        and source.get("state") != "empty"
        and bool(source.get("map_footprint_enabled", True))
        and str(source.get("map_obstacle_id") or key) in overlay_map.dynamic_start_ids
    }
    overlay_map.active_match_ids = {
        str(deposit.get("map_obstacle_id") or key)
        for key, deposit in field_state["deposits"].items()
        if (int(deposit.get("blue_items", 0)) + int(deposit.get("yellow_items", 0))) > 0
        and bool(deposit.get("map_footprint_enabled", True))
        and str(deposit.get("map_obstacle_id") or key) in overlay_map.match_obstacles
    }
    if enemy_position is not None:
        overlay_map.dynamic_circles = [
            (
                enemy_position[0],
                enemy_position[1],
                ActionTimingConfig().robot_separation_radius,
            )
        ]
    else:
        overlay_map.dynamic_circles = []
    overlay_map.rebuild()

    planning_map = overlay_map.planning_map
    if planning_map is None:
        return None, None

    occupied = planning_map > 0
    rgba = np.zeros((*planning_map.shape, 4), dtype=float)
    rgba[..., 0] = 0.10
    rgba[..., 1] = 0.13
    rgba[..., 2] = 0.16
    rgba[..., 3] = occupied.astype(float) * alpha

    origin_x, origin_y = overlay_map.config.origin_xy
    extent = (
        origin_x,
        origin_x + overlay_map.width_px * overlay_map.config.resolution_m,
        origin_y,
        origin_y + overlay_map.height_px * overlay_map.config.resolution_m,
    )
    return rgba, extent


def _draw_route_guides(ax, data: dict[str, Any]) -> None:
    for source in data["sources"].values():
        position = tuple(source["position"])
        for route in source.get("collect_routes", []):
            _draw_single_route_guide(ax, position, route["waypoints"], color="#adb5bd")

    for deposit in data["deposits"].values():
        position = tuple(deposit["position"])
        ring_radius = float(deposit.get("approach_ring_radius", 0.0) or 0.0)
        uses_ring_for_deposit = ring_radius > 0.0 and not deposit.get("deposit_routes")
        if uses_ring_for_deposit:
            ax.add_patch(
                Circle(
                    position,
                    ring_radius,
                    fill=False,
                    linestyle=":",
                    linewidth=1.2,
                    edgecolor="#ced4da",
                    alpha=0.7,
                    zorder=1,
                )
            )
        for route in deposit.get("deposit_routes", []):
            _draw_single_route_guide(ax, position, route["waypoints"], color="#ced4da")

    thermometer = data["thermometer"]
    thermometer_position = tuple(thermometer["position"])
    approach_point = _thermometer_approach_point(thermometer)
    ax.plot(
        [approach_point[0], thermometer_position[0]],
        [approach_point[1], thermometer_position[1]],
        linestyle=":",
        linewidth=1.1,
        color="#6a4c93",
        alpha=0.45,
        zorder=1,
    )
    ax.scatter(
        [approach_point[0]],
        [approach_point[1]],
        s=32,
        marker="x",
        color="#6a4c93",
        alpha=0.6,
        zorder=2,
    )
    for side, color in (("blue", "#118ab2"), ("yellow", "#ef476f")):
        slide_start = tuple(thermometer_position)
        slide_end = _thermometer_slide_target_for_side(thermometer, side)
        ax.plot(
            [slide_start[0], slide_end[0]],
            [slide_start[1], slide_end[1]],
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.2,
            zorder=1,
        )


def _draw_single_route_guide(ax, anchor: tuple[float, float], waypoints: Any, color: str) -> None:
    if not waypoints:
        return
    points = [anchor, *[tuple(point) for point in waypoints]]
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        linestyle=":",
        linewidth=1.0,
        color=color,
        alpha=0.55,
        zorder=1,
    )
    endpoint = points[-1]
    ax.scatter(endpoint[0], endpoint[1], s=28, marker="x", color=color, alpha=0.75, zorder=2)


def _add_robot_radius_circle(ax, color: str) -> Circle:
    circle = Circle(
        (0.0, 0.0),
        ROBOT_VISUAL_RADIUS_M,
        fill=False,
        linewidth=1.5,
        edgecolor=color,
        alpha=0.45,
        zorder=2.5,
    )
    ax.add_patch(circle)
    return circle


def _update_robot_radius_circle(circle: Circle, position: tuple[float, float]) -> None:
    circle.center = position


def _init_field_state_artists(ax, field_state: dict[str, Any]) -> dict[str, Any]:
    artists: dict[str, Any] = {
        "source_markers": {},
        "source_labels": {},
        "deposit_markers": {},
        "deposit_labels": {},
        "deposit_states": {},
        "thermometer_markers": {},
        "thermometer_traces": {},
        "mars_markers": {},
        "mars_labels": {},
    }

    for key, source in field_state["sources"].items():
        x, y = source["position"]
        marker, = ax.plot([], [], marker="s", markersize=9, linestyle="None", zorder=2)
        marker.set_data([x], [y])
        label = ax.text(x, y + 0.04, _format_semantic_id(source["semantic_id"]), ha="center", fontsize=8, zorder=3)
        artists["source_markers"][key] = marker
        artists["source_labels"][key] = label

    for key, deposit in field_state["deposits"].items():
        x, y = deposit["position"]
        marker_style = "D" if deposit["kind"] == "home" else "o"
        marker, = ax.plot([], [], marker=marker_style, markersize=10, linestyle="None", markeredgecolor="black", zorder=2)
        marker.set_data([x], [y])
        id_label = ax.text(x, y + 0.045, _format_semantic_id(deposit["semantic_id"]), ha="center", fontsize=8, zorder=3)
        state_label = ax.text(x, y - 0.055, "", ha="center", fontsize=7, color="#343a40", zorder=3)
        artists["deposit_markers"][key] = marker
        artists["deposit_labels"][key] = id_label
        artists["deposit_states"][key] = state_label

    tx, ty = field_state["thermometer"]["position"]
    approach_x, approach_y = field_state["thermometer"]["approach_point"]
    thermometer_base, = ax.plot(
        [],
        [],
        marker="o",
        markersize=9,
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="#6a4c93",
        markeredgewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    thermometer_base.set_data([tx], [ty])
    thermometer_approach, = ax.plot(
        [],
        [],
        marker="x",
        markersize=8,
        linestyle="None",
        color="#6a4c93",
        alpha=0.8,
        zorder=2,
    )
    thermometer_approach.set_data([approach_x], [approach_y])
    artists["thermometer_base"] = thermometer_base
    artists["thermometer_approach"] = thermometer_approach

    for side, color, marker_style in (
        ("blue", "#118ab2", "*"),
        ("yellow", "#ef476f", "P"),
    ):
        thermometer_trace, = ax.plot([], [], linewidth=2.4, alpha=0.65, color=color, zorder=2)
        thermometer_marker, = ax.plot([], [], marker=marker_style, markersize=13, linestyle="None", color=color, zorder=3)
        artists["thermometer_traces"][side] = thermometer_trace
        artists["thermometer_markers"][side] = thermometer_marker

    for side, color in (("blue", "#118ab2"), ("yellow", "#ef476f")):
        for mars in field_state.get("mars", {}).get(side, []):
            key = f"{side}:{mars['name']}"
            marker, = ax.plot([], [], marker="^", markersize=8, linestyle="None", color=color, alpha=0.5, zorder=3)
            label = ax.text(0.0, 0.0, "M", ha="center", fontsize=7, color=color, zorder=3)
            artists["mars_markers"][key] = marker
            artists["mars_labels"][key] = label

    _update_field_state_artists(artists, field_state)
    return artists


def _update_field_state_artists(artists: dict[str, Any], field_state: dict[str, Any]) -> None:
    current_time = float(field_state.get("time", 0.0))
    for key, source in field_state["sources"].items():
        marker = artists["source_markers"][key]
        label = artists["source_labels"][key]
        display = _source_display(source, current_time)
        visible = display["visible"]
        marker.set_visible(visible)
        label.set_visible(visible)
        if visible:
            marker.set_markerfacecolor(display["facecolor"])
            marker.set_markeredgecolor(display["edgecolor"])
            marker.set_alpha(display["alpha"])
            label.set_alpha(display["label_alpha"])

    for key, deposit in field_state["deposits"].items():
        marker = artists["deposit_markers"][key]
        state_label = artists["deposit_states"][key]
        total_items = int(deposit.get("blue_items", 0)) + int(deposit.get("yellow_items", 0))
        fill_color = _deposit_color(deposit)
        marker.set_markerfacecolor(fill_color)
        state_label.set_color(fill_color if total_items > 0 else "#343a40")
        marker.set_alpha(0.55 if total_items == 0 else 0.95)
        marker.set_markersize(10 + total_items)
        state_label.set_text(_deposit_state_text(deposit))

    thermometer = field_state["thermometer"]
    base_position = tuple(thermometer["position"])
    approach_point = tuple(thermometer.get("approach_point", _thermometer_approach_point(thermometer)))

    artists["thermometer_base"].set_data([base_position[0]], [base_position[1]])
    artists["thermometer_approach"].set_data([approach_point[0]], [approach_point[1]])

    for side, marker in artists["thermometer_markers"].items():
        trace = artists["thermometer_traces"][side]
        side_state = thermometer.get("sides", {}).get(side, {})
        visual_position = tuple(side_state.get("visual_position", base_position))
        is_animating = bool(side_state.get("is_animating", False))
        is_doing = bool(side_state.get("is_doing", False))
        is_done = bool(side_state.get("is_done", False))
        color = _thermometer_side_color(side)

        marker.set_data([visual_position[0]], [visual_position[1]])
        marker.set_color(color)
        marker.set_alpha(0.95 if is_animating or is_doing or is_done else 0.35)
        marker.set_markersize(16 if is_animating else (14 if is_done else (13 if is_doing else 11)))

        if _point_distance(base_position, visual_position) > 1e-6:
            trace.set_data(
                [base_position[0], visual_position[0]],
                [base_position[1], visual_position[1]],
            )
            trace.set_color(color)
            trace.set_alpha(0.7 if is_animating else (0.55 if is_doing else 0.45))
            trace.set_visible(True)
        else:
            trace.set_data([], [])
            trace.set_visible(False)

    for side in ("blue", "yellow"):
        for mars in field_state.get("mars", {}).get(side, []):
            key = f"{side}:{mars['name']}"
            marker = artists["mars_markers"][key]
            label = artists["mars_labels"][key]
            position = tuple(mars["position"])
            arrived = bool(mars.get("arrived", False))
            released = bool(mars.get("released", False))
            marker.set_data([position[0]], [position[1]])
            marker.set_alpha(0.95 if arrived else (0.75 if released else 0.35))
            marker.set_markersize(9 if arrived else 8)
            label.set_position((position[0], position[1] + 0.035))
            label.set_text("M")
            label.set_alpha(0.95 if arrived else (0.75 if released else 0.35))


def _materialize_field_state(data: dict[str, Any], history_entry: dict[str, Any] | None) -> dict[str, Any]:
    sources = {str(key): dict(value) for key, value in data["sources"].items()}
    deposits = {str(key): dict(value) for key, value in data["deposits"].items()}
    thermometer = dict(data["thermometer"])
    mars = {
        str(key): [dict(item) for item in value]
        for key, value in data.get("mars", {}).items()
    }
    current_time = 0.0

    if history_entry is not None:
        current_time = float(history_entry.get("time", 0.0))
        for key, value in history_entry.get("source_states", {}).items():
            source = sources.get(str(key))
            if source is None:
                continue
            source.update(value)

        for key, value in history_entry.get("deposit_states", {}).items():
            deposit = deposits.get(str(key))
            if deposit is None:
                continue
            deposit.update(value)

        if "thermometer_state" in history_entry:
            thermometer["state"] = history_entry["thermometer_state"]
        if "thermometer_doing_blue" in history_entry:
            thermometer["doing_blue"] = history_entry["thermometer_doing_blue"]
        if "thermometer_doing_yellow" in history_entry:
            thermometer["doing_yellow"] = history_entry["thermometer_doing_yellow"]
        if "mars_states" in history_entry:
            mars = {
                str(key): [dict(item) for item in value]
                for key, value in history_entry["mars_states"].items()
            }

    thermometer.update(_thermometer_visual_payload(data, thermometer, current_time))

    return {
        "sources": sources,
        "deposits": deposits,
        "thermometer": thermometer,
        "mars": mars,
        "time": current_time,
    }


def _plot_score(ax, data: dict[str, Any]) -> None:
    _plot_score_background(ax, data)
    times = [entry["time"] for entry in data["history"]]
    our_scores = [entry["our_score"] for entry in data["history"]]
    enemy_scores = [entry["enemy_score"] for entry in data["history"]]
    ax.plot(times, our_scores, label="our score", color="#118ab2", linewidth=2)
    ax.plot(times, enemy_scores, label="enemy score", color="#ef476f", linewidth=2)
    ax.legend()


def _plot_score_background(ax, data: dict[str, Any]) -> None:
    times = [entry["time"] for entry in data["history"]]
    scores = [entry["our_score"] for entry in data["history"]] + [entry["enemy_score"] for entry in data["history"]]
    ax.set_title("Score Progress")
    ax.set_xlabel("time, s")
    ax.set_ylabel("score")
    ax.set_xlim(0.0, max(times) if times else 100.0)
    ax.set_ylim(min(scores, default=0) - 2, max(scores, default=10) + 2)
    ax.grid(True, alpha=0.25)


def _plot_action_timeline(ax, data: dict[str, Any]) -> None:
    ax.set_title("Action Timeline (P=pick, D=deposit, A=attack, TH=thermo, EG=endgame)")
    y_positions = {"blue": 1, "yellow": 0}
    colors = {"blue": "#118ab2", "yellow": "#ef476f"}
    last_label_x = {"blue": -1e9, "yellow": -1e9}

    for entry in data["action_log"]:
        start = entry["time"]
        duration = max(entry["expected_duration"], 0.2)
        side = entry["side"]
        y_center = y_positions[side]
        ax.broken_barh(
            [(start, duration)],
            (y_center - 0.28, 0.56),
            facecolors=colors[side],
            alpha=0.75,
        )

        label = _compact_action_label(entry["action"])
        center_x = start + duration / 2
        should_draw = (
            entry["action"] == "START_ENDGAME"
            or duration >= 5.0 and center_x - last_label_x[side] >= 9.0
        )
        if should_draw:
            ax.text(
                center_x,
                y_center,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
                clip_on=True,
            )
            last_label_x[side] = center_x

    max_time = max(
        (entry["time"] + max(entry["expected_duration"], 0.2) for entry in data["action_log"]),
        default=100.0,
    )
    ax.set_xlim(0.0, max_time)
    ax.set_yticks([0, 1], labels=["yellow", "blue"])
    ax.set_xlabel("time, s")
    ax.grid(True, axis="x", alpha=0.25)


def _compact_action_label(label: str) -> str:
    if label.startswith("PICK_"):
        return "P" + _format_label_suffix(label.split("_", maxsplit=1)[1])
    if label.startswith("DEPOSIT_"):
        return "D" + _format_label_suffix(label.split("_", maxsplit=1)[1])
    if label.startswith("ATTACK_"):
        return "A" + _format_label_suffix(label.split("_", maxsplit=1)[1])
    if label == "THERMOMETER":
        return "TH"
    if label == "START_ENDGAME":
        return "EG"
    if label == "PLAY_TO_END":
        return "PE"
    if label == "WAIT":
        return "W"
    return label[:6]


def _source_color(state: str) -> str:
    return {
        "untouched": "#2a9d8f",
        "disturbed": "#e9c46a",
        "empty": "#bcb8b1",
    }.get(state, "#bcb8b1")


def _source_display(source: dict[str, Any], current_time: float) -> dict[str, Any]:
    available_from_t = float(source.get("available_from_t", 0.0))
    if current_time < available_from_t:
        return {
            "visible": True,
            "facecolor": "none",
            "edgecolor": "#adb5bd",
            "alpha": 0.7,
            "label_alpha": 0.55,
        }

    if source.get("state") == "empty" or int(source.get("available_items", 0)) <= 0:
        return {
            "visible": False,
            "facecolor": "none",
            "edgecolor": "#bcb8b1",
            "alpha": 0.0,
            "label_alpha": 0.0,
        }

    color = _source_color(source["state"])
    return {
        "visible": True,
        "facecolor": color,
        "edgecolor": color,
        "alpha": 1.0,
        "label_alpha": 1.0,
    }


def _deposit_color(deposit: dict[str, Any]) -> str:
    blue_items = int(deposit.get("blue_items", 0))
    yellow_items = int(deposit.get("yellow_items", 0))
    if deposit["kind"] == "storage":
        if blue_items > 0 and yellow_items == 0:
            return "#118ab2"
        if yellow_items > 0 and blue_items == 0:
            return "#ef476f"
        if blue_items > 0 and yellow_items > 0:
            return "#8d5fd3"
        return "#6c757d"
    if deposit["owner"] == "blue":
        return "#1d3557"
    if deposit["owner"] == "yellow":
        return "#d62828"
    return "#6c757d"


def _deposit_state_text(deposit: dict[str, Any]) -> str:
    blue_items = int(deposit.get("blue_items", 0))
    yellow_items = int(deposit.get("yellow_items", 0))
    if blue_items == 0 and yellow_items == 0:
        return ""
    return f"B{blue_items}/Y{yellow_items}"


def _thermometer_color(state: str) -> str:
    if state == "done_blue":
        return "#118ab2"
    if state == "done_yellow":
        return "#ef476f"
    if state == "done_both":
        return "#2a9d8f"
    return "#6a4c93"


def _thermometer_side_color(side: str) -> str:
    return "#118ab2" if side == "blue" else "#ef476f"


def _thermometer_visual_payload(
    data: dict[str, Any],
    thermometer: dict[str, Any],
    current_time: float,
) -> dict[str, Any]:
    approach_point = _thermometer_approach_point(thermometer)
    active_entries = _active_thermometer_actions(data.get("action_log", []), current_time)
    sides_payload: dict[str, dict[str, Any]] = {}

    for side in ("blue", "yellow"):
        active_entry = active_entries.get(side)
        is_done = _thermometer_side_is_done(str(thermometer.get("state", "not_done")), side)
        is_doing = bool(thermometer.get(f"doing_{side}", False))
        visual_position = _thermometer_final_position_for_side(thermometer, side) if is_done else tuple(thermometer["position"])
        is_animating = False

        if active_entry is not None:
            visual_position = _thermometer_position_during_action(
                thermometer,
                side,
                elapsed=max(0.0, current_time - float(active_entry["time"])),
                total_duration=float(active_entry["expected_duration"]),
            )
            is_animating = True

        sides_payload[side] = {
            "visual_position": visual_position,
            "is_animating": is_animating,
            "is_doing": is_doing,
            "is_done": is_done,
        }

    return {
        "approach_point": approach_point,
        "sides": sides_payload,
    }


def _active_thermometer_actions(
    action_log: list[dict[str, Any]],
    current_time: float,
) -> dict[str, dict[str, Any]]:
    active_entries = [
        entry
        for entry in action_log
        if entry.get("action") == "THERMOMETER"
        and float(entry.get("time", 0.0)) <= current_time < float(entry.get("time", 0.0)) + float(entry.get("expected_duration", 0.0))
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for entry in active_entries:
        side = str(entry.get("side"))
        previous = grouped.get(side)
        if previous is None or float(entry.get("time", 0.0)) > float(previous.get("time", 0.0)):
            grouped[side] = entry
    return grouped


def _thermometer_approach_point(thermometer: dict[str, Any]) -> tuple[float, float]:
    point = thermometer.get("approach_point")
    if point is None:
        return (0.0, -0.77)
    return tuple(point)


def _thermometer_route_for_side(
    thermometer: dict[str, Any],
    side: str,
) -> tuple[tuple[float, float], ...]:
    route = thermometer.get(f"{side}_route")
    if route:
        return tuple(tuple(point) for point in route)

    direction = 1.0 if side == "blue" else -1.0
    return (
        (0.0, -0.70),
        (0.0, -0.77),
        (0.63 * direction, -0.77),
        (0.63 * direction, -0.65),
        (0.80 * direction, -0.65),
    )


def _thermometer_drag_route_for_side(
    thermometer: dict[str, Any],
    side: str,
) -> tuple[tuple[float, float], ...]:
    route = _thermometer_route_for_side(thermometer, side)
    if len(route) >= 4:
        return route[1:4]
    if len(route) >= 2:
        return route[1:]
    return route


def _thermometer_slide_target_for_side(
    thermometer: dict[str, Any],
    side: str,
) -> tuple[float, float]:
    route = _thermometer_route_for_side(thermometer, side)
    base_position = tuple(thermometer["position"])
    if len(route) >= 3:
        return (route[2][0], base_position[1])
    if route:
        return (route[-1][0], base_position[1])
    return base_position


def _thermometer_side_is_done(state: str, side: str) -> bool:
    if side == "blue":
        return state in ("done_blue", "done_both")
    return state in ("done_yellow", "done_both")


def _thermometer_final_position_for_side(
    thermometer: dict[str, Any],
    side: str,
) -> tuple[float, float]:
    return _thermometer_slide_target_for_side(thermometer, side)


def _thermometer_position_during_action(
    thermometer: dict[str, Any],
    side: str,
    elapsed: float,
    total_duration: float,
) -> tuple[float, float]:
    route = _thermometer_route_for_side(thermometer, side)
    base_position = tuple(thermometer["position"])
    slide_target = _thermometer_slide_target_for_side(thermometer, side)
    if len(route) < 3:
        return tuple(thermometer["position"])

    timing = ActionTimingConfig()
    travel_duration = max(0.0, total_duration - timing.thermometer_duration)
    if travel_duration <= 1e-9:
        return slide_target

    path_distances = [_point_distance(a, b) for a, b in zip(route[:-1], route[1:])]
    total_path_distance = sum(path_distances)
    if total_path_distance <= 1e-9:
        return slide_target

    pre_slide_distance = path_distances[0] if path_distances else 0.0
    slide_distance = _point_distance(route[1], route[2])
    if slide_distance <= 1e-9:
        return slide_target

    travelled = total_path_distance * min(max(elapsed, 0.0), travel_duration) / travel_duration
    if travelled <= pre_slide_distance:
        return base_position
    if travelled >= pre_slide_distance + slide_distance:
        return slide_target

    progress = (travelled - pre_slide_distance) / slide_distance
    current_x = base_position[0] + (slide_target[0] - base_position[0]) * progress
    return (current_x, base_position[1])


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def _interpolate_point(
    start: tuple[float, float],
    end: tuple[float, float],
    progress: float,
) -> tuple[float, float]:
    clamped = min(max(progress, 0.0), 1.0)
    return (
        start[0] + (end[0] - start[0]) * clamped,
        start[1] + (end[1] - start[1]) * clamped,
    )


def _format_semantic_id(semantic_id: int) -> str:
    return f"{semantic_id:02d}" if 0 < semantic_id < 100 else str(semantic_id)


def _format_label_suffix(value: str) -> str:
    if value.isdigit():
        return _format_semantic_id(int(value))
    return value


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        history = [_dataclass_to_dict(entry) for entry in result.history]
        action_log = [_dataclass_to_dict(entry) for entry in result.action_log]
        return {
            "field_size": list(result.field_size),
            "history": history,
            "action_log": action_log,
            "sources": result.sources,
            "deposits": result.deposits,
            "thermometer": result.thermometer,
            "endgame": result.endgame,
            "mars": result.mars,
            "our_side": result.our_side,
        }
    return {
        "field_size": list(result.field_size),
        "history": [_dataclass_to_dict(entry) for entry in result.history],
        "action_log": [_dataclass_to_dict(entry) for entry in result.action_log],
        "sources": result.sources,
        "deposits": result.deposits,
        "thermometer": result.thermometer,
        "endgame": result.endgame,
        "mars": result.mars,
        "our_side": result.our_side,
    }


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        payload[field.name] = value.value if hasattr(value, "value") else value
    return payload
