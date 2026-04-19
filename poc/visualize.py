from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle


def plot_match_overview(
    result: Any,
    figsize: tuple[int, int] = (14, 8),
    field_ylim: tuple[float, float] = (-1.0, 1.0),
):
    data = _result_to_dict(result)
    fig, ax_field, ax_score, ax_actions = _build_overview_figure(figsize)

    _plot_field(ax_field, data, field_ylim=field_ylim)
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
):
    data = _result_to_dict(result)
    fig, ax_field, ax_score, ax_actions = _build_overview_figure(figsize)

    _plot_field_background(ax_field, data, field_ylim=field_ylim)
    _plot_score_background(ax_score, data)
    _plot_action_timeline(ax_actions, data)

    history = data["history"]
    frame_indices = list(range(0, len(history), max(frame_stride, 1)))
    if not frame_indices or frame_indices[-1] != len(history) - 1:
        frame_indices.append(len(history) - 1)

    initial_state = _materialize_field_state(data, history[0] if history else None)
    field_artists = _init_field_state_artists(ax_field, initial_state)

    our_line, = ax_field.plot([], [], color="#118ab2", linewidth=2, label="our robot")
    enemy_line, = ax_field.plot([], [], color="#ef476f", linewidth=2, label="enemy robot")
    our_marker, = ax_field.plot([], [], marker="o", markersize=10, linestyle="None", color="#118ab2")
    enemy_marker, = ax_field.plot([], [], marker="o", markersize=10, linestyle="None", color="#ef476f")
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
        _update_field_state_artists(field_artists, current_state)

        our_traj = [point["our_position"] for point in current_history]
        enemy_traj = [point["enemy_position"] for point in current_history]
        our_line.set_data([point[0] for point in our_traj], [point[1] for point in our_traj])
        enemy_line.set_data([point[0] for point in enemy_traj], [point[1] for point in enemy_traj])
        our_marker.set_data([current["our_position"][0]], [current["our_position"][1]])
        enemy_marker.set_data([current["enemy_position"][0]], [current["enemy_position"][1]])

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
            our_line,
            enemy_line,
            our_marker,
            enemy_marker,
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


def _build_overview_figure(figsize: tuple[int, int]):
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_field = fig.add_subplot(grid[:, 0])
    ax_score = fig.add_subplot(grid[0, 1])
    ax_actions = fig.add_subplot(grid[1, 1])
    return fig, ax_field, ax_score, ax_actions


def _plot_field(ax, data: dict[str, Any], field_ylim: tuple[float, float]) -> None:
    _plot_field_background(ax, data, field_ylim=field_ylim)
    final_state = _materialize_field_state(data, data["history"][-1] if data["history"] else None)
    _init_field_state_artists(ax, final_state)

    our_traj = [entry["our_position"] for entry in data["history"]]
    enemy_traj = [entry["enemy_position"] for entry in data["history"]]
    ax.plot([p[0] for p in our_traj], [p[1] for p in our_traj], color="#118ab2", linewidth=2, label="our robot")
    ax.plot([p[0] for p in enemy_traj], [p[1] for p in enemy_traj], color="#ef476f", linewidth=2, label="enemy robot")
    ax.legend(loc="upper center", ncol=3)


def _plot_field_background(ax, data: dict[str, Any], field_ylim: tuple[float, float]) -> None:
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

    _draw_route_guides(ax, data)

    endgame = data["endgame"][data["our_side"]]
    chill_x, chill_y = endgame["chill_point"]
    ax.scatter(chill_x, chill_y, s=150, marker="P", color="#8338ec", label="our chill point", zorder=3)
    home_points = endgame["home_waypoints"]
    ax.plot([p[0] for p in home_points], [p[1] for p in home_points], "--", color="#8338ec", alpha=0.9, zorder=1)


def _draw_route_guides(ax, data: dict[str, Any]) -> None:
    for source in data["sources"].values():
        position = tuple(source["position"])
        for route in source.get("collect_routes", []):
            _draw_single_route_guide(ax, position, route["waypoints"], color="#adb5bd")

    for deposit in data["deposits"].values():
        position = tuple(deposit["position"])
        ring_radius = float(deposit.get("approach_ring_radius", 0.0) or 0.0)
        if ring_radius > 0.0:
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


def _init_field_state_artists(ax, field_state: dict[str, Any]) -> dict[str, Any]:
    artists: dict[str, Any] = {
        "source_markers": {},
        "source_labels": {},
        "deposit_markers": {},
        "deposit_labels": {},
        "deposit_states": {},
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
    thermometer_marker, = ax.plot([], [], marker="*", markersize=14, linestyle="None", zorder=2)
    thermometer_marker.set_data([tx], [ty])
    artists["thermometer_marker"] = thermometer_marker

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

    artists["thermometer_marker"].set_color(_thermometer_color(field_state["thermometer"]["state"]))


def _materialize_field_state(data: dict[str, Any], history_entry: dict[str, Any] | None) -> dict[str, Any]:
    sources = {str(key): dict(value) for key, value in data["sources"].items()}
    deposits = {str(key): dict(value) for key, value in data["deposits"].items()}
    thermometer = dict(data["thermometer"])
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

    return {
        "sources": sources,
        "deposits": deposits,
        "thermometer": thermometer,
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
        "our_side": result.our_side,
    }


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        payload[field.name] = value.value if hasattr(value, "value") else value
    return payload
