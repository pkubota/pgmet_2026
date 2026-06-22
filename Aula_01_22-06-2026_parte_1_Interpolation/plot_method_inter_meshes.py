#!/usr/bin/env python3
"""
Interpola temperaturas aleatorias em uma malha retangular latitude/longitude.

As temperaturas aleatorias representam observacoes espalhadas dentro da area.
Depois elas sao interpoladas para:
  - vertices da malha;
  - pontos medios das arestas horizontais;
  - pontos medios das arestas verticais;
  - centros das celulas.

Metodos disponiveis:
  nearest : vizinho mais proximo
  linear  : interpolacao linear por triangulacao
  cubic   : interpolacao cubica 2D por triangulacao
  idw     : inverse distance weighting
  rbf     : radial basis function
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf, griddata


@dataclass(frozen=True)
class MeshTargets:
    lon_vertices: np.ndarray
    lat_vertices: np.ndarray
    lon_edges_h: np.ndarray
    lat_edges_h: np.ndarray
    lon_edges_v: np.ndarray
    lat_edges_v: np.ndarray
    lon_centers: np.ndarray
    lat_centers: np.ndarray


@dataclass(frozen=True)
class InterpolationResult:
    vertices: np.ndarray
    edges_h: np.ndarray
    edges_v: np.ndarray
    centers: np.ndarray


def build_mesh_targets(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, MeshTargets]:
    """Cria os pontos da malha onde a temperatura sera calculada."""
    lons = np.linspace(lon_min, lon_max, nx + 1)
    lats = np.linspace(lat_min, lat_max, ny + 1)

    lon_vertices, lat_vertices = np.meshgrid(lons, lats)

    lon_mid = 0.5 * (lons[:-1] + lons[1:])
    lat_mid = 0.5 * (lats[:-1] + lats[1:])

    lon_edges_h, lat_edges_h = np.meshgrid(lon_mid, lats)
    lon_edges_v, lat_edges_v = np.meshgrid(lons, lat_mid)
    lon_centers, lat_centers = np.meshgrid(lon_mid, lat_mid)

    targets = MeshTargets(
        lon_vertices=lon_vertices,
        lat_vertices=lat_vertices,
        lon_edges_h=lon_edges_h,
        lat_edges_h=lat_edges_h,
        lon_edges_v=lon_edges_v,
        lat_edges_v=lat_edges_v,
        lon_centers=lon_centers,
        lat_centers=lat_centers,
    )
    return lons, lats, targets


def random_temperature_points(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    n_points: int,
    temp_min: float,
    temp_max: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera observacoes aleatorias de temperatura dentro da area."""
    rng = np.random.default_rng(seed)
    lon_obs = rng.uniform(lon_min, lon_max, n_points)
    lat_obs = rng.uniform(lat_min, lat_max, n_points)
    temp_obs = rng.uniform(temp_min, temp_max, n_points)
    return lon_obs, lat_obs, temp_obs


def interpolate_griddata(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    temp_obs: np.ndarray,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpola com scipy.griddata e preenche bordas com nearest."""
    points = np.column_stack((lon_obs, lat_obs))
    targets = np.column_stack((lon_target.ravel(), lat_target.ravel()))

    values = griddata(points, temp_obs, targets, method=method)
    values = values.reshape(lon_target.shape)

    if np.isnan(values).any():
        nearest = griddata(points, temp_obs, targets, method="nearest")
        nearest = nearest.reshape(lon_target.shape)
        values = np.where(np.isnan(values), nearest, values)

    return values


def interpolate_idw(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    temp_obs: np.ndarray,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Interpolacao IDW: pesos inversamente proporcionais a distancia."""
    target = np.column_stack((lon_target.ravel(), lat_target.ravel()))
    obs = np.column_stack((lon_obs, lat_obs))

    diff = target[:, None, :] - obs[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    exact = dist == 0.0
    weights = np.zeros_like(dist)
    np.divide(1.0, dist**power, out=weights, where=~exact)

    weighted_sum = weights @ temp_obs
    weight_total = weights.sum(axis=1)
    values = weighted_sum / weight_total

    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        exact_cols = exact[exact_rows].argmax(axis=1)
        values[exact_rows] = temp_obs[exact_cols]

    return values.reshape(lon_target.shape)


def interpolate_rbf(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    temp_obs: np.ndarray,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
) -> np.ndarray:
    """Interpolacao por funcao de base radial."""
    rbf = Rbf(lon_obs, lat_obs, temp_obs, function="multiquadric", smooth=0.1)
    return rbf(lon_target, lat_target)


def interpolate_to_all_targets(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    temp_obs: np.ndarray,
    targets: MeshTargets,
    method: str,
) -> InterpolationResult:
    """Calcula temperaturas em vertices, arestas e centros."""

    def interp(lon_target: np.ndarray, lat_target: np.ndarray) -> np.ndarray:
        if method in {"nearest", "linear", "cubic"}:
            return interpolate_griddata(
                lon_obs,
                lat_obs,
                temp_obs,
                lon_target,
                lat_target,
                method,
            )
        if method == "idw":
            return interpolate_idw(lon_obs, lat_obs, temp_obs, lon_target, lat_target)
        if method == "rbf":
            return interpolate_rbf(lon_obs, lat_obs, temp_obs, lon_target, lat_target)
        raise ValueError(f"Metodo desconhecido: {method}")

    return InterpolationResult(
        vertices=interp(targets.lon_vertices, targets.lat_vertices),
        edges_h=interp(targets.lon_edges_h, targets.lat_edges_h),
        edges_v=interp(targets.lon_edges_v, targets.lat_edges_v),
        centers=interp(targets.lon_centers, targets.lat_centers),
    )


def save_table(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    temp: np.ndarray,
    method: str,
    point_type: str,
) -> None:
    """Salva uma tabela lon, lat, temperatura."""
    header = "method,point_type,lon,lat,temperature"
    rows = np.column_stack(
        (
            lon.ravel(),
            lat.ravel(),
            temp.ravel(),
        )
    )
    with path.open("w", encoding="utf-8") as file:
        file.write(header + "\n")
        for lon_i, lat_i, temp_i in rows:
            file.write(f"{method},{point_type},{lon_i:.8f},{lat_i:.8f},{temp_i:.6f}\n")


def plot_method(
    path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    temp_obs: np.ndarray,
    targets: MeshTargets,
    result: InterpolationResult,
    method: str,
) -> None:
    """Gera uma figura para um metodo de interpolacao."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for lat in lats:
        ax.plot([lons[0], lons[-1]], [lat, lat], color="0.75", linewidth=0.7)
    for lon in lons:
        ax.plot([lon, lon], [lats[0], lats[-1]], color="0.75", linewidth=0.7)

    image = ax.scatter(
        targets.lon_centers,
        targets.lat_centers,
        c=result.centers,
        cmap="coolwarm",
        marker="s",
        s=42,
        label="centros",
    )
    ax.scatter(
        targets.lon_vertices,
        targets.lat_vertices,
        c=result.vertices,
        cmap="coolwarm",
        marker="o",
        s=18,
        edgecolor="black",
        linewidth=0.25,
        label="vertices",
    )
    ax.scatter(
        targets.lon_edges_h,
        targets.lat_edges_h,
        c=result.edges_h,
        cmap="coolwarm",
        marker="_",
        s=90,
        label="arestas horizontais",
    )
    ax.scatter(
        targets.lon_edges_v,
        targets.lat_edges_v,
        c=result.edges_v,
        cmap="coolwarm",
        marker="|",
        s=90,
        label="arestas verticais",
    )
    ax.scatter(
        lon_obs,
        lat_obs,
        c=temp_obs,
        cmap="coolwarm",
        marker="x",
        s=55,
        linewidth=1.5,
        label="observacoes",
    )

    fig.colorbar(image, ax=ax, label="Temperatura")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Temperatura interpolada - metodo {method}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    lon_min, lon_max = -50.0, -40.0
    lat_min, lat_max = -30.0, -20.0
    nx, ny = 10, 10

    n_points = 45
    temp_min, temp_max = 15.0, 35.0
    seed = 42

    methods = ["nearest", "linear", "cubic", "idw", "rbf"]
    output_dir = Path("saida_interpolacao")
    output_dir.mkdir(exist_ok=True)

    lons, lats, targets = build_mesh_targets(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        nx,
        ny,
    )
    lon_obs, lat_obs, temp_obs = random_temperature_points(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        n_points,
        temp_min,
        temp_max,
        seed,
    )

    save_table(
        output_dir / "observacoes.csv",
        lon_obs,
        lat_obs,
        temp_obs,
        "original",
        "observacao",
    )

    for method in methods:
        result = interpolate_to_all_targets(lon_obs, lat_obs, temp_obs, targets, method)

        save_table(
            output_dir / f"{method}_vertices.csv",
            targets.lon_vertices,
            targets.lat_vertices,
            result.vertices,
            method,
            "vertice",
        )
        save_table(
            output_dir / f"{method}_arestas_horizontais.csv",
            targets.lon_edges_h,
            targets.lat_edges_h,
            result.edges_h,
            method,
            "aresta_horizontal",
        )
        save_table(
            output_dir / f"{method}_arestas_verticais.csv",
            targets.lon_edges_v,
            targets.lat_edges_v,
            result.edges_v,
            method,
            "aresta_vertical",
        )
        save_table(
            output_dir / f"{method}_centros.csv",
            targets.lon_centers,
            targets.lat_centers,
            result.centers,
            method,
            "centro",
        )
        plot_method(
            output_dir / f"{method}.png",
            lons,
            lats,
            lon_obs,
            lat_obs,
            temp_obs,
            targets,
            result,
            method,
        )

    print(f"Arquivos gerados em: {output_dir.resolve()}")
    print("Metodos:", ", ".join(methods))


if __name__ == "__main__":
    main()

