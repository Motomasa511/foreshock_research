import zipfile
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
import pickle
from math import radians, sin, cos
import time

os.makedirs("0_datafig", exist_ok=True)

# shape of Japan
def japan_map():
    shapefile_path = "ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
    japan_admin1 = gpd.read_file(shapefile_path)
    japan = japan_admin1[japan_admin1["admin"] == "Japan"]
    return japan

japan = japan_map()

print("load catalog")

# load JMA catalog
def parse_line(line):
    try:
        dt = datetime.strptime(line[1:15], "%Y%m%d%H%M%S")

        lat = float(line[22:24]) + float(line[24:28]) / (100*60)
        lon = float(line[33:36]) + float(line[36:40]) / (100*60)
        depth = float(line[44:49].replace(" ", "0")) / 100
        mag = float(line[52:54]) / 10
        typ = int(line[60])   # type 1=normal eq, 5=lfe
        return dt, lat, lon, depth, mag, typ
    except:
        return None

def load_jma_catalog(zip_dir="JMAcatalog"):
    records = []
    for fname in sorted(os.listdir(zip_dir)):
        if not fname.endswith(".zip"):
            continue

        print("Reading", fname)
        with zipfile.ZipFile(os.path.join(zip_dir, fname)) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    for raw in f:
                        line = raw.decode("utf-8", errors="ignore").rstrip()
                        if not line.startswith("J"):
                            continue
                        p = parse_line(line)
                        if p:
                            records.append(p)

    df = pd.DataFrame(records, columns=["datetime","latitude","longitude","depth","magnitude","type"])
    return df

# add cartesian coordinate
def add_xyz(df, lat_col="latitude", lon_col="longitude", depth_col="depth"):
    EARTH_RADIUS = 6371.0

    R = EARTH_RADIUS - df[depth_col].values
    lat = np.radians(df[lat_col].values)
    lon = np.radians(df[lon_col].values)

    df["x"] = R * np.cos(lat) * np.cos(lon)
    df["y"] = R * np.cos(lat) * np.sin(lon)
    df["z"] = R * np.sin(lat)

    return df


df_all = load_jma_catalog()

df_normal = df_all[df_all['type'] == 1]
df_normal = add_xyz(df_normal.copy())
df_lfe = df_all[df_all['type'] == 5]
df_lfe.to_csv('df_lfe.csv')
print('df_lfe exported')

# extract wider inland area (for clutering)
def extract_and_filter_events(df, buffer_km, area_threshold_km2=1500):

    def remove_holes(geometry):
        if isinstance(geometry, Polygon):
            return Polygon(geometry.exterior)
        elif isinstance(geometry, MultiPolygon):
            return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms])
        else:
            return geometry

    # extract inland
    gdf = japan.to_crs(epsg=3099)
    polygons = gdf.explode(ignore_index=True)
    polygons["area_km2"] = polygons["geometry"].area / 1e6
    main_polygons = polygons[polygons["area_km2"] >= area_threshold_km2].copy()
    main_polygons["geometry"] = main_polygons["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )
    mainland = main_polygons.union_all()

    # remove holes
    buffer_m = buffer_km * 1000
    buffered = mainland.buffer(buffer_m)
    buffered_no_holes = remove_holes(buffered)
    buffered_area = gpd.GeoDataFrame(geometry=[buffered_no_holes], crs=gdf.crs).to_crs(epsg=4326)

    # visualize area
    fig, ax = plt.subplots(figsize=(8, 10))
    japan.to_crs(epsg=4326).plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
    buffered_area.plot(ax=ax, color='skyblue', alpha=0.5, edgecolor='red', linewidth=1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(127, 147)
    ax.set_ylim(30, 47)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"0_datafig/clustering_area{buffer_km}.png", dpi=300, bbox_inches="tight")

    # event filtering
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    event_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    buffer_geom = buffered_area.iloc[0].geometry
    filtered_gdf = event_gdf[event_gdf.geometry.within(buffer_geom)].copy()
    result_df = filtered_gdf.drop(columns="geometry").reset_index(drop=True)
    return result_df

df_inland = extract_and_filter_events(df_normal[df_normal['depth'] <= 40], buffer_km=50)

print('clustering_area.png exported')

# plot all eq on map
def plot_earthquakes_on_japan_map(df, s, figname):

    gdf_eq = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )

    fig, ax = plt.subplots(figsize=(10, 12))

    japan.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.6)
    gdf_eq.plot(ax=ax, markersize=s, color="red", alpha=0.6)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(127, 147)
    plt.ylim(30, 47)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figname, dpi=300, bbox_inches="tight")

plot_earthquakes_on_japan_map(df_inland, s=0.1, figname="0_datafig/df_inland_events.png")

print('clustering_events.png exported')


# determin Mc
def maxc_analysis(df, mag_col="magnitude", bin_width=0.1, flag=""):

    mags = df[mag_col].dropna().values

    mmin = np.floor(mags.min()) - 0.05
    mmax = np.ceil(mags.max())
    bins = np.arange(mmin, mmax + bin_width, bin_width)

    counts, edges = np.histogram(mags, bins=bins)
    centers = edges[:-1] + bin_width / 2

    cumulative_counts = np.cumsum(counts[::-1])[::-1]

    # Mc estimated by MAXC
    idx_max = np.argmax(counts)
    mc = centers[idx_max]

    plt.figure()
    plt.bar(centers, counts,
        width=bin_width,
        align='center',
        alpha=0.6, label="Non-Cumulative")
    plt.scatter(centers, cumulative_counts, label="Cumulative")
    plt.axvline(mc, color='pink', linestyle='--', linewidth=2,
                label=f"Mc (MAXC) = {mc:.2f}")
    plt.axvline(mc+0.2, color='r', linestyle='--', linewidth=2,
                label=f"Mc (MAXC+0.2) = {mc+0.2:.2f}")
    plt.xlabel("Magnitude")
    plt.ylabel("Event Count")
    plt.yscale("log")
    plt.title(f"Frequency-Magnitude Distribution ({flag})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"0_datafig/gr_hist_{flag}.png")

    return mc

Mc_old = maxc_analysis(df_inland[(df_inland['datetime'] < pd.to_datetime('2016-04-01'))], flag="old")
Mc_new = maxc_analysis(df_inland[(df_inland['datetime'] >= pd.to_datetime('2016-04-01'))], flag="new")

print(f"Mc_old : {Mc_old:.2f}+0.2\nMc_new : {Mc_new:.2f}+0.2")

df_inland[(df_inland['datetime'] < pd.to_datetime('2016-04-01')) & (df_inland["magnitude"] >= Mc_old+0.2)].to_csv('df_inland_old.csv')
df_inland[(df_inland['datetime'] >= pd.to_datetime('2016-04-01')) & (df_inland["magnitude"] >= Mc_new+0.2)].to_csv('df_inland_new.csv')