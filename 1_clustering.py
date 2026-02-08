import os
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import pickle
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
from scipy.optimize import fsolve

def japan_map():
    shapefile_path = "ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
    japan_admin1 = gpd.read_file(shapefile_path)
    japan = japan_admin1[japan_admin1["admin"] == "Japan"]
    return japan

japan = japan_map()

df_old = pd.read_csv('df_inland_old.csv', index_col=0, parse_dates=["datetime"])
df_new = pd.read_csv('df_inland_new.csv', index_col=0, parse_dates=["datetime"])

# Nearest-Neighborhood Clustering
# calculate η_ij
@njit(parallel=True)
def eta_min_for_j_numba(x, y, z, time_year, power_ten_mag, d_f):
    n = len(x)
    eta_mins = np.full(n, 1e100)
    T_mins = np.full(n, 1e100)
    R_mins = np.full(n, 1e100)
    i_mins = np.full(n, -1)

    for j in prange(n):
        xj, yj, zj, tj, pm_j = x[j], y[j], z[j], time_year[j], power_ten_mag[j]

        for i in range(j):
            pm_i = power_ten_mag[i]
            dt = tj - time_year[i]

            if (dt >= 1 / pm_i) or (dt <= 0.0):
                continue

            dx = xj - x[i]
            dy = yj - y[i]
            dz = zj - z[i]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            if (dist == 0.0) or (dist > (1e4 / pm_i) ** (1 / d_f)):
                continue

            R_ij = (dist ** d_f) * pm_i
            T_ij = dt * pm_i
            eta_ij = R_ij * T_ij

            if eta_ij < eta_mins[j]:
                eta_mins[j] = eta_ij
                T_mins[j] = T_ij
                R_mins[j] = R_ij
                i_mins[j] = i

    return eta_mins, T_mins, R_mins, i_mins

def eta_min_func(df1, d_f=1.6, b=1.0, flag=''):
    magnitude = df1['magnitude'].values.astype(np.float64)
    time_year = pd.to_datetime(df1['datetime']).values.astype(np.int64) / (1e9 * 365.25 * 24 * 3600)
    x = df1["x"].values
    y = df1["y"].values
    z = df1["z"].values
    power_ten_mag = 10.0 ** (-0.5 * b * magnitude)

    n = len(x)

    eta_mins, T_mins, R_mins, i_mins = eta_min_for_j_numba(
        x, y, z, time_year, power_ten_mag, d_f
    )

    valid = eta_mins != 1e100
    i_valid = i_mins[valid]
    j_valid = np.nonzero(valid)[0]
    pairs = np.stack((i_valid, j_valid), axis=1)

    np.savez(f'1_clustering/eta_pair_{flag}.npz', pairs=pairs, eta_mins=eta_mins[valid])
    np.savez(f'1_clustering/TR_{flag}.npz', T_mins=T_mins[valid], R_mins=R_mins[valid])

def TR_distribbution(flag=""):
    data = np.load(f"1_clustering/TR_{flag}.npz")
    T, R = data['T_mins'], data['R_mins']
    log_T = np.log10(T)
    log_R = np.log10(R)

    plt.figure()
    plt.hist2d(log_T, log_R, bins=80, cmap='jet')
    plt.xlabel('Log(T)', fontsize=12)
    plt.ylabel('Log(R)', fontsize=12)
    plt.title(f"re-scaled T & R ({flag})")
    plt.colorbar(label='Count')

    plt.ylim(-5,5)
    plt.xlim(-10,0)
    plt.savefig(f"1_clustering/TR_{flag}.png")

if os.path.exists("1_clustering/eta_pair_old.npz"):
    print('skip calculating eta')
else:
    print("start calculating eta")
    eta_min_func(df_old, d_f=1.6, b=1.0, flag='old')
    TR_distribbution(flag="old")
    print("old eta calculated")
    eta_min_func(df_new, d_f=1.6, b=1.0, flag='new')
    TR_distribbution(flag="flag")
    print("new eta calculated")

def eta_data_load(flag: str):
    data = np.load(f'1_clustering/eta_pair_{flag}.npz')
    return data['pairs'], data['eta_mins']

# double Weibull distribution
def shifted_double_weibull(x, a, k1, lambda1, k2, lambda2, shift):
    x_shifted = x + shift
    pdf1 = weibull_min.pdf(x_shifted, k1, scale=lambda1)
    pdf2 = weibull_min.pdf(x_shifted, k2, scale=lambda2)
    return a * pdf1 + (1 - a) * pdf2

def shifted_weibull_component(x, k, lambda_, shift):
    x_shifted = x + shift
    return weibull_min.pdf(x_shifted, k, scale=lambda_)

def fit_find_eta0(flag: str):
    pairs, eta = eta_data_load(flag)
    log_eta = np.log10(eta)

    hist, bin_edges = np.histogram(log_eta, bins=1000, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    initial_shift = abs(min(log_eta)) + 1.0
    initial_guess = [0.5, 1.5, 5.0, 2.5, 10.0, initial_shift]
    lower_bounds = [0.0, 0.01, 1e-3, 0.01, 1e-3, 0.1]
    upper_bounds = [1.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0]

    popt, _ = curve_fit(
        shifted_double_weibull, bin_centers, hist,
        p0=initial_guess, bounds=(lower_bounds, upper_bounds)
    )
    a, k1, lambda1, k2, lambda2, shift = popt
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)

    fit_total = shifted_double_weibull(x_fit, *popt)
    fit_comp1 = a * shifted_weibull_component(x_fit, k1, lambda1, shift)
    fit_comp2 = (1 - a) * shifted_weibull_component(x_fit, k2, lambda2, shift)

    def intersection_shifted(x):
        return a * shifted_weibull_component(x, k1, lambda1, shift) - \
               (1 - a) * shifted_weibull_component(x, k2, lambda2, shift)
    x_crossing = fsolve(intersection_shifted, x0=-5.0)[0]
    eta_0 = 10**x_crossing

    plt.figure(figsize=(8,6))
    plt.hist(log_eta, bins=1000, density=True, alpha=0.5, label="Data")
    plt.plot(x_fit, fit_total, 'r-', label="Total Fit", linewidth=2)
    plt.plot(x_fit, fit_comp1, 'b--', label="Component 1", linewidth=1.5)
    plt.plot(x_fit, fit_comp2, 'g--', label="Component 2", linewidth=1.5)
    plt.xlabel('log(η)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.text(x_crossing, 0.3, f'log(η_0)={x_crossing:.3f}')
    plt.savefig(f"1_clustering/eta_dist_fit_{flag}.png", dpi=300, bbox_inches="tight")

    return eta_0
eta_0_old = fit_find_eta0(flag='old')
eta_0_new = fit_find_eta0(flag='new')

# Union-Find (Disjoint Set Union)
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

# clustering
def cluster_pairs(pairs, n):
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    clusters = {}
    for i in range(n):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    return clusters

def clustering(flag: str, eta_0: float):
    pairs, eta = eta_data_load(flag)

    filtered_pairs = [pair for i, pair in enumerate(pairs) if eta[i] < eta_0]

    n = np.max(pairs) + 1
    clusters = cluster_pairs(filtered_pairs, n)
    
    with open(f'1_clustering/clusters_{flag}.pkl', "wb") as f:
        pickle.dump(clusters, f)

clustering(flag='old', eta_0=eta_0_old)
clustering(flag='new', eta_0=eta_0_new)
print("clustering done")


# export mainshocks & preceding events
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
    plt.savefig(f"1_clustering/clustering_area{buffer_km}.png", dpi=300, bbox_inches="tight")

    # event filtering
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    event_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    buffer_geom = buffered_area.iloc[0].geometry
    filtered_gdf = event_gdf[event_gdf.geometry.within(buffer_geom)].copy()
    result_df = filtered_gdf.drop(columns="geometry").reset_index(drop=True)
    return result_df

def build_mainshock_and_before(flag: str, df, buffer_km=20):
    with open(f"1_clustering/clusters_{flag}.pkl", "rb") as f:
        clusters = pickle.load(f)

    candidate_mainshocks = []
    candidate_befores = []

    for root, indices in clusters.items():
        cluster_df = df.iloc[indices]
        if cluster_df.empty:
            continue

        max_row = cluster_df.loc[cluster_df["magnitude"].idxmax()]
        if max_row["magnitude"] < 4.0 or max_row["depth"] > 30:
            continue

        candidate_mainshocks.append(max_row)
        before_df = cluster_df[cluster_df["datetime"] < max_row["datetime"]].copy()
        candidate_befores.append(before_df)

    mainshock_df = pd.DataFrame(candidate_mainshocks).reset_index(drop=True)
    mainshock_df = extract_and_filter_events(mainshock_df, buffer_km=buffer_km).reset_index(drop=True)

    before_list = []
    for i, row in mainshock_df.iterrows():
        for j, cand in enumerate(candidate_mainshocks):
            if row.equals(cand):
                before_list.append(candidate_befores[j])
                break

    mainshock_df["number"] = range(len(mainshock_df))

    with open(f"1_clustering/before_list1_{flag}.pkl", "wb") as f:
        pickle.dump(before_list, f)
    
    foreshock_list = [len(df) > 0 for df in before_list]

    mainshock_df['foreshock1'] = foreshock_list
    mainshock_df.to_csv(f"mainshock_df_{flag}.csv", index=False)

    return mainshock_df, before_list

mainshock_df_old, before_old = build_mainshock_and_before("old", df_old)
mainshock_df_new, before_new = build_mainshock_and_before("new", df_new)

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

plot_earthquakes_on_japan_map(mainshock_df_old, s=5, figname="1_clustering/mainshock_old.png")
plot_earthquakes_on_japan_map(mainshock_df_new, s=5, figname="1_clustering/mainshock_new.png")

# foeeshock occurrence rate
print("-------------\nforeshock occurrence rate : ")

def occurrence_rate():
    df_old = pd.read_csv("mainshock_df_old.csv")
    df_new = pd.read_csv("mainshock_df_new.csv")
    foreshock_list_old = df_old["foreshock1"]
    foreshock_list_new = df_new["foreshock1"]

    print(f"old : \n{np.sum(foreshock_list_old)}/{len(foreshock_list_old)} = {100*np.sum(foreshock_list_old)/len(foreshock_list_old):.3f}%\n")
    print(f"new : \n{np.sum(foreshock_list_new)}/{len(foreshock_list_new)} = {100*np.sum(foreshock_list_new)/len(foreshock_list_new):.3f}%\n")
    print(f"old+new : \n({np.sum(foreshock_list_old)}+{np.sum(foreshock_list_new)})/({len(foreshock_list_old)}+{len(foreshock_list_new)}) = {100*(np.sum(foreshock_list_old)+np.sum(foreshock_list_new))/(len(foreshock_list_old)+len(foreshock_list_new)):.3f}%")

occurrence_rate()