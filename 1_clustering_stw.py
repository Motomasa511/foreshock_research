import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
from scipy.optimize import fsolve

df_inland = pd.read_csv('df_inland.csv', index_col=0, parse_dates=["time"])
Mc_old = ???
Mc_new = ???

df_old = df_inland[(df_inland['datetime'] < pd.to_datetime('2016-04-01')) & (df_inland["magnitude"] >= Mc_old)]
df_new = df_inland[(df_inland['datetime'] >= pd.to_datetime('2016-04-01')) & & (df_inland["magnitude"] >= Mc_new)]

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
    x, y, z = np.stack(df1["xyz"].values).T
    power_ten_mag = 10.0 ** (-0.5 * b * magnitude)

    n = len(x)

    eta_mins, T_mins, R_mins, i_mins = eta_min_for_j_numba(
        x, y, z, time_year, power_ten_mag, d_f
    )

    valid = eta_mins != 1e100
    i_valid = i_mins[valid]
    j_valid = np.nonzero(valid)[0]
    pairs = np.stack((i_valid, j_valid), axis=1)

    np.savez(f'eta_pair_{flag}.npz', pairs=pairs, eta_mins=eta_mins[valid])
    np.savez(f'TR_{flag}.npz', T_mins=T_mins[valid], R_mins=R_mins[valid])

if os.path.exists("eta_pair_old.npz"):
    print('skip calculating eta')
else:
    eta_min_func(df_old, d_f=1.6, b=1.0, flag='old')
    print("old eta calculated")
    eta_min_func(df_new, d_f=1.6, b=1.0, flag='new')
    print("new eta calculated")

def eta_data_load(flag: str):
    data = np.load(f'eta_pair_{flag}.npz')
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
    plt.savefig(f"eta_dist_fit_{flag}.png", dpi=300, bbox_inches="tight")

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
    
    with open(f'clusters_{flag}.pkl', "wb") as f:
        pickle.dump(clusters, f)

clustering(flag='old', eta_0=eta_0_old)
clustering(flag='new', eta_0=eta_0_new)
print("clustering done")

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

def build_mainshock_and_before(dfs: dict, buffer_km=20):
    all_mainshocks = []
    all_befores = []

    for flag, df_inland in dfs.items():
        with open(f"clusters_{flag}.pkl", "rb") as f:
            clusters = pickle.load(f)

        candidate_mainshocks = []
        candidate_befores = []

        for root, indices in clusters.items():
            cluster_df = df_inland.iloc[indices]
            if cluster_df.empty:
                continue

            max_magnitude_row = cluster_df.loc[cluster_df["magnitude"].idxmax()]
            if max_magnitude_row["magnitude"] < 4.0 or max_magnitude_row["depth"] > 30:
                continue

            candidate_mainshocks.append(max_magnitude_row)
            before_df = cluster_df[cluster_df["datetime"] < max_magnitude_row["datetime"]].copy()
            candidate_befores.append(before_df)

        mainshock_df = pd.DataFrame(candidate_mainshocks).reset_index(drop=True)
        mainshock_df = extract_and_filter_events(mainshock_df, buffer_km=buffer_km).reset_index(drop=True)

        before_list = []
        for i, row in mainshock_df.iterrows():
            for j, cand in enumerate(candidate_mainshocks):
                if row.equals(cand):
                    before_list.append(candidate_befores[j])
                    break

        all_mainshocks.append(mainshock_df)
        all_befores.extend(before_list)

    merged_mainshock_df = pd.concat(all_mainshocks, ignore_index=True)
    merged_mainshock_df["number"] = range(len(merged_mainshock_df))

    merged_mainshock_df.to_csv("mainshock_df.csv", index=False)
    with open("before_list1.pkl", "wb") as f:
        pickle.dump(all_befores, f)

    return merged_mainshock_df, all_befores

mainshock_df, before_list = build_mainshock_and_before({"old": df_old, "new": df_new})

def foreshock_check1():
    mainshock_df = pd.read_csv('mainshock_df.csv', index_col=0, parse_dates=["datetime"])
    with open("before_list1.pkl", "rb") as f:
        before_event_list = pickle.load(f)

    foreshock_list = []
    for i in range(len(mainshock_df)):
        before_df = before_event_list[i]
        foreshock_list.append(len(before_df)>0)
    mainshock_df1 = mainshock_df.copy()
    mainshock_df1['foreshock'] = foreshock_list
    mainshock_df1.to_csv("mainshock_df_1.csv")
    return mainshock_df

foreshock_check1()