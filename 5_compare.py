import pandas as pd
import subprocess
from pathlib import Path
import numpy as np
import geopandas as gpd

def velocity_assign(df, work_dir="."):

    work_dir = Path(work_dir)
    param_path = work_dir / "5_veldata/param"
    outfile_path = work_dir / "5_veldata/outfile"

    with open(param_path, "w") as f:
        f.write(f"{len(df)}\n")
        for _, r in df.iterrows():
            lon = r["longitude"]
            lat = r["latitude"]
            depth = max(r["depth"], 0.1)
            f.write(f"{lon:.2f} {lat:.2f} {depth:.2f}\n")

    subprocess.run(["./finterpolate.exe"], cwd="5_veldata")

    result = pd.read_csv(
        outfile_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "depth",
               "vp", "vs", "perror", "serror"]
    )
    df[["vp", "vs", "perror", "serror"]] = result[["vp", "vs", "perror", "serror"]]

    return df

mainshock_df_old = pd.read_csv(f'mainshock_df_old.csv', parse_dates=["datetime"])
mainshock_df_new = pd.read_csv(f'mainshock_df_new.csv', parse_dates=["datetime"])

mainshock_df_old = velocity_assign(mainshock_df_old)
mainshock_df_new = velocity_assign(mainshock_df_new)


def assign_stress_Aphi(df, max_deg_diff=0.2):
    cols = [
        "lon","lat","dep","time",
        "Shmax","Shmax_std","Shmax_azm","Shmax_csd",
        "Shmin","Shmin_std","Shmin_azm","Shmin_csd",
        "S1","S1_trend","S1_plunge",
        "S2","S2_trend","S2_plunge",
        "S3","S3_trend","S3_plunge",
        "See","Sen","Seu","Snn","Snu","Suu",
        "sratio","sratio_std",
        "A_phi","A_phi_std",
        "nev"
    ]

    stress = pd.read_csv(
        "5_stressdata/stress_0.2deg_Japan_Uchide2022_corr_Aug2023.txt",
        sep=r"\s+",
        comment="#",
        header=None,
        names=cols
    )

    stress_pts = stress[["lon","lat","A_phi"]].dropna()

    Aphi_values = []

    for lon, lat in zip(df["longitude"], df["latitude"]):

        candidates = stress_pts[
            (np.abs(stress_pts["lon"] - lon) <= max_deg_diff) &
            (np.abs(stress_pts["lat"] - lat) <= max_deg_diff)
        ]

        if len(candidates) == 0:
            Aphi_values.append(np.nan)
            continue

        d2 = (candidates["lon"] - lon)**2 + (candidates["lat"] - lat)**2

        nearest_idx = d2.idxmin()
        Aphi_values.append(candidates.loc[nearest_idx, "A_phi"])

    df_out = df.copy()
    df_out["A_phi"] = Aphi_values

    return df_out

mainshock_df_old = assign_stress_Aphi(mainshock_df_old)
mainshock_df_new = assign_stress_Aphi(mainshock_df_new)

def assign_polygon_value(foreshock_df):
    gdf_temp = gpd.read_file("GSJ_DB_GRES-DB_ONSEN_2020_Temperature-polygon/Temperature-polygon.shp")

    gdf_points = gpd.GeoDataFrame(
        foreshock_df.copy(),
        geometry=gpd.points_from_xy(foreshock_df["longitude"], foreshock_df["latitude"]),
        crs="EPSG:4326" 
    )

    gdf_polygons = gpd.GeoDataFrame(
        gdf_temp.copy(),
        geometry=gdf_temp["geometry"]
    ).set_crs("EPSG:4326", allow_override=True)

    joined = gpd.sjoin(gdf_points, gdf_polygons[["PolygonVal", "geometry"]], how="left", predicate="within")

    result = foreshock_df.copy()
    result["onsen_heat"] = joined["PolygonVal"].values

    return result

mainshock_df_old = assign_polygon_value(mainshock_df_old)
mainshock_df_new = assign_polygon_value(mainshock_df_new)


mainshock_df_old.to_csv(f"mainshock_df_old.csv", index=False)
mainshock_df_new.to_csv(f"mainshock_df_new.csv", index=False)
