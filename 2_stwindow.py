import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def space_time_window(distance_threshold_km=10.0, day_threshold=20, flag=""):
    mainshock_df = pd.read_csv(f'mainshock_df_{flag}.csv', index_col=0, parse_dates=["datetime"])
    df = pd.read_csv(f"df_inland_{flag}.csv", index_col=0, parse_dates=["datetime"])

    for _, row in tqdm(mainshock_df.iterrows(), total=len(mainshock_df)):
        main_time = pd.to_datetime(row["datetime"])
        main_lat = row["latitude"]
        main_lon = row["longitude"]
        main_depth = row["depth"]
        mx, my, mz = row["x"], row["y"], row["z"]

        time_mask = (pd.to_datetime(df["datetime"]) >= main_time - timedelta(days=day_threshold)) & \
                    (pd.to_datetime(df["datetime"]) < main_time)

        df_candidate = df[time_mask].copy()
        if df_candidate.empty:
            nearby_event_dfs.append(pd.DataFrame())
            continue

        dx = df_candidate["x"].values - mx
        dy = df_candidate["y"].values - my
        dz = df_candidate["z"].values - mz

        df_candidate["distance"] = np.sqrt(dx*dx + dy*dy + dz*dz)
        df_nearby = df_candidate[df_candidate["distance"] <= distance_threshold_km].copy()
        mainshock_df["foreshock2"] = (len(df_nearby) > 0)
        mainshock_df["foreshock2 count"] = len(df_nearby)
        mainshock_df.to_csv(f"mainshock_df_{flag}.csv", index=False)

space_time_window(flag="old")
space_time_window(flag="new")

# foeeshock occurrence rate
print("oreshock occurrence rate : ")

def occurrence_rate():
    df_old = pd.read_csv("mainshock_df_old.csv")
    df_new = pd.read_csv("mainshock_df_new.csv")
    foreshock_list_old = df_old["foreshock2"]
    foreshock_list_new = df_new["foreshock2"]

    print(f"old : \n{np.sum(foreshock_list_old)}/{len(foreshock_list_old)} = {100*np.sum(foreshock_list_old)/len(foreshock_list_old):.3f}%\n")
    print(f"new : \n{np.sum(foreshock_list_new)}/{len(foreshock_list_new)} = {100*np.sum(foreshock_list_new)/len(foreshock_list_old_new):.3f}%\n")
    print(f"old+new : \n({np.sum(foreshock_list_old)}+{np.sum(foreshock_list_new)})/({len(foreshock_list_old)}+{len(foreshock_list_new)}) = {100*(np.sum(foreshock_list_old)+np.sum(foreshock_list_new))/(len(foreshock_list_old)+len(foreshock_list_new)):.3f}%")

occurrence_rate()
