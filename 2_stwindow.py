import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta

def space_time_window(distance_threshold_km=10.0, day_threshold=20, flag=""):
    mainshock_df = pd.read_csv(f'mainshock_df_{flag}.csv', parse_dates=["datetime"])
    df = pd.read_csv(f"df_inland_{flag}.csv", parse_dates=["datetime"])

    foreshock_flags = []
    foreshock_counts = []

    for _, row in tqdm(mainshock_df.iterrows(), total=len(mainshock_df)):
        main_time = row["datetime"]
        mx, my, mz = row["x"], row["y"], row["z"]

        time_mask = (df["datetime"] >= main_time - timedelta(days=day_threshold)) & \
                    (df["datetime"] < main_time)

        df_candidate = df.loc[time_mask]

        dx = df_candidate["x"].values - mx
        dy = df_candidate["y"].values - my
        dz = df_candidate["z"].values - mz

        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        count = np.sum(dist <= distance_threshold_km)

        foreshock_flags.append(count > 0)
        foreshock_counts.append(count)

    mainshock_df["foreshock2"] = foreshock_flags
    mainshock_df["foreshock2 count"] = foreshock_counts

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
    print(f"new : \n{np.sum(foreshock_list_new)}/{len(foreshock_list_new)} = {100*np.sum(foreshock_list_new)/len(foreshock_list_new):.3f}%\n")
    print(f"old+new : \n({np.sum(foreshock_list_old)}+{np.sum(foreshock_list_new)})/({len(foreshock_list_old)}+{len(foreshock_list_new)}) = {100*(np.sum(foreshock_list_old)+np.sum(foreshock_list_new))/(len(foreshock_list_old)+len(foreshock_list_new)):.3f}%")

occurrence_rate()
