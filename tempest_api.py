import json
import os
import requests
import time
from typing import Any, Dict, List

import pandas as pd

TOKEN = os.environ.get("TEMPESTTOKEN")
TEMPEST_URL = "https://swd.weatherflow.com/swd/rest"
HIRES_PATH =  "data/hour_data.csv"
STATION_DATA_PATH = "data/station_metadata.json"
LATEST_OBS_PATH = "data/latest_observation.json"

def get_tempest_station_id() -> str:
    station_meta_url = f"{TEMPEST_URL}/stations"
    resp = requests.get(f"{station_meta_url}?token={TOKEN}")
    station_json = resp.json()
    write_json(station_json, STATION_DATA_PATH)  # TODO: only write if successful
    return station_json["stations"][0]["station_id"]

def write_json(json_obj: Dict[str, Any], json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)

def get_latest_observation(station_id: str) -> Dict[str, Any]:
    observations_url = f"{TEMPEST_URL}/observations/station/{station_id}"
    resp = requests.get(f"{observations_url}?token={TOKEN}")
    print(resp.json())
    return resp.json()


def get_observations(
    station_id: str, freq: int = 60, write: int = 3600
    ) -> pd.DataFrame:
    obs_list: List[Dict[str, Any]] = []
    for i in range(write // freq):
        latest_obs = get_latest_observation(station_id)
        obs_list.extend(latest_obs["obs"])
        time.sleep(freq)
    write_json(latest_obs, LATEST_OBS_PATH)  # TODO: only write if successful
    return pd.DataFrame.from_dict(obs_list)


def write_observations(station_id: str) -> None:
    df = get_observations(station_id)
    df = df.set_index("timestamp")
    df = df.loc[df.index.drop_duplicates()]
    print(df)
    df.to_csv(HIRES_PATH)

    
if __name__ == "__main__":
    station_id = get_tempest_station_id()
    write_observations(station_id)
