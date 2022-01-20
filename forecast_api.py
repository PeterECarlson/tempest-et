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
station_id = 64977
conditions_url = f"{TEMPEST_URL}/better_forecast"
resp = requests.get(f"{conditions_url}?station_id={station_id}&token={TOKEN}")
print(resp.json())
