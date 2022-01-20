import calendar
import json
import math
from typing import Any, Dict, Type, TypeVar

import numpy as np
import pandas as pd
from tempest_api import HIRES_PATH, LATEST_OBS_PATH, STATION_DATA_PATH

S = TypeVar("S", bound="WeatherFlowStation")


def eto(
    t_max: float,
    t_min: float,
    rh_max: float,
    rh_min: float,
    doy: int,
    utc_off: float,
    tp: float,
    time: float,
    lon: float,
    lat: float,
    uz: float,
    r_s: float,
    p: float,
    alt: float,
    z: float,
    alb: float = 0.23,
    a_s: float = 0.25,
    b_s: float = 0.5,
    surface: str = "grass",
):
    """Hourly FAO-56 and asCE-EWRI Reference Evapotranspiration
    The function calculates hourly or less than hour FAO-56 (grass) and
    asCE-EWRI Reference (alfalfa) Evapotranspiration (ETo ETr).
    The "surface" parameter determines the type of ET.
    The function can also be used to scale hourly or less than hour actual evapotranspiration.

    Parameters
    ----------
    t_max : float
        Maximum temperature [deg C] in the measurement period
    t_min : float
        Minimum temperature [deg C] in the measurement period
    rh_max : float
        Maximum Relative Humidity [%] in the measurement period
    rh_min : float
        Minimum Relative Humidity [%] in the measurement period
    doy : int
        day of year
    utc_off : int
        offset from UTC in minutes
    tp : float
        The length of the calculation period in hour; 1 for hour, 0.5 for 30 minutes
    time : float
        The midpoint of the time of measurement[hour]; for example time is 12.5 for a period between 12:00 and 13:00
    lat : float
        Decimal latitude of the measurement site.
    lon : float
        Decimal longitude of the measurement site
    uz : float
        Wind speed [m/s]
    r_s : float
        Incoming Shortwave Solar radiation [MJ/m2/hr]
    p : float
        Air pressure,[kPa]
    alt : float
        Altitude of the observations, in meters above sea level.
    z : float
        Height, in meters above ground, of the wind measurement. Will be normalized to 2 meters.
    alb : float
        Albedo [-], default 0.23 for cropland / grass.
    a_s : float
        Regression constant intercept: fraction of total radiation reaching the earth on overcast days (n=0). Default 0.25, or calibrated
    b_s : float
        Regression constant slope: fraction of total radiation reaching the earth on overcast days (n=0). Default 0.5, or calibrated
    surface : str
        Reference surface type: `grass` or `alfalfa`, default `grass`.

    Returns
    -------
    TODO: This.

    References
    ----------
    ALLEN, R. g., PEREIRA, L. S., RAES, D., & SMITH, M. 1998.
        Crop Evapotranspiration (guidelines for computing crop water requirements)
        FAO Irrigation and Drainage Paper No. 56: FAO.
    Jackson, R. D., Hatfield, J. L., Reginato, R. J., Idso, S. B., & Pinter Jr, P. J. (1983).
        Estimation of daily evapotranspiration from one time-of-day measurements.
        Agricultural Water Management, 7(1-3), 351-362. doi: http://dx.doi.org/10.1016/0378-3774(83)90095-1
    French, A. N., Fitzgerald, G., Hunsaker, D., Bar_nes, E., Clarke, T., Lesch, S., Pinter, P. (2005).
        Estimating spatially distributed cotton water use from thermal infrared aerial imagery.
        Paper presented at the World Water Congress 2005:  Impacts of global Climate Change - Proceedings of the 2005 World Water and
        Environmental Resources Congress, Reston, Va.
    Colaizzi, P. D., Kusta_s, W. P., Anderson, M. C., Agam, N., Tolk, J. A., Evett, S. R., O'Shaughnessy, S. A. (2012).
        Two-source energy balance model estimates of evapotranspiration using component and composite surface temperatures.
        Advances in Water Resources, 50, 134-151. doi: http://dx.doi.org/10.1016/j.advwatres.2012.06.004
    """
    # Input checks
    night = r_s < 0.03 * tp  # TODO: Adjust to timestep
    if night:
        raise ValueError("Nighty Night!")
    surface = surface.lower()
    if 8 < tp < 24:
        raise ValueError(f"Timestep should be >= 24 (Daily) or <= 8 (Hourly). Got a timestep of {tp} hours.")
    if surface == "grass":
        cn = 37.5 * tp
        if tp <= 8:
            if night:
                cd = 0.96  # surface resistance / aerodynamic resistance / wind speed at 2m [-] [3]
            else:
                cd = 0.24
        else:
            cd = 0.34
    elif surface == "alfalfa":
        cn = 67 * tp
        if tp <= 8:
            if night:
                cd = 1.7
            else:
                cd = 0.25
        else:
            cd = 0.38
    else:
        raise ValueError(f"Unknown surface {surface}. Use either `grass` or `alfalfa`.")
    t_mean = (t_min + t_max) / 2.0  # [9]

    # Psychrometric Constant
    cp = 0.001013  # Specific heat of air at constant pressure [MJ / kg / deg C?]
    ratio = 0.622  # Molecular weight ratio of water vapor/dry air [-]
    vaporization = 2.45  # TODO: Latent heat of vaporization at 20 C [MJ / kg]
    y = (cp * p) / (ratio * vaporization)  # Psychrometric constant [kPa/deg C] [8]

    # Vapor Pressure Deficit (Teton's Formula)
    def sat_vap_press(t: float) -> float:
        """FAO 56 [11]"""
        return 0.61078 * math.exp((17.269 * t) / (t + 237.3))

    def d_sat_vap_press_dt(t: float) -> float:
        """FAO 56 [13]"""
        numer = 4098 * (0.61078 * math.exp((17.269 * t) / (t + 237.3)))
        denom = (t + 237.3) ** 2.0
        return numer / denom
    
    sat_vap_tmax = sat_vap_press(t_max)
    sat_vap_tmin = sat_vap_press(t_min)
    es = (sat_vap_tmax + sat_vap_tmin) / 2.0  # [12]
    slope = d_sat_vap_press_dt(t_mean)
    ea = (sat_vap_tmax * rh_min  + sat_vap_tmin * rh_max) / 200.0  # [17]
    vpd = es - ea  # Vapor Pressure Deficit [kPa]

    # Radiation
    # Extraterrestrial radiation [MJ / m2 / hour] [28]
    lat_rad = (math.pi / 180.0) * lat  # Latitude in radians [22]
    gsc = 0.0820  # Solar constant [MJ / m2 / min]
    # Inverse square relative solar distance
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)  # TODO: check leap year. Check 365 != 365.25?? [-] [23]
    
    declination = 0.409 * math.sin(2.0 * math.pi / 365 * doy - 1.39)  # `decimation` in FAO. [radians] [24]
    ws = math.acos(-1.0 * math.tan(lat_rad) * math.tan(declination))  # sunrise(+)/sunset(-) hour angle [25]
    b = 2.0 * math.pi * (doy - 81) / 364.0  # [3] TODO: Check to see if this has been updated
    sc = 0.1645 * math.sin(2.0 * b) - 0.1255 * math.cos(b) - 0.025 * math.sin(b)  # seasonal correction for solar time [32]
    tz_lon = utc_off / 4
    w = math.pi / 12.0 * ((time + (tz_lon - lon) / 15.0 + sc) - 12.0)  # solar time angle at midpoint [31]
    ws1 = w - math.pi * tp / 24.0  # Initial hour angle of calculation period [29]
    ws2 = w + math.pi * tp / 24.0  # Final hour angle of calculation period [30]
    r_a = 12.0 * 60.0 / math.pi * gsc * dr * (
        (ws2 - ws1) * math.sin(lat_rad) * math.sin(declination) + math.cos(lat_rad) * math.cos(declination) * (math.sin(ws2) - math.sin(ws1))
    ) # / time period
    # r_a = 24.0 * 60.0 / math.pi * gsc * dr * (
    #     ws * math.sin(lat_rad) * math.sin(declination) + math.cos(lat_rad) * math.cos(declination) * (math.sin(ws))
    # ) # / day    
    # Solar (Shortwave) Radiation
    r_so = (0.75 + 0.00002 * alt) * r_a  # clear sky shortwave radiation [MJ / m2 / day] [37]
    # r_so = (a_s + b_s) * r_a  # clear sky shortwave radiation near sea level or a+b calibrated [MJ / m2 / day] [36]
    r_ns = (1 - alb) * r_s  # Net shortwave radiation [MJ / m2 / day] [38]
    stefan = (4.093E-9 / 24 * tp)  # [MJ / m2 / time_period]  Maybe needs to be hourly?
    rel_sw_rad = r_s / r_so
    # TODO: For night: Either fix Rs/Rso to the 3 hour period before sunset (ws-0.79 < w < ws-0.52), or use Rs/Rso = 0.4-0.6 (humid) or 0.7-0.8 (arid)

    # Net longwave radiation [MJ / m2 / day] [39]
    r_nl = stefan * (
        (t_max + 273.16) ** 4.0 + (t_min + 273.16) ** 4.0
    ) / 2.0 * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * rel_sw_rad - 0.35)
    r_n = r_ns - r_nl  # Net Radiation [40]

    # Ground Heat Flux
    g = 0.5 * r_n if night else 0.1 * r_n  # [45, 46]
    
    # Wind Speed Normalization to 2m height.
    u_2m = uz * (4.87 / math.log((67.8 * z) - 5.42))  # [m/s] [47]

    eto_numer = 0.408 * slope * (r_n -g) + y * cn / (t_mean + 273.16) * u_2m * vpd
    eto_denom = slope + y * (1.0 + cd * u_2m)
    return eto_numer / eto_denom


def read_json(json_path: str) -> Dict[str, Any]:
    with open(json_path) as f:
        json_obj = json.load(f)
    return json_obj


class WeatherFlowStation:
    def __init__(
        self,
        timezone: str,
        latitude: float,
        longitude: float,
        altitude: float,
        height: float,
        utc_offset: int,
    ) -> None:
        self.timezone = timezone
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.height = height
        self.utc_offset = utc_offset
        self.df: pd.DataFrame

    @classmethod
    def from_device_status_json(cls: Type[S], json_path: str = STATION_DATA_PATH) -> S:
        station_metadata = read_json(STATION_DATA_PATH)
        station_json = station_metadata["stations"][0]
        tz = station_json["timezone"]
        lat = station_json["latitude"]
        lon = station_json["longitude"]
        alt = station_json["station_meta"]["elevation"]
        # Get the "above ground level [m]" for the "ST" device in the station.
        height = [device["device_meta"]["agl"] for device in station_json["devices"] if device["device_type"] == "ST"][0]
        utc_off = station_json["timezone_offset_minutes"]
        return cls(
            timezone=tz,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            height=height,
            utc_offset=utc_off,
        )

    def load_observations(self, timestep: float = 0.25) -> None:
        timestep_s = timestep * 3600.0
        hires_df = pd.read_csv(HIRES_PATH)
        hires_gb = hires_df.groupby(
            pd.cut(
                hires_df["timestamp"],
                np.arange(hires_df["timestamp"].iloc[0], hires_df["timestamp"].iloc[-1] + timestep_s, timestep_s),
            ),
        )

        minmax_cols = ["air_temperature", "relative_humidity"]
        mean_cols = ["timestamp", "wind_avg", "station_pressure", "solar_radiation"]
        mean_df = hires_gb[mean_cols].mean()
        minmax_df = pd.merge(hires_gb[minmax_cols].min(), hires_gb[minmax_cols].max(), left_index=True, right_index=True, suffixes=["_min", "_max"])
        self.df = pd.concat([mean_df, minmax_df], axis=1)
        self.df["datetime"] = (
            pd.to_datetime(self.df["timestamp"], unit="s", utc=True)
            .dt.tz_convert(self.timezone)
            .dt.tz_localize(None)
        )
        self.df["doy"] = self.df["datetime"].dt.dayofyear
        self.df["time"] = self.df["datetime"].dt.hour + self.df["datetime"].dt.minute / 60.0

    def calc_eto_series(self, timestep: float = 0.25) -> None:
        self.df["eto"] = self.df.apply(
            lambda row: eto(
                t_max=row["air_temperature_max"],
                t_min=row["air_temperature_min"],
                rh_max=row["relative_humidity_max"],
                rh_min=row["relative_humidity_min"],
                doy=row["doy"],
                utc_off=self.utc_offset,
                tp=timestep,
                time=row["time"],
                lon=self.longitude,
                lat=self.latitude,
                uz=row["wind_avg"],
                r_s=row["solar_radiation"] * 0.0036 * timestep,  # W/m2 == J/m2/s => MJ/m2/timestep
                p=row["station_pressure"] / 10.0,  # mb => kPa
                alt=self.altitude,
                z=self.height,
                alb=0.23,
                a_s=0.25,
                b_s=0.5,
                surface="grass",
            ),
            axis=1,
        )
station = WeatherFlowStation.from_device_status_json()
timestep = 5.0 / 60.0  # [hours]
station.load_observations(timestep)
station.calc_eto_series(timestep)
print(station.df["eto"].describe())
print(station.df["eto"].sum())
