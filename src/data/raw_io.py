import pandas as pd
from pathlib import Path
from typing import Union

def read_gps_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read raw GPS data from parquet or csv.
    Expected columns: vehicle_id, timestamp, lat, lon
    """
    file_path = Path(file_path)
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    # Basic validation
    required = ['vehicle_id', 'timestamp', 'lat', 'lon']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
            
    return df

def read_road_network(file_path: Union[str, Path]):
    """
    Placeholder for reading road network (OSM pbf or shapefile).
    """
    raise NotImplementedError("Network reading not yet implemented. Use map-matching library.")
