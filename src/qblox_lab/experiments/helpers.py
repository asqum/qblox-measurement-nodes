# helpers.py
import json
import math

def save_strict_json_config(device, filepath):
    """
    Extracts the device config and safely converts all invalid NaN values 
    to strictly valid JSON 'null' values before saving.
    """
    raw_json = device.to_json()
    data_dict = json.loads(raw_json)
    
    def replace_nan_with_none(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: replace_nan_with_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_nan_with_none(v) for v in obj]
        return obj

    clean_dict = replace_nan_with_none(data_dict)
    
    with open(filepath, "w") as f:
        json.dump(clean_dict, f, indent=4, allow_nan=False)
        
    print(f"Strictly valid JSON saved to {filepath}")

def load_strict_json_config(filepath):
    """
    Loads a strictly valid JSON config file and safely converts all 
    'null' (None) values back to math.nan so Pydantic can validate them as floats.
    """
    with open(filepath, "r") as f:
        data_dict = json.load(f)
        
    def replace_none_with_nan(obj):
        if obj is None:
            return math.nan
        elif isinstance(obj, dict):
            return {k: replace_none_with_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_none_with_nan(v) for v in obj]
        return obj

    return replace_none_with_nan(data_dict)

def create_viewable_json(native_filepath, viewable_filepath):
    """
    Reads a Qblox config with NaNs and saves a strict-JSON copy (with nulls) 
    purely for visualization in external editors.
    """
    # 1. Read the native file (Python allows NaNs during load)
    with open(native_filepath, "r") as f:
        data_dict = json.load(f)
        
    # 2. Recursively replace NaN with None (which dumps to null)
    def replace_nan_with_none(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: replace_nan_with_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_nan_with_none(v) for v in obj]
        return obj

    clean_dict = replace_nan_with_none(data_dict)
    
    # 3. Save the pretty, strict JSON copy
    with open(viewable_filepath, "w") as f:
        json.dump(clean_dict, f, indent=4, allow_nan=False)
        
    print(f"Viewable JSON successfully generated at: {viewable_filepath}")
