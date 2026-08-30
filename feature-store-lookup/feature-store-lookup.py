def feature_store_lookup(feature_store: dict, requests: list, defaults: dict) -> list:
    """
    Returns a list of feature dictionaries.
    """
    combined = []
    for req in requests:
        offline = feature_store.get(req["user_id"], defaults)
        combined.append({**offline, **req["online_features"]})
    return combined

        