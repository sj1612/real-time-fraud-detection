import time
import math
import threading
from collections import defaultdict
from typing import Dict, Any, List

class StatefulFeatureStore:
    def __init__(self):
        # Maps account ID (nameOrig) to list of historical transactions
        # Each transaction: {'timestamp': float, 'latitude': float, 'longitude': float, 'feature_vector': List[float]}
        self.history = defaultdict(list)
        self.lock = threading.Lock()
        
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Great Circle Distance between two lat/lon coordinates in kilometers
        using the Haversine formula.
        """
        if lat1 == lat2 and lon1 == lon2:
            return 0.0
            
        R = 6371.0  # Earth's radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi / 2) ** 2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def enrich(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a raw transaction dictionary with stateful, sliding-window features.
        Modifies a copy of the dictionary and returns it.
        """
        enriched_tx = tx.copy()
        sender = tx.get("nameOrig", "unknown")
        amount = float(tx.get("amount", 0.0))
        old_bal_org = float(tx.get("oldbalanceOrg", 0.0))
        new_bal_org = float(tx.get("newbalanceOrig", 0.0))
        old_bal_dest = float(tx.get("oldbalanceDest", 0.0))
        new_bal_dest = float(tx.get("newbalanceDest", 0.0))
        
        is_transfer = 1.0 if tx.get("type") == "TRANSFER" else 0.0
        is_cash_out = 1.0 if tx.get("type") == "CASH_OUT" else 0.0
        enriched_tx["is_transfer"] = is_transfer
        enriched_tx["is_cash_out"] = is_cash_out
        
        # Geolocation fields (generate mock ones if not present in the dataset)
        lat = float(tx.get("latitude", 40.7128))
        lon = float(tx.get("longitude", -74.0060))
        enriched_tx["latitude"] = lat
        enriched_tx["longitude"] = lon
        
        # Simulating real-time unix epoch if not explicitly in message
        current_time = tx.get("timestamp", time.time())
        enriched_tx["timestamp"] = current_time

        # Static engineered features
        # 1. Balance error orig: should be 0 for CASH_OUT/TRANSFER, negative/positive shows discrepancy
        enriched_tx["balance_error_orig"] = old_bal_org - new_bal_org - amount
        # 2. Balance error dest: discrepancy at receiver
        enriched_tx["balance_error_dest"] = old_bal_dest + amount - new_bal_dest

        with self.lock:
            user_txs = self.history[sender]
            
            # 3. Velocity Features (Last 10 minutes and 1 hour)
            ten_min_ago = current_time - 600.0
            one_hour_ago = current_time - 3600.0
            
            velocity_10m = 0
            velocity_1h = 0
            monetary_sum = 0.0
            
            # Prune ancient history (older than 24 hours to prevent memory leak)
            twenty_four_hours_ago = current_time - 86400.0
            self.history[sender] = [t for t in user_txs if t["timestamp"] > twenty_four_hours_ago]
            user_txs = self.history[sender]

            for past_tx in user_txs:
                if past_tx["timestamp"] > ten_min_ago:
                    velocity_10m += 1
                if past_tx["timestamp"] > one_hour_ago:
                    velocity_1h += 1
                monetary_sum += past_tx["amount"]
            
            # 4. Monetary Deviation (amount relative to running mean of user's past txs)
            if len(user_txs) > 0:
                running_mean = monetary_sum / len(user_txs)
                enriched_tx["monetary_deviation"] = amount / running_mean if running_mean > 0 else 1.0
            else:
                enriched_tx["monetary_deviation"] = 1.0

            # 5. Impossible Travel Geographic check
            impossible_travel = 0
            if len(user_txs) > 0:
                last_tx = user_txs[-1]
                time_delta_sec = current_time - last_tx["timestamp"]
                
                if time_delta_sec > 0:
                    distance_km = self._calculate_distance(lat, lon, last_tx["latitude"], last_tx["longitude"])
                    time_hours = time_delta_sec / 3600.0
                    speed_kmh = distance_km / time_hours
                    
                    # If velocity exceeds 1000 km/h, flag impossible travel
                    if speed_kmh > 1000.0 and distance_km > 5.0:
                        impossible_travel = 1

            enriched_tx["impossible_travel"] = impossible_travel
            enriched_tx["velocity_10m"] = float(velocity_10m)
            enriched_tx["velocity_1h"] = float(velocity_1h)

            # Construct the complete normalized feature vector matching config.py feature list:
            # ALL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
            # RAW_NUMERIC_FEATURES = [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
            # ENGINEERED_FEATURES = [balance_error_orig, balance_error_dest, velocity_10m, velocity_1h, monetary_deviation, impossible_travel]
            feature_vector = [
                amount,
                old_bal_org,
                new_bal_org,
                old_bal_dest,
                new_bal_dest,
                is_transfer,
                is_cash_out,
                enriched_tx["balance_error_orig"],
                enriched_tx["balance_error_dest"],
                enriched_tx["velocity_10m"],
                enriched_tx["velocity_1h"],
                enriched_tx["monetary_deviation"],
                float(impossible_travel)
            ]

            # Record this transaction to the state store
            self.history[sender].append({
                "timestamp": current_time,
                "amount": amount,
                "latitude": lat,
                "longitude": lon,
                "feature_vector": feature_vector
            })
            
        return enriched_tx

    def get_recent_sequence(self, sender: str, sequence_length: int, feature_dim: int) -> List[List[float]]:
        """
        Return the sequence of last N enriched transaction feature vectors for a given sender account.
        This provides sequence modeling input for our LSTM Autoencoder.
        If history is smaller than sequence_length, we use zero-padding at the beginning.
        """
        with self.lock:
            user_txs = self.history[sender]
            recent = user_txs[-sequence_length:]
            
            # Extract only the feature vectors
            seq = [item["feature_vector"] for item in recent]
            
            # If the user's history is shorter than required sequence length, zero-pad the beginning
            if len(seq) < sequence_length:
                padding_size = sequence_length - len(seq)
                padding = [[0.0] * feature_dim] * padding_size
                seq = padding + seq
                
            return seq
            
    def clear(self):
        """Clear all historical state."""
        with self.lock:
            self.history.clear()
