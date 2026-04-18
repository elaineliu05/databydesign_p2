# flattening original json
import json

# load original file
with open("stock.json") as f:
    raw = json.load(f)

documents = []

for entry in raw["data"]:
    doc = {
        "symbol": raw["symbol"],
        "name": raw["name"],
        "interval": raw["interval"],
        "date": entry["date"],
        "open": float(entry["open"]),
        "high": float(entry["high"]),
        "low": float(entry["low"]),
        "close": float(entry["close"])
    }
    documents.append(doc)

# save flattened version
with open("stock_flat.json", "w") as f:
    json.dump(documents, f, indent=2)

# load to mongo
from pymongo import MongoClient
import json

uri = "mongodb+srv://elaineyliu05_db_user:helloting@cluster0.jqslx4f.mongodb.net/synthea"
client = MongoClient(uri)
db = client["project2"]
collection = db["stock_data"]

with open("stock_flat.json") as f:
    data = json.load(f)

collection.insert_many(data)

print("Data inserted successfully!")
