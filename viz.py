!pip install pymongo
from pymongo import MongoClient
import json
import pandas as pd
import matplotlib.pyplot as plt

# connect to mongo
uri = "mongodb+srv://elaineyliu05_db_user:helloting@cluster0.jqslx4f.mongodb.net/synthea"
client = MongoClient(uri)
db = client["project2"]
collection = db["stock_data"]

# fetch data
data = list(collection.find({}, {"_id": 0}))

# convert to DataFrame
df = pd.DataFrame(data)

# preprocess
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# plot
plt.figure()
plt.plot(df['date'], df['close'])
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("S&P 500 Closing Prices Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
