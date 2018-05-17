import pymongo
from pymongo import MongoClient

def set_up_mongodb():
    MONGODB_URI = "mongodb://test:test@ds129043.mlab.com:29043/cryptocurrency_blotter"
    client = MongoClient(MONGODB_URI, connectTimeoutMS = 30000)
    db = client.get_database("cryptocurrency_blotter")
    user_records = db.user_records
    return user_records

def push_record(record):
    user_records = set_up_mongodb()
    user_records.insert_one(record)

def push_blotter_data(blotter):
    blotter_dict = blotter.to_dict('index')
    push_record(blotter_dict)