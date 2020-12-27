from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

items_db = []

class sku_item(BaseModel):
	sku: int
	name: str
	maker: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/items")
def read_all_items():
    return items_db

@app.post("/items")
def create_item(item: sku_item):
	items_db.append(item.dict())
	return items_db[-1]
