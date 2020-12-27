from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from matplotlib.pyplot import imshow
import random
import os
import datetime
import hashlib
from io import BytesIO
import requests



app = FastAPI()

class image(BaseModel):
    name: str
    path: str

@app.get("/")
def read_root():
    return "Welcome to the cmprxn API :D"


@app.post("/cmprxn/kmeans")
def post_image(image_val: image):

    text = image_val.path
	#url = image_val.path
	response = requests.__dict__
	#img = Image.open(BytesIO(response.content))

    return_obj = {"Image Name": text}
    return return_obj
