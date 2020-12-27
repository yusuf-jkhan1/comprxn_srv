from PIL import Image
import uvicorn
import cmprxn
import utils
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()


cmprxr_obj = {}

class image(BaseModel):
    name: str
    path: str

@app.get("/")
def read_root():
    return "Welcome to the cmprxn API :D"


@app.post("/cmprxn/kmeans")
def post_image(image: image):

    text = image.name
    img_utils_obj = utils.img_utils(image.path)
    my_array = img_utils_obj.reshape_img()
    dims = str(my_array.shape)
    cmprxr_obj = cmprxn.k_cluster(array=my_array,k=3, algo_type="means")
    cmprxr_obj.run()
    print(cmprxr_obj.group_assignment_vect)
    return_obj = {"Image Name": text,
                  "Image Dimensions": dims,
                  "Group Assignment_vect": cmprxr_obj.group_assignment_vect.tolist()[0:10]}
    return return_obj

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)