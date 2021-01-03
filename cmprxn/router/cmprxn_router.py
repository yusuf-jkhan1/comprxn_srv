from fastapi import APIRouter
from cmprxn.cmprxn import k_cluster, img_utils
from starlette.responses import JSONResponse

router = APIRouter()



@router.post('/cmprxn_kmeans')
def compress_image(path: str):
    img_utils_obj = img_utils(path)
    my_array = img_utils_obj.reshape_img()
    cmprxr_obj = k_cluster(array=my_array,k=2, algo_type="means")
    cmprxr_obj.run()
    #group_assignments = cmprxr_obj.group_assignment_vect[0:5]
    #centroids = cmprxr_obj.current_centers_vect[0:5]
    #array_head = JSONResponse(my_array[0:5])
    my_array_str = cmprxr_obj.group_assignment_vect.tolist()
    response_dict = {"Dim": ['a','b','c'], "Head": my_array_str}  
    return response_dict

