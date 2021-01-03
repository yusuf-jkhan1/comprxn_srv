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
    group_assignments = cmprxr_obj.group_assignment_vect.tolist()
    centroids = cmprxr_obj.current_centers_vect.tolist()
    response_dict = {"Centroids": centroids, "Labels": group_assignments}  
    return response_dict

