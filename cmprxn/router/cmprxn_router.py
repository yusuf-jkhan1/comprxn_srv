from fastapi import APIRouter
from cmprxn.cmprxn import k_cluster, img_utils
#from starlette.responses import JSONResponse

router = APIRouter()


@router.post('/cmprxn_kmeans')
def compress_image(path):
    img_utils_obj = img_utils(path)
    my_array = img_utils_obj.reshape_img()
    cmprxr_obj = k_cluster(array=my_array,k=3, algo_type="means")
    cmprxr_obj.run()
    group_assignments = cmprxr_obj.group_assignment_vect
    centroids = cmprxr_obj.current_centers_vect
    response_dict = {"Group Assignments": group_assignments, "Centroids": centroids}   
    return response_dict