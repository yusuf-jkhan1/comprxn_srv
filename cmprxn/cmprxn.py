import numpy as np
import random
from PIL import Image
from matplotlib.pyplot import imshow
from urllib.request import urlretrieve
import os
import datetime
import hashlib



class k_cluster:

    seed = 1

    def __init__(self, array, k, n_dim = 3, algo_type = "means"):
        """ """ 
        self.array = array
        self.k = k
        self.n_dim = n_dim
        self.algo_type = algo_type
        self.initial_centers_vect = self._setup(array=array, k=k, n_dim=n_dim)

    @classmethod
    def set_seed(cls, seed):
        cls.seed = seed
        

    def _setup(self, array, n_dim, k):
        """ """ 
        
        #Check if input array matches expectation
        assert array.shape[1] == n_dim
        #Check for appropriate selection of k
        assert k < array.shape[0], "Error: More clusters than observations"

        #For debugging and reproducibility
        random.seed(self.seed)

        #Initialize k Cluster Center Indexes by sample from range of array length without replacement
        c_vect_inds = random.sample(range(array.shape[0]),k)
        #Retrieve 'center' vectors
        c_vect = array[c_vect_inds]
        
        return c_vect

    def _allocation(self, array, c_vect):
        """ """ 

        #Create holder for group assignment for every pixel
        group_assignment_vect = np.zeros(len(array))

        #For each pixel find the index of the 'center' that represents the minimum euclidean distance
        for pixel_vector in enumerate(array):
            
            #temp_array to hold distances of pixel_vector to each 'center'
            temp_array = np.zeros(len(c_vect))

            for center in enumerate(c_vect):
                temp_array[center[0]] = (np.linalg.norm(pixel_vector[1] - center[1]))**2
            #Get the index of the minimum value which will represent cluster membership
            group_assignment_vect[pixel_vector[0]] = np.argmin(temp_array)
        return group_assignment_vect

    def _calibrate(self, array,group_assignment_vect,k, algo_type = 'mediods'):
        """ """ 
        
        assert len(group_assignment_vect) == len(array), "Length group assignment vect doesn't match array length"
        
        centers_vect = []
        #For every group, find the average of the pixels
        for group in enumerate(range(k)):
            

            #Create boolean mask for group
            group_vect_ind = group_assignment_vect == group[1]

            
            #Find the centroid of the group
            centroid = np.mean(array[group_vect_ind], axis = 0)

            if algo_type == 'means':
                centers_vect.append(centroid)
            elif algo_type == 'mediods':
                #Find the point in the group closest to the centroid
                temp_array = np.zeros(len(array[group_vect_ind]))

                for pixel_vector in enumerate(array[group_vect_ind]):

                    temp_array[pixel_vector[0]] = np.linalg.norm(pixel_vector[1] - centroid)

                #Index of vector closest to theoretical centroid            
                ind = np.argmin(temp_array)
                #New 'center' of group
                centers_vect.append(array[group_vect_ind][ind])
            else:
                print("\n Invalid algo_type argument passed. Can only be 'means' or 'mediods' \n")

        centers_vect = np.array(centers_vect)

        return centers_vect

    def _step(self, current_centers_vect):
        """ """    
        self.group_assignment_vect = self._allocation(array=self.array,c_vect=current_centers_vect)
        self.new_centers_vect = self._calibrate(array=self.array, 
                                                group_assignment_vect=self.group_assignment_vect,
                                                k=self.k,
                                                algo_type=self.algo_type)
        return self.new_centers_vect


    def run(self):

        current_centers_vect = self._step(current_centers_vect=self.initial_centers_vect)
        
        check_equality = lambda x,y: (x==y).all()
        
        are_equal = False

        step_i = 0
        while are_equal == False:
            new_centers_vect = np.zeros(self.k,dtype="int64")
            new_centers_vect = self._step(current_centers_vect=current_centers_vect)
            are_equal = check_equality(current_centers_vect, new_centers_vect)
            current_centers_vect = new_centers_vect.copy()
            step_i += 1
            #print(f"Step: {step_i}")

        self.num_steps = step_i
        self.current_centers_vect = current_centers_vect
        

class img_utils:

    def __init__(self, path):
        """
        Read in image bitmap and convert to array
        
        """
        self.path = path
        try:
            self.arr = self._read_image_local(self.path)
        except:
            self.arr = self._read_image_from_url()

    def _read_image_local(self, path):
        """
        Read image and store it as an array

        """
        if not os.path.exists('data'):
            os.makedirs('data')

        Image.open(path).save("data/img.bmp")
        img = Image.open("data/img.bmp")
        img_arr = np.array(img, dtype='int32')
        img.close()
        return img_arr

    def _read_image_from_url(self):
        """
        Read image from url and store it as an array

        """

        if not os.path.exists('data'):
            os.makedirs('data')

        fname = self._create_hashed_filename()

        urlretrieve(url= self.path, filename= fname)
        
        img_arr = self._read_image_local(path = fname)

        return img_arr

    def _create_hashed_filename(self):
        """
        Create a filename by hashing current time as string

        """

        time_str = str(datetime.datetime.now()).encode()
        hashed_obj = hashlib.md5(time_str)
        hashed_str = hashed_obj.hexdigest()
        ftype = self.path.split(".")[-1]
        fname = "data/" + hashed_str + "." + ftype

        return fname

    def display_image(self, img_array=None):
        """
        Display image

        """
        if img_array is None:
            self.arr = self.arr.astype(dtype='uint8')
            img = Image.fromarray(self.arr, 'RGB')
            imshow(np.asarray(img))
        else:
            arr = img_array.astype(dtype="uint8")
            arr = Image.fromarray(arr, 'RGB')
            imshow(np.asarray(arr))

    def reshape_img(self):
        """
        Takes in array generated by read_image function and converts to long format

        """
        long_array = np.reshape(self.arr, (self.arr.shape[0] * self.arr.shape[1], self.arr.shape[2]))
        self.long_array = long_array
        self.status = "Here I am"
        return long_array

    def rebuild_compressed_image(self, centroids, labels):
        """Takes k_cluster object and displays the compressed image"""
        
        centers = centroids
        class_assignments = labels
        original_array = self.arr

        class_assignment_list = [x for x in np.unique(class_assignments)]

        centroids_dict = {}
        for center,class_ in zip(centers,class_assignment_list):
            centroids_dict[class_] = center

        long_array = self.long_array
        compressed_array = np.array([[0,0,0]]*len(long_array))
        for ind, array_i in enumerate(zip(long_array, class_assignments)):
            x = array_i[1]
            compressed_array[ind] = centroids_dict[x]

        prime_shape = (original_array.shape[0],original_array.shape[1],original_array.shape[2])
        compressed_array_prime = np.reshape(compressed_array, prime_shape)

        self.display_image(compressed_array_prime)