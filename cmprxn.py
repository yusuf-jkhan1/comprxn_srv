import numpy as np
import random



class k_cluster:

    seed = 1

    def __init__(self, array, k, n_dim = 3, algo_type = "mediods"):
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

 