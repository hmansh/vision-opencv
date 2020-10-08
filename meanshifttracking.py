import numpy as np
import cv2

class MeanShiftTracker(object):

    # initialization of tracker
    def __init__(self, centroid_x, centroid_y, obj_width, obj_height):

        self._prev_centroid_x = centroid_x
        self._prev_centroid_y = centroid_y

        self._curr_centroid_x = centroid_x
        self._curr_centroid_y = centroid_y

        self._prev_similarity_BC = 0.0
        self._curr_similarity_BC = 0.0

        if(obj_width % 2 == 0):
            obj_width += 1

        if(obj_height % 2 == 0):
            obj_height += 1

        self._prev_width = obj_width
        self._prev_height = obj_height

        self._curr_width = obj_width
        self._curr_height = obj_height

        self._curr_half_width = int((self._curr_width - 1) * 0.5)
        self._curr_half_height = int((self._curr_height - 1) * 0.5)

        # specification for the features
        self._bins_per_channel = 16
        self._bin_size = int(np.floor(256 / self._bins_per_channel))
        self._model_dim = np.power(self._bins_per_channel, 3)

        # The object models
        self._target_model = np.zeros(self._model_dim)
        self._prev_model = np.zeros(self._model_dim)
        self._curr_model = np.zeros(self._model_dim)

        # Array which stores the index to which each color  value will be assigned in the color histogram
        self.combined_index = np.zeros([self._curr_height, self._curr_width])

        self._max_itr = 5

        self.compute_ellipse_kernel()

    def compute_ellipse_kernel(self):
        """ compute the ellipse kernel weights 
        """

        error_code = 0

        half_width = (self._curr_width - 1) * 0.5
        half_height = (self._curr_height - 1) * 0.5

        x_limit = int(np.floor((self._prev_width - 1) * 0.5))

        y_limit = int(np.floor((self._prev_height - 1) * 0.5))

        x_range = np.array([range(-x_limit, x_limit + 1)])
        y_range = np.array([range(-y_limit, y_limit + 1)])
        y_range = np.transpose(y_range)
        x_matrix = np.repeat(x_range, y_limit * 2 + 1, axis=0)
        y_matrix = np.repeat(y_range, x_limit*2 + 1, axis=1)

        x_square = np.multiply(x_matrix, x_matrix)
        y_square = np.multiply(y_matrix, y_matrix)

        x_square = np.divide(x_square, float(half_width * half_width))
        y_square = np.divide(y_square, float(half_height * half_height))

        self._kernel_mask = np.ones(
            [self._curr_height, self._curr_width]) - (y_square + x_square)

        self._kernel_mask[self._kernel_mask < 0] = 0

        print('kerbnel computation complete ')

        return error_code

    def compute_target_model(self, ref_image):

        error_code = 0

        self.compute_object_model(ref_image)

        self._target_model = np.copy(self._curr_model)

        print('Target model computation complete')
        return error_code

    def compute_object_model(self, image):

        self._curr_model = self._curr_model * 0.0

        self.combined_index = self.combined_index * 0

        # converting to a floating point image
        image = image.astype(float)

        half_width = int((self._curr_width - 1) * 0.5)
        half_height = int((self._curr_height - 1) * 0.5)

        # extract the object region from the image IMP the upper bound is not included
        obj_image = image[self._curr_centroid_y - half_height: self._curr_centroid_y + half_height +
                          1,  self._curr_centroid_x - half_width: self._curr_centroid_x + half_width + 1, :]

        index_matrix = np.divide(obj_image, self._bin_size)
        index_matrix = np.floor(index_matrix)
        index_matrix = index_matrix.astype(int)

        b_index, g_index, r_index = cv2.split(index_matrix)

        combined_index = b_index * \
            np.power(self._bins_per_channel, 2) + \
            self._bins_per_channel * g_index + r_index
        combined_index = combined_index.astype(int)

        self.combined_index = combined_index.astype(int)

        print(self._curr_model.shape)
        for i in range(0, self._curr_height):
            for j in range(0, self._curr_width):
                self._curr_model[combined_index[i, j]
                                 ] += self._kernel_mask[i, j]

        # l1 normalize the feature( histogram )
        sum_val = np.sum(self._curr_model)
        self._curr_model = self._curr_model/float(sum_val)

        print('Object model computed ')

    def perform_mean_shift(self, image):

        half_width = (self._curr_width - 1) * 0.5
        half_height = (self._curr_height - 1) * 0.5

        norm_factor = 0.0
        self._curr_x = 0.0
        self._curr_y = 0.0

        itr = 1

        tmp_x = 0.0
        tmp_y = 0.0

        # Initialize to start the iterations from the current frame
        self._curr_centroid_x = self._prev_centroid_x
        self._curr_centroid_y = self._prev_centroid_y

        # Performing mean shift iterations
        for itr in range(0,  self._max_itr):

            print('mean shift iteration %s ', itr)
            print('max target = %s', np.max(self._target_model))
            print(id(self._target_model))
            print(id(self._curr_model))

            print('max_diff = %s', np.max(
                np.fabs(self._target_model - self._curr_model)))
            # compute the object model in the current frame keeping the current postioin as the position from the previous frame
            self.compute_object_model(image)
            print('max_diff = %s', np.max(
                np.fabs(self._target_model - self._curr_model)))
            print('max target = %s', np.max(self._curr_model))

            self.compute_similarity_value()
            self._prev_similarity_BC = self._curr_similarity_BC

            # Avoid divide by zero error
            self._curr_model[self._curr_model == 0] = 0.001

            # weight value computed as teh ratio of the target  ansd the candidate model
            feature_ratio = np.divide(self._target_model, self._curr_model)

            # computing the new position
            for i in range(0, self._curr_height):
                for j in range(0, self._curr_width):

                    tmp_x += (j - half_width) * \
                        feature_ratio[self.combined_index[i, j]]
                    tmp_y += (i - half_height) * \
                        feature_ratio[self.combined_index[i, j]]

                    norm_factor += feature_ratio[self.combined_index[i, j]]

            mean_shift_x = tmp_x / norm_factor
            mean_shift_y = tmp_y / norm_factor

            # computing the new position using mean-shift
            self._curr_centroid_x += np.round(mean_shift_x)
            self._curr_centroid_y += np.round(mean_shift_y)

            self._curr_centroid_x = int(self._curr_centroid_x)
            self._curr_centroid_y = int(self._curr_centroid_y)

            # compute the object model at the new position
            self.compute_object_model(image)

            # compute the similarity of the target and the current model
            self.compute_similarity_value()

            # Performing line search
            while(self._curr_similarity_BC - self._prev_similarity_BC < -0.01):
                # while( self._curr_similarity_BC < self._prev_similarity_BC   ):

                self._curr_centroid_x = int(
                    np.floor((self._curr_centroid_x + self._prev_centroid_x) * 0.5))
                self._curr_centroid_y = int(
                    np.floor((self._curr_centroid_y + self._prev_centroid_y) * 0.5))

                #self._prev_similarity_BC = self._curr_similarity_BC

                # this section of code was written as the round off error prevents the while loop from converging
                # compute the current location object model
                self.compute_object_model(image)
                self.compute_similarity_value()

                diff_x = self._prev_centroid_x - self._curr_centroid_x
                diff_y = self._prev_centroid_y - self._curr_centroid_y

                # euclidean distance between the points obtained in two consecutive iteration
                euc_dist = np.power(diff_x, 2) + np.power(diff_y, 2)

                # Check for convergence
                if(euc_dist <= 2):  # if converged
                    break

            # difference between the centroid values in the current iteration and previous iteration
            diff_x = self._prev_centroid_x - self._curr_centroid_x
            diff_y = self._prev_centroid_y - self._curr_centroid_y

            # euclidean distance between the points obtained in two consecutive iteration
            euc_dist = np.power(diff_x, 2) + np.power(diff_y, 2)

            self._prev_centroid_x = self._curr_centroid_x
            self._prev_centroid_y = self._curr_centroid_y

            # Check for convergence
            if(euc_dist <= 2):  # if converged
                break

            # else: # comntinue for the next iteration

    def compute_similarity_value(self):
        """ compute the similarity value between two distributions using Bhattacharyya similarity
        """

        error_code = 0
        self._curr_similarity_BC = 0.0
        # Bhattacharya similariy between two distributions
        for i in range(self._model_dim):
            if(self._target_model[i] != 0 and self._curr_model[i] != 0):

                #print( 'val 1 = %s a nd val2 = %s ', self._target_model[i], self._curr_model[ i ] )
                self._curr_similarity_BC += np.sqrt(
                    self._target_model[i] * self._curr_model[i])

        #print( 'max_val =%s', np.max( self._target_model ))

        #print( 'sum_val =%s', np.sum( self._target_model ))

        #print( 'max_val =%s', np.max( self._curr_model ))

        #print( 'sum_val =%s', np.sum( self._curr_model ))

        #print( 'sim val = ', self._curr_similarity_BC )

        return error_code
