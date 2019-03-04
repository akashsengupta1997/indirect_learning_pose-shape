import numpy as np
from keras.preprocessing.image import DirectoryIterator


class ImagesWithFnames(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super(ImagesWithFnames, self).__init__(*args, **kwargs)
        self.filenames_np = np.array(self.filenames)

    def _get_batches_of_transformed_samples(self, index_array):
        return (super(ImagesWithFnames, self)._get_batches_of_transformed_samples(index_array),
                self.filenames_np[index_array])