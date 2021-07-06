import os
import fnmatch
import copy
from PIL import Image
import tensorflow as tf
import numpy

class DatasetTF(tf.keras.utils.Sequence):
    # max_count = 0 -> no limit, otherwise cap the number of samples to max_count
    # point_cloud_point_count = 0 -> do not resample the mesh, otherwise subsample to this many points
    def __init__(self, device, root_dir, image_width, max_count=0, shuffle=True, file_extension="jpg", center_from_name=False):
        super().__init__()
        self.device = device
        self.target_files = []
        self.image_width = image_width
        self.file_extension = file_extension
        self.extract_center_from_name = center_from_name

        for root, _, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, f'*.{self.file_extension}'):
                if "input" in filename:
                    self.target_files.append(os.path.join(root, filename))

        if shuffle:
            random_indexes = numpy.random.permutation(len(self.target_files))
            self.target_files = [self.target_files[ri] for ri in random_indexes[:max_count if max_count > 0 else len(self.target_files)]]

        if not shuffle and max_count > 0:
            self.target_files = self.target_files[0:max_count]

        self.training_dataset = [None] * len(self.target_files)

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, index):
        index = index % len(self.target_files)
        if self.training_dataset[index]:
          return self.training_dataset[index]

        print(f">{index}<", end="")
        trg_file = self.target_files[index]
        output_file = trg_file.replace('input','output')
        input_img_pil = Image.open(trg_file).convert('RGB')
        input_img = tf.image.resize(numpy.array(input_img_pil), [self.image_width, self.image_width], method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=True)
        output_img_pil = Image.open(output_file)
        output_img = tf.image.resize(numpy.array(output_img_pil), [self.image_width, self.image_width], method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=True)

        self.training_dataset[index] = {
            'identifier': trg_file.split('/')[-1],
            'input_file_path':trg_file,
            'output_file_path':output_file,
            'input_img':input_img,
            'output_img':output_img,
        }

        if self.extract_center_from_name:
            self.training_dataset[index]['center_coordinate'] = [int(trg_file.split('.')[-2].split('_')[-1]), int(trg_file.split('.')[-2].split('_')[-2])]

        return self.training_dataset[index]
