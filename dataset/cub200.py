""" CUB200 Pytorch Dataset """
""" Adapted from ContinualAI https://avalanche.continualai.org/"""

"""
CUB200 Pytorch Dataset: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an
extended version of the CUB-200 dataset, with roughly double the number of
images per class and new part location annotations. For detailed information
about the dataset, please check the official website:
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
"""

import os
import csv
from pathlib import Path
from typing import Union

import gdown
import os
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive
from torch import Tensor
from torchvision.transforms.functional import crop


class CUB200(Dataset):
    """Basic CUB200 PathsDataset to be used as a standard PyTorch Dataset.
    A classic continual learning benchmark built on top of this dataset
    can be found in 'benchmarks.classic', while for more custom benchmark
    design please use the 'benchmarks.generators'."""

    images_folder = "CUB_200_2011/images"
    official_url = (
        "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/"
        "CUB_200_2011.tgz"
    )
    gdrive_url = (
        "https://drive.google.com/u/0/uc?id="
        "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
    )
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root=None,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True
    ):
        """
        :param root: root dir where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'CUB_200_2011' will be used.
        :param train: train or test subset of the original dataset. Default
            to True.
        :param transform: eventual input data transformations to apply.
            Default to None.
        :param target_transform: eventual target data transformations to apply.
            Default to None.
        :param loader: method to load the data from disk. Default to
            torchvision default_loader.
        :param download: default set to True. If the data is already
            downloaded it will skip the download.
        """

        self.root = root        
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.download = download
        self.verbose = True

        self._load_dataset()

        self.targets = [img_data[1] for img_data in self._images]

    def _load_dataset(self) -> None:
        """
        The standardized dataset download and load procedure.

        For more details on the coded procedure see the class documentation.

        This method shouldn't be overridden.

        This method will raise and error if the dataset couldn't be loaded
        or downloaded.

        :return: None
        """
        metadata_loaded = False
        metadata_load_error = None
        try:
            metadata_loaded = self._load_metadata()
        except Exception as e:
            metadata_load_error = e

        if metadata_loaded:
            if self.verbose:
                print("Files already downloaded and verified")
            return

        if not self.download:
            msg = (
                "Error loading dataset metadata (dataset download was "
                'not attempted as "download" is set to False)'
            )
            if metadata_load_error is None:
                raise RuntimeError(msg)
            else:
                print(msg)
                raise metadata_load_error

        try:
            self._download_dataset()
        except Exception as e:
            err_msg = self._download_error_message()
            print(err_msg, flush=True)
            raise e

        if not self._load_metadata():
            err_msg = self._download_error_message()
            print(err_msg)
            raise RuntimeError(
                "Error loading dataset metadata (... but the download "
                "procedure completed successfully)"
            )

    def _extract_archive(
        self,
        path: Union[str, Path],
        sub_directory: str = None,
        remove_archive: bool = False,
    ) -> Path:
        """
        Utility method that can be used to extract an archive.

        :param path: The complete path to the archive (for instance obtained
            by calling `_download_file`).
        :param sub_directory: The name of the sub directory where to extract the
            archive. Can be None, which means that the archive will be extracted
            in the root. Beware that some archives already have a root directory
            inside of them, in which case it's probably better to use None here.
            Defaults to None.
        :param remove_archive: If True, the archive will be deleted after a
            successful extraction. Defaults to False.
        :return:
        """

        if sub_directory is None:
            extract_root = self.root
        else:
            extract_root = os.path.join(self.root, sub_directory)

        extract_archive(
            str(path), to_path=str(extract_root), remove_finished=remove_archive
        )

        return extract_root

    def _download_dataset(self) -> None:
        try:
            self._download_and_extract_archive(
                CUB200.official_url, CUB200.filename, checksum=CUB200.tgz_md5
            )
        except Exception:
            if self.verbose:
                print(
                    "[CUB200] Direct download may no longer be possible, "
                    "will try GDrive."
                )

        filepath = os.path.join(self.root, self.filename)
        gdown.download(self.gdrive_url, str(filepath), quiet=False)
        gdown.cached_download(self.gdrive_url, str(filepath), md5=self.tgz_md5)

        self._extract_archive(filepath)

    def _download_error_message(self) -> str:
        return (
            "[CUB200] Error downloading the dataset. Consider downloading "
            "it manually at: " + CUB200.official_url + " and placing it "
            "in: " + str(self.root)
        )

    def _load_metadata(self):
        """Main method to load the CUB200 metadata"""

        cub_dir = os.path.join(self.root, "CUB_200_2011")
        self._images = OrderedDict()

        with open(os.path.join(cub_dir, "train_test_split.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                is_train_instance = int(row[1]) == 1
                if is_train_instance == self.train:
                    self._images[img_id] = []

        with open(os.path.join(cub_dir, "images.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                if img_id in self._images:
                    self._images[img_id].append(os.path.join(self.images_folder, row[1]))

        with open(os.path.join(cub_dir, "image_class_labels.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                if img_id in self._images:
                    # CUB starts counting classes from 1 ...
                    self._images[img_id].append(int(row[1]) - 1)

        with open(os.path.join(cub_dir, "bounding_boxes.txt")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            for row in csv_reader:
                img_id = int(row[0])
                if img_id in self._images:
                    box_cub = [int(float(x)) for x in row[1:]]
                    box_avl = [box_cub[1], box_cub[0], box_cub[3], box_cub[2]]
                    # PathsDataset accepts (top, left, height, width)
                    self._images[img_id].append(box_avl)

        images_tuples = []
        for _, img_tuple in self._images.items():
            images_tuples.append(tuple(img_tuple))
        self._images = images_tuples

        # Integrity check
        for row in self._images:
            filepath = os.path.join(self.root, row[0])
            if not os.path.isfile(filepath):
                if self.verbose:
                    print("[CUB200] Error checking integrity of:", filepath)
                return False

        return True

    def __getitem__(self, index):
        """
        Returns next element in the dataset given the current index.
        :param index: index of the data to get.
        :return: loaded item.
        """

        img_description = self._images[index]
        impath = img_description[0]
        target = img_description[1]
        bbox = None
        if len(img_description) > 2:
            bbox = img_description[2]

        if self.root is not None:
            impath = os.path.join(self.root, impath)
        img = self.loader(impath)

        # If a bounding box is provided, crop the image before passing it to
        # any user-defined transformation.
        if bbox is not None:
            if isinstance(bbox, Tensor):
                bbox = bbox.tolist()
            img = crop(img, *bbox)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns the total number of elements in the dataset.
        :return: Total number of dataset items.
        """

        return len(self._images)


__all__ = ["CUB200"]