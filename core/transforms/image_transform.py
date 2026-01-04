# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from functools import reduce
from logging import getLogger
from typing import Any, Callable, Tuple

import numpy as np
import mlx.core as mx
from PIL import Image

logger = getLogger()


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def get_image_transform(
    vision_input_type: str = "vanilla",
    image_res: int = 336,
    max_num_tiles: int = 1,
    normalize_img: bool = True,
) -> Tuple[Callable, int]:

    if vision_input_type == "thumb+tile":
        transforms = VariableSizeImageTransform(
            size=image_res,
            max_num_tiles=max_num_tiles,
            normalize_img=normalize_img,
            use_thumbnail="before",
        )
    else:
        transforms = ImageTransform(
            size=image_res,
            normalize_img=normalize_img,
        )

    logger.info(
        f"Initialized transforms with: vision_input_type: '{vision_input_type}' and max_num_tiles: {max_num_tiles}."
    )

    return transforms


class ImageTransform(object):
    """
    Image transform will resize the longer edge to a given size and pad the shorter edge with mean pixel value of the image.
    """

    def __init__(
        self,
        size: int = 336,
        normalize_img: bool = True,
    ) -> None:
        self.size = size
        self._mean = MEAN
        self._std = STD

        logger.info(f"ImageTransform size: {self.size}")

        self.normalize_img = normalize_img

    def __call__(self, image: Image.Image):
        w, h = image.size
        # Resize using PIL
        image = image.resize((self.size, self.size), resample=Image.Resampling.BICUBIC)
        
        # Convert PIL Image to MLX array
        # PIL Image is (H, W, C), convert to (C, H, W) and normalize to [0, 1]
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        # Convert HWC to CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        image = mx.array(img_array)
        
        # Normalize if needed
        if self.normalize_img:
            mean = mx.array(self._mean).reshape(3, 1, 1)
            std = mx.array(self._std).reshape(3, 1, 1)
            image = (image - mean) / std
        
        # Add chunk dim to make it compatible with existing dataloaders
        image = mx.expand_dims(image, axis=0)  # (1, 3, H, W)
        return image, (w, h)

    def _transform_mlx_array(self, image: mx.array):
        # Image shape (C, H, W) or (N, C, H, W)
        if image.ndim == 3:
            c, h, w = image.shape
        else:
            n, c, h, w = image.shape
        
        # Convert MLX array to PIL Image for resizing
        # Assuming image is in [0, 1] range, convert to [0, 255]
        img_array_np = mx.asnumpy(image)
        if img_array_np.max() <= 1.0:
            img_array = (img_array_np * 255.0).astype(np.uint8)
        else:
            img_array = img_array_np.astype(np.uint8)
        
        # Handle different input shapes
        if image.ndim == 3:
            # (C, H, W) -> (H, W, C)
            img_array = np.transpose(img_array, (1, 2, 0))
        else:
            # (N, C, H, W) -> take first image -> (H, W, C)
            img_array = np.transpose(img_array[0], (1, 2, 0))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_array)
        # Resize
        pil_image = pil_image.resize((self.size, self.size), resample=Image.Resampling.BICUBIC)
        
        # Convert back to MLX array
        img_array = np.asarray(pil_image, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = mx.array(img_array)
        
        # Normalize if needed
        if self.normalize_img:
            mean = mx.array(self._mean).reshape(3, 1, 1)
            std = mx.array(self._std).reshape(3, 1, 1)
            image = (image - mean) / std
        
        return image, (w, h)


class VariableSizeImageTransform(object):
    """
    The variable size image transform will resize the image dynamically
    based on the image aspect ratio and the number of image chunks we allow.

    The algorithm will not upsample low-res images to fit a certain aspect
    ratio, because that leads to a significant degradation in image quality.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image chunks, it will find the closest aspect ratio that
    is allowed within 16 image chunks, i.e., 2:5 = 2 horizontal patches and
    5 vertical patches, giving a total of 10 chunks.

    The image will then be resized to products of the base size (default is
    224px because MetaCLIP takes that), so in this case it will  be resized to
    2*224:5*224 = 448:1120, where we maintain the original aspect ratio and
    pad with the mean value for the rest. This approach minimizes the amount
    of padding required for any arbitrary resolution.

    The final output will therefore be of shape (11, 3, 224, 224), where 10
    patches are coming from the resizing and chunking, and the first patch
    is a downsampled version of the image that preserves aspect ratios.
    """

    def __init__(
        self,
        size: int = 336,
        normalize_img: bool = True,
        max_num_tiles: int = 1,
        use_thumbnail: str = "no",
        area_limit: bool = False,
    ) -> None:
        self.size = size
        self._mean = MEAN
        self._std = STD

        logger.info(f"VariableSizeImageTransform size: {self.size}")

        self.normalize_img = normalize_img
        self.area_limit = area_limit
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        if self.use_thumbnail != "no":
            self.thumbnail_transform = ImageTransform(
                size=self.size,
                normalize_img=normalize_img,
            )

    @staticmethod
    def _factors(n: int):
        """Return all factors of a number."""
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )

    def _find_supported_aspect_ratios(self):
        """
        This function computes all the allowed aspect ratios for a fixed
        number of input chunks.

        For example, with `num_tiles=5`, it will return:
        {
            0.2: [(1, 5)],
            5.0: [(5, 1)],
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.3333333333333333: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
        }
        """
        asp_dict = {}
        for chunk_size in range(self.max_num_tiles, 0, -1):
            _factors = sorted(VariableSizeImageTransform._factors(chunk_size))
            _asp_ratios = [(x, chunk_size // x) for x in _factors]
            for ratio in _asp_ratios:
                k = ratio[0] / ratio[1]
                if k not in asp_dict:
                    asp_dict[k] = [ratio]
                else:
                    asp_dict[k].append(ratio)
        return asp_dict

    def _find_closest_aspect_ratio(self, img_width: int, img_height: int) -> Tuple:
        """
        Given an image width, height and target number of chunks
        this function will find the closest supported aspect ratio.
        """
        tgt_ar = img_width / img_height
        asp_dict = self._find_supported_aspect_ratios()
        cl_d, cl_p = 1e23, None
        if tgt_ar >= 1:
            cl_p = min(
                [k for k in asp_dict.keys() if k <= tgt_ar],
                key=lambda x: abs(x - tgt_ar),
            )
            v = asp_dict[cl_p]
            # select width
            widths = [(idx, self.size * vv[0]) for idx, vv in enumerate(v)]
            tgt_idx = max(widths, key=lambda x: x[1])[0]
        else:
            cl_p = min(
                [k for k in asp_dict.keys() if k > tgt_ar],
                key=lambda x: abs(1 / x - 1 / tgt_ar),
            )
            v = asp_dict[cl_p]
            # select height
            heights = [(idx, self.size * vv[1]) for idx, vv in enumerate(v)]
            tgt_idx = max(heights, key=lambda x: x[1])[0]
        out = v[tgt_idx]
        return out

    def _resize(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        # Resize longer edge to given size.
        w, h = image.size
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = target_width
            new_h = math.floor(new_w / scale)
        else:
            # height >= width
            new_h = target_height
            new_w = math.floor(new_h * scale)

        image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        return image

    def _pad(self, image: Image.Image, new_width: int, new_height: int) -> Image.Image:
        mean_per_channel = tuple(
            np.clip(np.array(image).mean(axis=(0, 1)), 0, 255).astype(np.uint8)
        )
        new_im = Image.new(mode="RGB", size=(new_width, new_height), color=(0, 0, 0))  # type: ignore
        new_im.paste(image)
        return new_im

    def _split(self, image: mx.array, ncw: int, nch: int) -> mx.array:
        # Split image into number of required tiles (width x height)
        # image shape: (num_channels, height, width)
        num_channels, height, width = image.shape
        # Reshape to (num_channels, nch, height // nch, ncw, width // ncw)
        image = mx.reshape(image, (num_channels, nch, height // nch, ncw, width // ncw))
        # Permute dimensions to reorder the axes: (nch, ncw, num_channels, height // nch, width // ncw)
        image = mx.transpose(image, (1, 3, 0, 2, 4))
        # Reshape into the desired output shape (ncw * nch, num_channels, height // nch, width // ncw)
        image = mx.reshape(image, (ncw * nch, num_channels, height // nch, width // ncw))
        return image

    def _get_image_height_width(
        self, image_width: int, image_height: int, target_width: int, target_height: int
    ) -> Tuple[int, int]:
        """
        Given image width, height and target width, height for the canvas, return the dimensions of how the image would be resized
        with aspect ratio preservation.
        """
        scale = image_width / image_height

        if scale > 1.0:
            # Width is larger than height

            # Rescaling factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
            rescaling_factor = min(
                target_width / image_width, target_height / image_height
            )

            # Set new width to target width and height to the rescaled height.
            new_w = rescaling_factor * image_width
            new_h = math.floor(new_w / scale)

        else:
            # Height is larger than width

            # Rescaling factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
            rescaling_factor = min(
                target_width / image_width, target_height / image_height
            )

            # Set new height to target height and width to the rescaled width.
            new_h = rescaling_factor * image_height
            new_w = math.floor(new_h * scale)

        return new_w, new_h

    def _fit_image_to_canvas(
        self, img_width: int, img_height: int, area_limit: bool
    ) -> Any:
        """
        Given an image width, height and target number of chunks this function will see if the image
        can be fit into any of the canvases that can be build from arranging the tiles in a grid.
        If the image can be fit onto several canvases, it will return the canvas where the shorter edge
        of the image will be largest.

        If area_limit is set to True, the tie-breaking prefers the canvas where area is less than 2x the original area.
        """
        # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
        optimal_canvas = None
        optimal_image_width_height = None

        scale = img_width / img_height

        # Gather all potential supported image resolutions and iterate through them to find best match
        potential_arrangements = [
            item
            for sublist in self._find_supported_aspect_ratios().values()
            for item in sublist
        ]
        for n_w, n_h in potential_arrangements:
            # Compute the canvas size
            canvas_width, canvas_height = n_w * self.size, n_h * self.size

            # Check if image can fit into the canvas without downsampling
            if canvas_width >= img_width and canvas_height >= img_height:
                # If we did not find a good canvas yet, we will use the current one
                if optimal_canvas is None:
                    # Set optimal canvas and determine the actual image height and width in the canvas with aspect ratio preserving resampling
                    optimal_canvas = (n_w, n_h)
                    optimal_image_width_height = self._get_image_height_width(
                        image_width=img_width,
                        image_height=img_height,
                        target_width=n_w * self.size,
                        target_height=n_h * self.size,
                    )
                else:
                    # If we already found an optimal canvas before, we will check if the shorter edge of the image will be larger than the current optimal canvas.
                    # This means we can potentially upsample the image resolution which is beneficial to performance.
                    image_width_height = self._get_image_height_width(
                        image_width=img_width,
                        image_height=img_height,
                        target_width=n_w * self.size,
                        target_height=n_h * self.size,
                    )
                    if area_limit:
                        # Prioritize aspect ratio, and choose best within area limit when tied.
                        curr_scale = image_width_height[0] / image_width_height[1]
                        optim_scale = (
                            optimal_image_width_height[0]
                            / optimal_image_width_height[1]
                        )
                        if abs(scale - curr_scale) < abs(scale - optim_scale):
                            # 1. optimize aspect ratio
                            optimal_canvas = (n_w, n_h)
                            optimal_image_width_height = image_width_height
                        elif abs(scale - curr_scale) == abs(scale - optim_scale):
                            # 2. optimize area
                            if (
                                image_width_height[0] * image_width_height[1]
                                < 2 * img_width * img_height
                            ):
                                # 2.1 area is less than 2x the original area
                                optimal_canvas = (n_w, n_h)
                                optimal_image_width_height = image_width_height
                    else:
                        # NOTE: L3V dynamic tiling. Prioritize biggest canvas.
                        if (
                            scale < 1.0
                            and (image_width_height[0] >= optimal_image_width_height[0])
                        ) or (
                            scale >= 1.0
                            and (image_width_height[1] >= optimal_image_width_height[1])
                        ):
                            optimal_canvas = (n_w, n_h)
                            optimal_image_width_height = image_width_height
        return optimal_canvas

    def __call__(self, image: Image.Image) -> Tuple[Any, Any]:
        assert isinstance(image, Image.Image), type(image)
        if self.use_thumbnail != "no":
            thumbnail = self.thumbnail_transform(image)[0]

        w, h = image.size
        # Check if the image can be fit to the canvas without downsampling
        ar = self._fit_image_to_canvas(
            img_width=w, img_height=h, area_limit=self.area_limit
        )
        if ar is None:
            # If we did not find a canvas, we have to find the closest aspect ratio and downsample the image
            ar = self._find_closest_aspect_ratio(img_width=w, img_height=h)

        image = image.resize(
            (ar[0] * self.size, ar[1] * self.size),  # (w, h) for PIL
            resample=Image.Resampling.BICUBIC,
        )
        image = self._pad(image, ar[0] * self.size, ar[1] * self.size)
        
        # Convert PIL Image to MLX array
        # PIL Image is (H, W, C), convert to (C, H, W) and normalize to [0, 1]
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        # Convert HWC to CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        image = mx.array(img_array)
        
        # Normalize if needed
        if self.normalize_img:
            mean = mx.array(self._mean).reshape(3, 1, 1)
            std = mx.array(self._std).reshape(3, 1, 1)
            image = (image - mean) / std
        
        image = self._split(image, ar[0], ar[1])  # type: ignore
        if self.use_thumbnail == "before":
            image = mx.concatenate([thumbnail, image], axis=0)
        elif self.use_thumbnail == "after":
            image = mx.concatenate([image, thumbnail], axis=0)
        elif self.use_thumbnail == "both":
            image = mx.concatenate([thumbnail, image, thumbnail], axis=0)

        return image, ar
