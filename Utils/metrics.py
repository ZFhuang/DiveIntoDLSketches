# from: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py

import logging
import torch
import numpy as np
import cv2

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

def img_metric(target_path, label_path):
    img_target=cv2.imread(target_path).astype(np.float32)
    img_label=cv2.imread(label_path).astype(np.float32)
    img_label=cv2.resize(img_label, (img_target.shape[1],img_target.shape[0]), interpolation=cv2.INTER_LINEAR)
    logging.info(target_path+' SSIM is: '+str(SSIM._ssim(img_target,img_label)))
    img_target = torch.from_numpy(img_target.transpose((2, 0, 1)))
    img_label = torch.from_numpy(img_label.transpose((2, 0, 1)))
    logging.info(target_path+' PSNR is: '+str((PSNR.__call__(img_target,img_label)).item()))