import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch # 引入 torch 用于二值化操作
import os
import glob
from PIL import Image
from torch.utils.data import Dataset


def center_crop_arr(pil_image, image_size, crop_ratio=1.0):
    """
    将图像缩放并居中裁剪到指定大小
    
    Args:
        pil_image: PIL Image 对象
        image_size: 目标图像大小
        crop_ratio: 裁剪比例，可以是 float 或 tuple(min, max)
    
    Returns:
        PIL Image: 裁剪后的图像
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)

    if isinstance(crop_ratio, float):
        crop_ratio = crop_ratio
    else:
        crop_ratio = np.random.uniform(crop_ratio[0], crop_ratio[1])
    crop_size = int(round(image_size * crop_ratio))
    crop_y = (arr.shape[0] - crop_size) // 2
    crop_x = (arr.shape[1] - crop_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size])


class ImageWithCanny(transforms.Compose):
    """
    Canny 边缘检测 transform，用于基于 Canny 边缘的控制信号生成
    
    Args:
        image_size: 目标图像大小
        low: Canny 边缘检测的低阈值
        high: Canny 边缘检测的高阈值
        is_training: 是否为训练模式，训练模式下会进行数据增强（随机翻转）
    
    Returns:
        tuple: (img_tensor, edge_tensor)
            - img_tensor: 归一化到 [-1, 1] 的图像 tensor
            - edge_tensor: [0, 1] 范围的边缘 tensor
    """
    def __init__(self, image_size, is_training, low=100, high=200):
        self.image_size = image_size
        self.low = low
        self.high = high
        self.is_training = is_training
        
        # 根据模式选择是否添加随机翻转
        transform_list = [transforms.Lambda(lambda pil: center_crop_arr(pil, image_size, crop_ratio=1.0))]
        if is_training:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        super().__init__(transform_list)
    
    def __call__(self, img):
        img_t = super().__call__(img)
        np_img = np.array(img)
        
        if np_img.ndim == 3:
            gray = (0.299 * np_img[..., 0] + 0.587 * np_img[..., 1] + 0.114 * np_img[..., 2]).astype(np.uint8)
        else:
            gray = np_img.astype(np.uint8)
        
        try:
            import cv2
            edges = cv2.Canny(gray, self.low, self.high)
        except Exception:
            # Fallback: simple Sobel magnitude threshold
            gx = np.zeros_like(gray, dtype=np.float32)
            gy = np.zeros_like(gray, dtype=np.float32)
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
            edges = (np.hypot(gx, gy) > 64).astype(np.uint8) * 255

        img_t = transforms.ToTensor()(img)  # [0, 1]
        img_t = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_t)  # [-1, 1]
        
        edge_t = torch.from_numpy(edges).float().unsqueeze(0) / 255.0  # [1, H, W]
        
        return img_t, edge_t


class PairedTransform:
    def __init__(self, image_size, is_training):
        self.image_size = image_size
        self.is_training = is_training
   
    def resize_and_pad(self, pil_image, target_size, is_mask=False):
        """
        将图像或 Mask 缩放，使其长边等于 target_size，然后填充到 target_size x target_size。
        """
        w, h = pil_image.size
       
        # 1. 计算缩放比例 (以长边为基准)
        scale = target_size / max(w, h)
       
        # 2. 计算缩放后的新尺寸
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
       
        # 3. 缩放图像
        # Mask 必须使用 NEAREST，防止引入非 0/1 的像素值
        resample_mode = Image.NEAREST if is_mask else Image.BICUBIC
           
        resized_image = pil_image.resize((new_w, new_h), resample=resample_mode)
       
        # 4. 创建目标尺寸的画布并居中填充
        pad_color = 0
       
        # 计算填充量，使其居中
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
       
        # 使用 TF.pad 进行填充，保持同步
        # 如果是 RGB 图像，填充 0, 否则填充 0
        padded_image = TF.pad(resized_image, padding, fill=pad_color)

        return padded_image

    def __call__(self, img, mask):
        # 1. 确定随机参数（仅在训练模式下进行数据增强）
        do_flip = random.random() > 0.5 if self.is_training else False

        # --- 预处理 Mask：二值化 (新增步骤) ---
        # 1. 将 PIL Mask 转换为 NumPy 数组
        mask_array = np.array(mask, dtype=np.float32)
       
        # 2. 找到 Mask 中的最大值
        max_val = np.max(mask_array)
       
        # 3. 二值化：将所有非零像素都设置为 255，零像素保持 0
        if max_val > 0:
            # 找到所有非零的位置
            mask_array[mask_array > 122] = 255
            mask_array[mask_array <= 122] = 0
            # 转换为 PIL 图像，确保模式是 'L' (8位灰度)
            mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')
        # 如果 max_val == 0，则 Mask 已经是纯黑，无需操作
       
        # 2. 应用变换
       
        # --- Image: 长边缩放 + 居中填充 ---
        img = self.resize_and_pad(img, self.image_size, is_mask=False)
        if do_flip: img = TF.hflip(img)
       
        # ToTensor & Normalize
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [-1, 1]

        # --- Mask: 长边缩放 + 居中填充 ---
        mask = self.resize_and_pad(mask, self.image_size, is_mask=True)
        if do_flip: mask = TF.hflip(mask)
       
        # ToTensor (将 255 变为 1.0)
        mask_t = TF.to_tensor(mask)
       
        return img_t, mask_t


class PairedLayeredDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录，例如 "imagedataall"
            transform (callable): 同步的数据增强 transform
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # 存储 (img_path, mask_path, label_idx)
       
        # 1. 扫描根目录下的所有子文件夹 (即类别 data1, data2...)
        # 使用 sorted 确保 label 的顺序在所有 GPU 上一致
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
       
        # 建立类别名到索引的映射: {'data1': 0, 'data2': 1, ...}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
       
        print(f"Found {len(self.classes)} classes: {self.class_to_idx}")

        # 2. 遍历每个类别文件夹，收集图片和 Mask
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_path = os.path.join(root_dir, class_name)
           
            img_dir = os.path.join(class_path, 'image')
            mask_dir = os.path.join(class_path, 'mask')
           
            # 检查 image 和 mask 文件夹是否存在
            if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
                print(f"[Warning] Skipping {class_name}: 'image' or 'mask' folder missing.")
                continue
           
            # 获取文件并排序，确保一一对应
            # 假设 image 和 mask 的文件名顺序是一致的 (例如 1.jpg 对应 1.png)
            img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
            mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*')))
           
            if len(img_paths) != len(mask_paths):
                print(f"[Warning] Class {class_name} mismatch: {len(img_paths)} images, {len(mask_paths)} masks.")
                # 这里可以选择报错 assert，或者取最小长度截断
                min_len = min(len(img_paths), len(mask_paths))
                img_paths = img_paths[:min_len]
                mask_paths = mask_paths[:min_len]
           
            # 将路径和标签加入列表
            for i in range(len(img_paths)):
                self.samples.append((img_paths[i], mask_paths[i], class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
       
        # 1. 读取图片 (RGB)
        img = Image.open(img_path).convert('RGB')
       
        # 2. 读取 Mask (L)
        mask = Image.open(mask_path).convert('L')
       
        # 3. 同步增强 (img 和 mask 做完全相同的变换)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
       
        # 4. 获取文件名（不包含路径和扩展名）
        filename = os.path.splitext(os.path.basename(img_path))[0]
       
        # 返回 ((img, mask), label, filename)
        # 这里的 label 就是根据 data1/data2 对应的 0, 1, 2...
        return (img, mask), label, filename
