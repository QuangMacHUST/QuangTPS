"""
Phân đoạn tự động sử dụng phương pháp Atlas-based Segmentation.
"""

import os
import numpy as np
import SimpleITK as sitk
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle

from quangtps.core.config import Config
from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class AtlasSegmentor:
    """Lớp phân đoạn sử dụng Atlas"""
    
    def __init__(self, atlas_dir=None):
        """
        Khởi tạo segmentor.
        
        Parameters:
            atlas_dir (str, optional): Đường dẫn đến thư mục atlas
        """
        # Sử dụng đường dẫn mặc định nếu không chỉ định
        if atlas_dir is None:
            config = Config.get_instance()
            atlas_dir = config.get('atlas_dir', None)
        
        self.atlas_dir = atlas_dir
        self.atlases = {}  # {organ_name: [{'image': img, 'mask': mask}, ...]}
        
        # Tải atlas nếu có
        if self.atlas_dir and os.path.exists(self.atlas_dir):
            self._load_atlases()
    
    def _load_atlases(self):
        """Tải các atlas từ thư mục"""
        try:
            # Lấy danh sách các thư mục cơ quan
            for organ_dir in os.listdir(self.atlas_dir):
                organ_path = os.path.join(self.atlas_dir, organ_dir)
                
                if os.path.isdir(organ_path):
                    # Kiểm tra file index (nếu có)
                    index_file = os.path.join(organ_path, 'index.pkl')
                    if os.path.exists(index_file):
                        # Tải index từ file
                        with open(index_file, 'rb') as f:
                            self.atlases[organ_dir] = pickle.load(f)
                        logger.info(f"Loaded atlas index for {organ_dir}")
                        continue
                    
                    # Nếu không có file index, tải các hình ảnh và mask
                    self.atlases[organ_dir] = []
                    
                    # Tìm các cặp hình ảnh và mask
                    image_files = [f for f in os.listdir(organ_path) if f.endswith('_image.npy')]
                    
                    for img_file in image_files:
                        # Tạo tên file mask tương ứng
                        mask_file = img_file.replace('_image.npy', '_mask.npy')
                        img_path = os.path.join(organ_path, img_file)
                        mask_path = os.path.join(organ_path, mask_file)
                        
                        if os.path.exists(mask_path):
                            # Tải hình ảnh và mask
                            image = np.load(img_path)
                            mask = np.load(mask_path)
                            
                            # Thêm vào atlas
                            self.atlases[organ_dir].append({
                                'image': image,
                                'mask': mask
                            })
                    
                    logger.info(f"Loaded {len(self.atlases[organ_dir])} atlases for {organ_dir}")
            
            logger.info(f"Loaded atlases for {len(self.atlases)} organs")
        
        except Exception as e:
            logger.error(f"Error loading atlases: {str(e)}")
    
    def _find_similar_atlases(self, image, organ, n_similar=3):
        """
        Tìm n atlas giống nhất với hình ảnh đầu vào.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh đầu vào
            organ (str): Tên cơ quan
            n_similar (int, optional): Số lượng atlas giống nhất cần tìm
        
        Returns:
            list: Danh sách các atlas giống nhất
        """
        if organ not in self.atlases or not self.atlases[organ]:
            return []
        
        try:
            # Chuẩn hóa hình ảnh đầu vào
            image_flat = image.flatten().astype(np.float32)
            image_flat = (image_flat - np.mean(image_flat)) / (np.std(image_flat) + 1e-8)
            
            # Tính độ tương đồng với từng atlas
            similarities = []
            
            for i, atlas in enumerate(self.atlases[organ]):
                atlas_image = atlas['image']
                atlas_flat = atlas_image.flatten().astype(np.float32)
                atlas_flat = (atlas_flat - np.mean(atlas_flat)) / (np.std(atlas_flat) + 1e-8)
                
                # Tính cosine similarity
                similarity = cosine_similarity(image_flat.reshape(1, -1), atlas_flat.reshape(1, -1))[0, 0]
                similarities.append((i, similarity))
            
            # Sắp xếp theo độ tương đồng giảm dần
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy n atlas giống nhất
            similar_atlases = [self.atlases[organ][i] for i, _ in similarities[:n_similar]]
            
            return similar_atlases
        
        except Exception as e:
            logger.error(f"Error finding similar atlases: {str(e)}")
            return []
    
    def _register_image(self, moving_image, fixed_image, transform_type='affine'):
        """
        Đăng ký hình ảnh sử dụng SimpleITK.
        
        Parameters:
            moving_image (numpy.ndarray): Hình ảnh cần đăng ký
            fixed_image (numpy.ndarray): Hình ảnh tham chiếu
            transform_type (str, optional): Loại biến đổi ('rigid', 'affine', 'bspline')
        
        Returns:
            tuple: (registered_image, transform)
        """
        try:
            # Chuyển đổi sang SimpleITK Image
            moving_sitk = sitk.GetImageFromArray(moving_image)
            fixed_sitk = sitk.GetImageFromArray(fixed_image)
            
            # Tạo registration framework
            registration_method = sitk.ImageRegistrationMethod()
            
            # Thiết lập metric
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            
            # Thiết lập interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Thiết lập optimizer
            registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Thiết lập biến đổi
            if transform_type == 'rigid':
                transform = sitk.Euler2DTransform()
            elif transform_type == 'affine':
                transform = sitk.AffineTransform(2)
            elif transform_type == 'bspline':
                transform = sitk.BSplineTransformInitializer(fixed_sitk, [10, 10])
            else:
                raise ValueError(f"Unsupported transform type: {transform_type}")
            
            registration_method.SetInitialTransform(transform)
            
            # Thực hiện đăng ký
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            
            # Áp dụng biến đổi vào hình ảnh
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_sitk = resampler.Execute(moving_sitk)
            
            # Chuyển lại thành numpy array
            registered_image = sitk.GetArrayFromImage(registered_sitk)
            
            return registered_image, final_transform
        
        except Exception as e:
            logger.error(f"Error registering image: {str(e)}")
            return moving_image, None
    
    def _transform_mask(self, mask, transform, reference_image):
        """
        Áp dụng biến đổi vào mask.
        
        Parameters:
            mask (numpy.ndarray): Mask cần biến đổi
            transform: Biến đổi từ _register_image
            reference_image (numpy.ndarray): Hình ảnh tham chiếu
        
        Returns:
            numpy.ndarray: Mask sau khi biến đổi
        """
        try:
            # Chuyển đổi sang SimpleITK Image
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            reference_sitk = sitk.GetImageFromArray(reference_image)
            
            # Áp dụng biến đổi vào mask
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_sitk)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Sử dụng nearest neighbor cho mask
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(transform)
            
            transformed_mask_sitk = resampler.Execute(mask_sitk)
            
            # Chuyển lại thành numpy array
            transformed_mask = sitk.GetArrayFromImage(transformed_mask_sitk)
            
            return transformed_mask
        
        except Exception as e:
            logger.error(f"Error transforming mask: {str(e)}")
            return mask
    
    def segment(self, image, organ, n_similar=3, transform_type='affine', voting_threshold=0.5):
        """
        Phân đoạn hình ảnh sử dụng phương pháp atlas.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh đầu vào
            organ (str): Tên cơ quan
            n_similar (int, optional): Số lượng atlas giống nhất cần sử dụng
            transform_type (str, optional): Loại biến đổi ('rigid', 'affine', 'bspline')
            voting_threshold (float, optional): Ngưỡng bỏ phiếu cho kết hợp các mask
        
        Returns:
            numpy.ndarray: Mask sau khi phân đoạn
        """
        try:
            # Kiểm tra cơ quan
            if organ not in self.atlases or not self.atlases[organ]:
                raise ValidationError(f"No atlas found for organ: {organ}")
            
            # Tìm n atlas giống nhất
            similar_atlases = self._find_similar_atlases(image, organ, n_similar)
            
            if not similar_atlases:
                raise ValidationError(f"Could not find similar atlases for {organ}")
            
            # Đăng ký và biến đổi mask
            transformed_masks = []
            
            for atlas in similar_atlases:
                atlas_image = atlas['image']
                atlas_mask = atlas['mask']
                
                # Đăng ký hình ảnh
                registered_image, transform = self._register_image(atlas_image, image, transform_type)
                
                if transform is None:
                    continue
                
                # Biến đổi mask
                transformed_mask = self._transform_mask(atlas_mask, transform, image)
                
                transformed_masks.append(transformed_mask)
            
            if not transformed_masks:
                raise ValidationError(f"Failed to transform masks for {organ}")
            
            # Kết hợp các mask bằng cách bỏ phiếu
            combined_mask = np.zeros_like(image, dtype=np.float32)
            
            for mask in transformed_masks:
                combined_mask += mask
            
            combined_mask /= len(transformed_masks)
            
            # Áp dụng ngưỡng
            final_mask = (combined_mask > voting_threshold).astype(np.uint8)
            
            return final_mask
        
        except Exception as e:
            logger.error(f"Error segmenting using atlas: {str(e)}")
            raise ValidationError(f"Error segmenting using atlas: {str(e)}")
    
    def add_to_atlas(self, image, mask, organ, save_to_disk=True):
        """
        Thêm hình ảnh và mask vào atlas.
        
        Parameters:
            image (numpy.ndarray): Hình ảnh
            mask (numpy.ndarray): Mask
            organ (str): Tên cơ quan
            save_to_disk (bool, optional): Có lưu vào đĩa không
        
        Returns:
            bool: True nếu thêm thành công
        """
        try:
            # Kiểm tra kích thước
            if image.shape != mask.shape:
                raise ValidationError("Image and mask shapes must match")
            
            # Tạo thư mục cơ quan nếu cần
            if organ not in self.atlases:
                self.atlases[organ] = []
                
                if save_to_disk and self.atlas_dir:
                    organ_dir = os.path.join(self.atlas_dir, organ)
                    os.makedirs(organ_dir, exist_ok=True)
            
            # Thêm vào atlas
            self.atlases[organ].append({
                'image': image,
                'mask': mask
            })
            
            # Lưu vào đĩa nếu cần
            if save_to_disk and self.atlas_dir:
                organ_dir = os.path.join(self.atlas_dir, organ)
                idx = len(self.atlases[organ]) - 1
                
                # Tạo tên file
                img_file = os.path.join(organ_dir, f"{idx:04d}_image.npy")
                mask_file = os.path.join(organ_dir, f"{idx:04d}_mask.npy")
                
                # Lưu file
                np.save(img_file, image)
                np.save(mask_file, mask)
                
                # Cập nhật file index
                index_file = os.path.join(organ_dir, 'index.pkl')
                with open(index_file, 'wb') as f:
                    pickle.dump(self.atlases[organ], f)
            
            logger.info(f"Added new atlas for {organ}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding to atlas: {str(e)}")
            return False
    
    def get_available_organs(self):
        """
        Lấy danh sách các cơ quan có sẵn atlas.
        
        Returns:
            list: Danh sách tên các cơ quan
        """
        return list(self.atlases.keys())
