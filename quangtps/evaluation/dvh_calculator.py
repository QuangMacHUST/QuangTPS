import numpy as np
import logging

logger = logging.getLogger(__name__)

class DVHCalculator:
    def calculate_dvh(self, structure_masks, dose_grid, dose_grid_spacing=None, num_bins=100, 
                      dose_max=None, relative=False, volume_type='relative'):
        """
        Tính toán Đồ thị Thể tích Liều (DVH) cho các cấu trúc đã cho.
        
        Parameters
        ----------
        structure_masks : dict
            Dictionary chứa mặt nạ nhị phân cho mỗi cấu trúc.
            Key là tên cấu trúc, value là mặt nạ 3D.
        dose_grid : numpy.ndarray
            Mảng 3D chứa dữ liệu liều.
        dose_grid_spacing : tuple, optional
            Khoảng cách giữa các voxel theo (x, y, z) tính theo mm.
            Mặc định là None, tức là sử dụng khoảng cách 1 mm theo mỗi chiều.
        num_bins : int, optional
            Số lượng bins sử dụng cho histogram, mặc định là 100.
        dose_max : float, optional
            Giá trị liều tối đa để tính toán DVH, mặc định là None.
            Nếu None, sẽ sử dụng giá trị lớn nhất trong dose_grid.
        relative : bool, optional
            Nếu True, các giá trị liều sẽ được chuẩn hóa theo % của liều tối đa.
            Mặc định là False.
        volume_type : str, optional
            Loại thể tích ('relative' hoặc 'absolute'). Mặc định là 'relative'.
            'relative': Thể tích được biểu diễn dưới dạng % của tổng thể tích.
            'absolute': Thể tích được biểu diễn dưới dạng cm³.
            
        Returns
        -------
        dict
            Dictionary chứa dữ liệu DVH cho mỗi cấu trúc.
            Mỗi key là tên cấu trúc, value là một dictionary với các keys:
            - 'dose_bins': Mảng chứa các giá trị liều.
            - 'volume_bins': Mảng chứa các giá trị thể tích tương ứng.
            - 'stats': Dictionary chứa các thông số thống kê của DVH.
        """
        if dose_grid_spacing is None:
            dose_grid_spacing = (1.0, 1.0, 1.0)
        
        # Tính thể tích của một voxel (mm³ -> cm³)
        voxel_volume = dose_grid_spacing[0] * dose_grid_spacing[1] * dose_grid_spacing[2] / 1000.0
        
        # Xác định giá trị liều tối đa
        if dose_max is None:
            dose_max = np.max(dose_grid)
        
        # Tạo bins liều
        dose_bins = np.linspace(0, dose_max, num_bins + 1)
        bin_centers = 0.5 * (dose_bins[1:] + dose_bins[:-1])
        
        result = {}
        
        for structure_name, structure_mask in structure_masks.items():
            # Lấy giá trị liều trong cấu trúc
            structure_dose = dose_grid[structure_mask]
            
            if len(structure_dose) == 0:
                logger.warning(f"Không có voxel nào trong cấu trúc {structure_name}")
                continue
            
            # Tính tổng thể tích
            total_volume = np.sum(structure_mask) * voxel_volume
            
            # Tính histogram
            hist, _ = np.histogram(structure_dose, bins=dose_bins)
            
            # Tính DVH tích lũy (cumulative DVH)
            cum_hist = np.cumsum(hist[::-1])[::-1]
            
            # Chuyển đổi sang thể tích tương đối hoặc tuyệt đối
            if volume_type == 'relative':
                volume_bins = cum_hist / cum_hist[0] * 100.0 if cum_hist[0] > 0 else np.zeros_like(cum_hist)
            else:  # 'absolute'
                volume_bins = cum_hist * voxel_volume
            
            # Chuyển đổi liều sang tương đối nếu cần
            if relative:
                bin_centers_output = bin_centers / dose_max * 100.0
            else:
                bin_centers_output = bin_centers
            
            # Tính các thông số thống kê
            stats = self._calculate_dvh_statistics(structure_dose, bin_centers, volume_bins, total_volume, relative, dose_max)
            
            result[structure_name] = {
                'dose_bins': bin_centers_output,
                'volume_bins': volume_bins,
                'total_volume': total_volume,
                'stats': stats
            }
        
        return result
    
    def _calculate_dvh_statistics(self, structure_dose, dose_bins, volume_bins, total_volume, relative, dose_max):
        """
        Tính toán các thông số thống kê từ DVH.
        
        Parameters
        ----------
        structure_dose : numpy.ndarray
            Mảng 1D chứa giá trị liều trong cấu trúc.
        dose_bins : numpy.ndarray
            Mảng 1D chứa các giá trị liều trung tâm của bins.
        volume_bins : numpy.ndarray
            Mảng 1D chứa các giá trị thể tích tương ứng với mỗi bin liều.
        total_volume : float
            Tổng thể tích của cấu trúc (cm³).
        relative : bool
            Nếu True, các giá trị liều được chuẩn hóa theo % của liều tối đa.
        dose_max : float
            Giá trị liều tối đa.
            
        Returns
        -------
        dict
            Dictionary chứa các thông số thống kê DVH.
        """
        stats = {}
        
        # Liều nhỏ nhất, lớn nhất, trung bình
        min_dose = np.min(structure_dose)
        max_dose = np.max(structure_dose)
        mean_dose = np.mean(structure_dose)
        median_dose = np.median(structure_dose)
        
        # Chuẩn hóa liều nếu cần
        if relative:
            min_dose = min_dose / dose_max * 100.0
            max_dose = max_dose / dose_max * 100.0
            mean_dose = mean_dose / dose_max * 100.0
            median_dose = median_dose / dose_max * 100.0
        
        stats['min_dose'] = min_dose
        stats['max_dose'] = max_dose
        stats['mean_dose'] = mean_dose
        stats['median_dose'] = median_dose
        
        # Độ lệch chuẩn
        stats['std_dose'] = np.std(structure_dose)
        if relative:
            stats['std_dose'] = stats['std_dose'] / dose_max * 100.0
        
        # Chỉ số đồng nhất (Homogeneity Index) - HI = (D2% - D98%) / D50%
        # Trong đó D2%, D98%, D50% là liều nhận bởi 2%, 98%, và 50% thể tích
        try:
            d2, d98, d50 = self.get_dose_at_volume_percent(dose_bins, volume_bins, [2, 98, 50])
            if d50 > 0:
                stats['homogeneity_index'] = (d2 - d98) / d50
            else:
                stats['homogeneity_index'] = float('nan')
        except:
            stats['homogeneity_index'] = float('nan')
        
        # Chỉ số phù hợp (Conformity Index) - CI = (V95% / TV) * (V95% / V_irr)
        # Trong đó V95% là thể tích nhận ít nhất 95% liều, TV là thể tích mục tiêu
        # và V_irr là tổng thể tích nhận ít nhất 95% liều
        # Lưu ý: Tính chỉ số này cần thêm thông tin về thể tích tham chiếu và liều tổng
        # Hiện tại chỉ tính V95%
        try:
            v95 = self.get_volume_at_dose_percent(dose_bins, volume_bins, 95)
            stats['V95'] = v95
        except:
            stats['V95'] = float('nan')
        
        # Các chỉ số lâm sàng phổ biến
        # V5, V10, V20, V30, V40, V50 (% thể tích nhận ít nhất X Gy)
        for dose_level in [5, 10, 20, 30, 40, 50]:
            try:
                dose_value = dose_level
                if relative:
                    dose_value = dose_level / 100.0 * dose_max
                vol_at_dose = self.get_volume_at_dose(dose_bins, volume_bins, dose_value)
                stats[f'V{dose_level}'] = vol_at_dose
            except:
                stats[f'V{dose_level}'] = float('nan')
        
        # D90, D95, D98, D99 (liều nhận bởi X% thể tích)
        for vol_percent in [90, 95, 98, 99]:
            try:
                dose_at_vol = self.get_dose_at_volume_percent(dose_bins, volume_bins, vol_percent)
                stats[f'D{vol_percent}'] = dose_at_vol[0] if isinstance(dose_at_vol, list) else dose_at_vol
            except:
                stats[f'D{vol_percent}'] = float('nan')
        
        return stats
    
    def get_volume_at_dose(self, dose_bins, volume_bins, dose_value):
        """
        Lấy thể tích nhận ít nhất một giá trị liều nhất định.
        
        Parameters
        ----------
        dose_bins : numpy.ndarray
            Mảng 1D chứa các giá trị liều trung tâm của bins.
        volume_bins : numpy.ndarray
            Mảng 1D chứa các giá trị thể tích tương ứng với mỗi bin liều.
        dose_value : float
            Giá trị liều cần truy vấn.
            
        Returns
        -------
        float
            Thể tích nhận ít nhất giá trị liều đã cho.
        """
        if dose_value > np.max(dose_bins):
            return 0.0
        
        if dose_value <= np.min(dose_bins):
            return volume_bins[0]
        
        # Nội suy tuyến tính
        idx = np.searchsorted(dose_bins, dose_value)
        if idx >= len(dose_bins):
            return 0.0
        
        if idx == 0:
            return volume_bins[0]
        
        x0, x1 = dose_bins[idx-1], dose_bins[idx]
        y0, y1 = volume_bins[idx-1], volume_bins[idx]
        
        # Nội suy tuyến tính
        if x1 == x0:
            return y0
        
        return y0 + (y1 - y0) * (dose_value - x0) / (x1 - x0)
    
    def get_volume_at_dose_percent(self, dose_bins, volume_bins, dose_percent):
        """
        Lấy thể tích nhận ít nhất một phần trăm liều tối đa.
        
        Parameters
        ----------
        dose_bins : numpy.ndarray
            Mảng 1D chứa các giá trị liều trung tâm của bins.
        volume_bins : numpy.ndarray
            Mảng 1D chứa các giá trị thể tích tương ứng với mỗi bin liều.
        dose_percent : float
            Phần trăm liều tối đa cần truy vấn (0-100).
            
        Returns
        -------
        float
            Thể tích nhận ít nhất phần trăm liều đã cho.
        """
        max_dose = np.max(dose_bins)
        dose_value = dose_percent / 100.0 * max_dose
        return self.get_volume_at_dose(dose_bins, volume_bins, dose_value)
    
    def get_dose_at_volume_percent(self, dose_bins, volume_bins, volume_percent):
        """
        Lấy giá trị liều nhận bởi một phần trăm thể tích nhất định.
        
        Parameters
        ----------
        dose_bins : numpy.ndarray
            Mảng 1D chứa các giá trị liều trung tâm của bins.
        volume_bins : numpy.ndarray
            Mảng 1D chứa các giá trị thể tích tương ứng với mỗi bin liều.
        volume_percent : float hoặc list
            Phần trăm thể tích cần truy vấn (0-100).
            Có thể là một giá trị hoặc danh sách các giá trị.
            
        Returns
        -------
        float hoặc list
            Giá trị liều nhận bởi phần trăm thể tích đã cho.
            Nếu đầu vào là list, kết quả cũng là list.
        """
        if isinstance(volume_percent, list):
            return [self._get_dose_at_single_volume_percent(dose_bins, volume_bins, vp) for vp in volume_percent]
        else:
            return self._get_dose_at_single_volume_percent(dose_bins, volume_bins, volume_percent)
    
    def _get_dose_at_single_volume_percent(self, dose_bins, volume_bins, volume_percent):
        """
        Lấy giá trị liều nhận bởi một phần trăm thể tích nhất định (hàm helper).
        
        Parameters
        ----------
        dose_bins : numpy.ndarray
            Mảng 1D chứa các giá trị liều trung tâm của bins.
        volume_bins : numpy.ndarray
            Mảng 1D chứa các giá trị thể tích tương ứng với mỗi bin liều.
        volume_percent : float
            Phần trăm thể tích cần truy vấn (0-100).
            
        Returns
        -------
        float
            Giá trị liều nhận bởi phần trăm thể tích đã cho.
        """
        if volume_percent < 0 or volume_percent > 100:
            raise ValueError("volume_percent phải nằm trong khoảng 0-100")
        
        volume_target = volume_bins[0] * (1 - volume_percent / 100.0)
        
        # DVH là giảm dần, nên volume_target có thể bé hơn giá trị thể tích nhỏ nhất
        if volume_target <= volume_bins[-1]:
            return dose_bins[-1]
        
        # Nếu volume_target lớn hơn giá trị thể tích lớn nhất
        if volume_target >= volume_bins[0]:
            return dose_bins[0]
        
        # Tìm vị trí trong mảng volume_bins
        idx = np.searchsorted(volume_bins[::-1], volume_target)
        if idx >= len(volume_bins):
            return dose_bins[0]
        
        idx = len(volume_bins) - idx - 1
        
        if idx == len(volume_bins) - 1:
            return dose_bins[-1]
        
        # Nội suy tuyến tính
        x0, x1 = volume_bins[idx], volume_bins[idx+1]
        y0, y1 = dose_bins[idx], dose_bins[idx+1]
        
        if x0 == x1:
            return y0
        
        return y0 + (y1 - y0) * (volume_target - x0) / (x1 - x0)
    
    def generate_dvh_plot(self, dvh_data, output_path=None, structures_to_plot=None, 
                          title="Dose Volume Histogram", figsize=(10, 6)):
        """
        Tạo biểu đồ DVH từ dữ liệu DVH đã tính toán.
        
        Parameters
        ----------
        dvh_data : dict
            Dữ liệu DVH từ phương thức calculate_dvh.
        output_path : str, optional
            Đường dẫn để lưu biểu đồ. Nếu None, biểu đồ sẽ được hiển thị.
        structures_to_plot : list, optional
            Danh sách các cấu trúc cần vẽ. Nếu None, tất cả cấu trúc sẽ được vẽ.
        title : str, optional
            Tiêu đề biểu đồ, mặc định là "Dose Volume Histogram".
        figsize : tuple, optional
            Kích thước biểu đồ, mặc định là (10, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure của Matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MultipleLocator
            
            plt.figure(figsize=figsize)
            
            # Màu sắc và kiểu đường cho các cấu trúc khác nhau
            colors = plt.cm.tab10.colors
            line_styles = ['-', '--', '-.', ':']
            
            # Lọc các cấu trúc cần vẽ
            if structures_to_plot is None:
                structures_to_plot = list(dvh_data.keys())
            
            for i, structure in enumerate(structures_to_plot):
                if structure not in dvh_data:
                    logger.warning(f"Cấu trúc {structure} không có trong dữ liệu DVH")
                    continue
                
                structure_data = dvh_data[structure]
                dose_bins = structure_data['dose_bins']
                volume_bins = structure_data['volume_bins']
                
                color_idx = i % len(colors)
                style_idx = (i // len(colors)) % len(line_styles)
                
                plt.plot(
                    dose_bins, volume_bins, 
                    label=structure,
                    color=colors[color_idx],
                    linestyle=line_styles[style_idx],
                    linewidth=2
                )
            
            plt.xlabel('Liều (Gy)' if dose_bins[0] < 10 else 'Liều (%)')
            plt.ylabel('Thể tích (%)')
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Tùy chỉnh trục x
            plt.xlim(0, max(np.max(dvh_data[structure]['dose_bins']) for structure in structures_to_plot if structure in dvh_data))
            plt.ylim(0, 100)
            
            # Thêm đường dọc ở mức liều quan trọng (ví dụ 95%)
            if dose_bins[0] >= 10:  # Đây là liều tương đối
                plt.axvline(x=95, color='red', linestyle='--', alpha=0.5)
                plt.text(95, 50, '95%', rotation=90, verticalalignment='center')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Đã lưu biểu đồ DVH vào {output_path}")
            
            return plt.gcf()
        
        except ImportError:
            logger.error("Không thể tạo biểu đồ DVH: matplotlib không được cài đặt")
            return None 