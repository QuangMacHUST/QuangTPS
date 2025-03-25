#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module huấn luyện mô hình KBP (Knowledge-Based Planning) cho hệ thống QuangTPS.

Module này cung cấp các công cụ để thu thập dữ liệu từ các kế hoạch xạ trị đã có,
trích xuất đặc trưng, và huấn luyện các mô hình dự đoán cho các ràng buộc liều và tham số tối ưu.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import glob
import json
from tqdm import tqdm

from quangtps.optimization.kbp.model import KBPModel, KBPFeatures, ModelType
from quangtps.database.patient_db import PatientDatabase
from quangtps.database.plan_db import PlanDB
from quangtps.database.structure_db import StructureDatabase
from quangtps.core.exceptions import ModelError, TrainingError
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh import calculate_dvh
from quangtps.imaging.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class KBPDataCollector:
    """
    Thu thập dữ liệu từ kế hoạch đã có cho huấn luyện mô hình KBP.
    """
    
    def __init__(self):
        """Khởi tạo bộ thu thập dữ liệu KBP."""
        self.patient_db = PatientDatabase()
        self.plan_db = PlanDB()
        self.structure_db = StructureDatabase()
    
    def collect_plan_data(
        self,
        site: Optional[str] = None,
        min_plans: int = 10,
        max_plans: int = 1000,
        approved_only: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Thu thập dữ liệu từ các kế hoạch điều trị đã có.
        
        Args:
            site: Vị trí điều trị cụ thể để lọc (ví dụ: 'Prostate', 'H&N')
            min_plans: Số lượng kế hoạch tối thiểu cần thu thập
            max_plans: Số lượng kế hoạch tối đa để thu thập
            approved_only: Chỉ thu thập các kế hoạch đã được phê duyệt
            
        Returns:
            Dict: Dữ liệu kế hoạch theo vị trí điều trị
        """
        # Lấy danh sách bệnh nhân
        patients = self.patient_db.get_all_patients()
        
        # Kết quả theo vị trí điều trị
        results = {}
        
        total_plans_collected = 0
        
        for patient in tqdm(patients, desc="Thu thập dữ liệu kế hoạch"):
            patient_id = patient["id"]
            
            # Lấy tất cả kế hoạch cho bệnh nhân
            plans = self.plan_db.get_plans_for_patient(patient_id)
            
            for plan in plans:
                plan_id = plan["id"]
                
                # Kiểm tra trạng thái kế hoạch nếu cần
                if approved_only and plan["approval_status"] != "APPROVED":
                    continue
                
                # Lấy vị trí điều trị của kế hoạch
                plan_site = plan.get("site", "Unknown")
                
                if site is not None and plan_site != site:
                    continue
                
                # Lấy cấu trúc và kế hoạch chi tiết
                structures = self.structure_db.get_structures_for_plan(plan_id)
                plan_details = self.plan_db.get_plan_details(plan_id)
                
                if not structures or not plan_details:
                    continue
                
                # Tách PTV và OARs
                ptvs = [s for s in structures if s["name"].lower().startswith("ptv")]
                oars = [s for s in structures if not s["name"].lower().startswith("ptv")]
                
                if not ptvs:
                    continue
                
                # Lấy thông tin liều và phân đoạn
                prescription = plan_details.get("prescription", {})
                dose = prescription.get("dose", 0)
                fractions = prescription.get("fractions", 0)
                
                if dose <= 0 or fractions <= 0:
                    continue
                
                # Thu thập dữ liệu tối ưu
                optimization_data = plan_details.get("optimization", {})
                objectives = optimization_data.get("objectives", [])
                constraints = optimization_data.get("constraints", [])
                
                # Tạo bản ghi dữ liệu cho kế hoạch này
                plan_data = {
                    "patient_id": patient_id,
                    "plan_id": plan_id,
                    "site": plan_site,
                    "dose": dose,
                    "fractions": fractions,
                    "ptvs": ptvs,
                    "oars": oars,
                    "objectives": objectives,
                    "constraints": constraints,
                    "dvh_data": self._get_dvh_data(plan_id),
                    "metadata": plan
                }
                
                # Thêm vào kết quả
                if plan_site not in results:
                    results[plan_site] = []
                
                results[plan_site].append(plan_data)
                total_plans_collected += 1
                
                if total_plans_collected >= max_plans:
                    break
            
            if total_plans_collected >= max_plans:
                break
        
        # Kiểm tra số lượng kế hoạch thu thập được
        total_all_sites = sum(len(plans) for plans in results.values())
        
        if total_all_sites < min_plans:
            logger.warning(f"Chỉ thu thập được {total_all_sites} kế hoạch, nhỏ hơn yêu cầu tối thiểu {min_plans}")
        
        # Log kết quả
        for site_name, plans in results.items():
            logger.info(f"Đã thu thập {len(plans)} kế hoạch cho vị trí {site_name}")
        
        return results
    
    def _get_dvh_data(self, plan_id: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu DVH cho một kế hoạch.
        
        Args:
            plan_id: ID kế hoạch
            
        Returns:
            Dict: Dữ liệu DVH
        """
        # Trong trường hợp thực tế, bạn sẽ truy vấn DVH từ cơ sở dữ liệu
        # Đây là một phiên bản đơn giản hóa
        try:
            dvh_data = {}
            # TODO: Thực hiện truy vấn DVH từ cơ sở dữ liệu
            return dvh_data
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu DVH cho kế hoạch {plan_id}: {str(e)}")
            return {}
    
    def extract_training_features(
        self,
        plan_data_by_site: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, np.ndarray]]]:
        """
        Trích xuất các đặc trưng cho huấn luyện.
        
        Args:
            plan_data_by_site: Dữ liệu kế hoạch theo vị trí điều trị
            
        Returns:
            Tuple: (DataFrame đặc trưng theo vị trí, Nhãn đầu ra theo vị trí)
        """
        features_by_site = {}
        targets_by_site = {}
        
        for site, plans in plan_data_by_site.items():
            logger.info(f"Trích xuất đặc trưng cho {len(plans)} kế hoạch tại vị trí {site}")
            
            site_features = []
            site_targets = {}
            
            for plan in tqdm(plans, desc=f"Trích xuất đặc trưng {site}"):
                try:
                    # Trích xuất đặc trưng cơ bản
                    plan_feature = {
                        "patient_id": plan["patient_id"],
                        "plan_id": plan["plan_id"],
                        "site": plan["site"],
                        "prescription_dose": plan["dose"],
                        "fractions": plan["fractions"]
                    }
                    
                    # Thêm đặc trưng hình học
                    ptvs = plan["ptvs"]
                    oars = plan["oars"]
                    
                    for ptv in ptvs:
                        ptv_name = ptv["name"]
                        plan_feature[f"{ptv_name}_volume"] = ptv.get("volume", 0)
                        
                        # Thêm đặc trưng hình học khác nếu có
                        if "surface_area" in ptv:
                            plan_feature[f"{ptv_name}_surface_area"] = ptv["surface_area"]
                        
                        if "center" in ptv:
                            plan_feature[f"{ptv_name}_center_x"] = ptv["center"][0]
                            plan_feature[f"{ptv_name}_center_y"] = ptv["center"][1]
                            plan_feature[f"{ptv_name}_center_z"] = ptv["center"][2]
                    
                    for oar in oars:
                        oar_name = oar["name"]
                        plan_feature[f"{oar_name}_volume"] = oar.get("volume", 0)
                        
                        # Thêm khoảng cách từ PTV đến OAR nếu có
                        for ptv in ptvs:
                            ptv_name = ptv["name"]
                            distance_key = f"distance_{ptv_name}_to_{oar_name}"
                            
                            if distance_key in ptv:
                                plan_feature[distance_key] = ptv[distance_key]
                            else:
                                # Tính giá trị mặc định nếu không có
                                plan_feature[distance_key] = -1
                    
                    # Thêm đặc trưng chồng lấp
                    for ptv in ptvs:
                        ptv_name = ptv["name"]
                        for oar in oars:
                            oar_name = oar["name"]
                            overlap_key = f"overlap_{ptv_name}_to_{oar_name}"
                            
                            if overlap_key in ptv:
                                plan_feature[overlap_key] = ptv[overlap_key]
                            else:
                                plan_feature[overlap_key] = 0
                    
                    # Thêm đặc trưng này vào danh sách
                    site_features.append(plan_feature)
                    
                    # Trích xuất nhãn đầu ra (mục tiêu và ràng buộc)
                    for objective in plan["objectives"]:
                        obj_type = objective["type"]
                        struct_name = objective["structure"]
                        value = objective["value"]
                        weight = objective.get("weight", 1.0)
                        
                        # Tạo tên mục tiêu
                        target_name = f"objective_{struct_name}_{obj_type}"
                        
                        if target_name not in site_targets:
                            site_targets[target_name] = []
                        
                        site_targets[target_name].append(value)
                        
                        # Thêm trọng số
                        weight_name = f"weight_{struct_name}_{obj_type}"
                        
                        if weight_name not in site_targets:
                            site_targets[weight_name] = []
                        
                        site_targets[weight_name].append(weight)
                    
                    for constraint in plan["constraints"]:
                        const_type = constraint["type"]
                        struct_name = constraint["structure"]
                        value = constraint["value"]
                        
                        # Tạo tên ràng buộc
                        target_name = f"constraint_{struct_name}_{const_type}"
                        
                        if target_name not in site_targets:
                            site_targets[target_name] = []
                        
                        site_targets[target_name].append(value)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi trích xuất đặc trưng cho kế hoạch {plan['plan_id']}: {str(e)}")
                    continue
            
            # Chuyển đổi thành DataFrame
            if site_features:
                features_df = pd.DataFrame(site_features)
                features_by_site[site] = features_df
                
                # Chuyển đổi danh sách nhãn thành mảng numpy
                targets_dict = {}
                for target_name, values in site_targets.items():
                    if len(values) == len(site_features):
                        targets_dict[target_name] = np.array(values)
                    else:
                        logger.warning(f"Số lượng mẫu không khớp cho {target_name}: {len(values)} vs {len(site_features)}")
                
                targets_by_site[site] = targets_dict
            else:
                logger.warning(f"Không có đặc trưng nào được trích xuất cho vị trí {site}")
        
        return features_by_site, targets_by_site


class KBPTrainer:
    """
    Huấn luyện mô hình KBP từ dữ liệu kế hoạch.
    """
    
    def __init__(self, output_dir: str = "models/kbp"):
        """
        Khởi tạo bộ huấn luyện KBP.
        
        Args:
            output_dir: Thư mục đầu ra cho mô hình
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.data_collector = KBPDataCollector()
        self.models = {}  # Lưu trữ mô hình theo vị trí điều trị
    
    def collect_training_data(
        self,
        sites: Optional[List[str]] = None,
        min_plans_per_site: int = 10
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, np.ndarray]]]:
        """
        Thu thập dữ liệu huấn luyện.
        
        Args:
            sites: Danh sách vị trí điều trị cần thu thập
            min_plans_per_site: Số lượng kế hoạch tối thiểu cho mỗi vị trí
            
        Returns:
            Tuple: (DataFrame đặc trưng theo vị trí, Nhãn đầu ra theo vị trí)
        """
        all_plan_data = {}
        
        if sites:
            # Thu thập dữ liệu cho các vị trí cụ thể
            for site in sites:
                plan_data = self.data_collector.collect_plan_data(
                    site=site,
                    min_plans=min_plans_per_site,
                    approved_only=True
                )
                
                if site in plan_data and len(plan_data[site]) >= min_plans_per_site:
                    all_plan_data[site] = plan_data[site]
                else:
                    logger.warning(f"Không đủ dữ liệu cho vị trí {site}, bỏ qua")
        else:
            # Thu thập tất cả dữ liệu
            all_plan_data = self.data_collector.collect_plan_data(
                min_plans=min_plans_per_site,
                approved_only=True
            )
            
            # Lọc bỏ các vị trí có ít kế hoạch
            all_plan_data = {
                site: plans for site, plans in all_plan_data.items() 
                if len(plans) >= min_plans_per_site
            }
        
        # Trích xuất đặc trưng và nhãn
        features, targets = self.data_collector.extract_training_features(all_plan_data)
        
        # Lưu dữ liệu đã thu thập
        for site in features:
            features_path = os.path.join(self.output_dir, f"{site}_features.csv")
            features[site].to_csv(features_path, index=False)
            
            targets_path = os.path.join(self.output_dir, f"{site}_targets.json")
            targets_dict = {k: v.tolist() for k, v in targets[site].items()}
            
            with open(targets_path, 'w') as f:
                json.dump(targets_dict, f)
            
            logger.info(f"Đã lưu dữ liệu huấn luyện cho {site} tại {self.output_dir}")
        
        return features, targets
    
    def train_models(
        self,
        features_by_site: Dict[str, pd.DataFrame],
        targets_by_site: Dict[str, Dict[str, np.ndarray]],
        model_type: ModelType = ModelType.GRADIENT_BOOSTING,
        do_grid_search: bool = False,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Huấn luyện mô hình KBP cho các vị trí điều trị.
        
        Args:
            features_by_site: DataFrame đặc trưng theo vị trí
            targets_by_site: Nhãn đầu ra theo vị trí
            model_type: Loại mô hình học máy
            do_grid_search: Có thực hiện tìm kiếm lưới tham số không
            model_params: Tham số cho mô hình
            
        Returns:
            Dict: Các chỉ số đánh giá cho mỗi mô hình theo vị trí
        """
        results = {}
        
        for site in features_by_site:
            if site not in targets_by_site:
                logger.warning(f"Không có nhãn cho vị trí {site}, bỏ qua")
                continue
            
            features_df = features_by_site[site]
            targets_dict = targets_by_site[site]
            
            if features_df.empty or not targets_dict:
                logger.warning(f"Dữ liệu trống cho vị trí {site}, bỏ qua")
                continue
            
            # Tạo mô hình KBP
            model = KBPModel(
                model_type=model_type,
                model_params=model_params or {},
                model_name=f"kbp_{site.lower()}"
            )
            
            # Tìm tham số tốt nhất nếu cần
            grid_params = None
            if do_grid_search:
                if model_type == ModelType.GRADIENT_BOOSTING:
                    grid_params = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5]
                    }
                elif model_type == ModelType.RANDOM_FOREST:
                    grid_params = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
            
            # Huấn luyện mô hình
            logger.info(f"Huấn luyện mô hình KBP cho vị trí {site}")
            try:
                metrics = model.train(
                    features_df, 
                    targets_dict,
                    do_grid_search=do_grid_search,
                    grid_params=grid_params
                )
                
                # Lưu mô hình
                model_dir = os.path.join(self.output_dir, site.lower())
                model.save(model_dir)
                
                # Lưu mô hình vào từ điển
                self.models[site] = model
                
                results[site] = metrics
                
                logger.info(f"Đã huấn luyện và lưu mô hình KBP cho vị trí {site}")
                
            except Exception as e:
                logger.error(f"Lỗi khi huấn luyện mô hình cho vị trí {site}: {str(e)}")
                continue
        
        return results
    
    def evaluate_models(
        self,
        test_features_by_site: Optional[Dict[str, pd.DataFrame]] = None,
        test_targets_by_site: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Đánh giá mô hình KBP.
        
        Args:
            test_features_by_site: DataFrame đặc trưng kiểm tra theo vị trí
            test_targets_by_site: Nhãn đầu ra kiểm tra theo vị trí
            test_size: Tỷ lệ dữ liệu kiểm tra nếu không cung cấp dữ liệu riêng
            
        Returns:
            Dict: Các chỉ số đánh giá cho mỗi mô hình và mục tiêu
        """
        if not self.models:
            raise ModelError("Không có mô hình nào để đánh giá")
        
        evaluation = {}
        
        for site, model in self.models.items():
            if test_features_by_site and test_targets_by_site:
                if site in test_features_by_site and site in test_targets_by_site:
                    test_features = test_features_by_site[site]
                    test_targets = test_targets_by_site[site]
                else:
                    logger.warning(f"Không có dữ liệu kiểm tra cho vị trí {site}, bỏ qua")
                    continue
            else:
                # Tải lại dữ liệu đã lưu
                features_path = os.path.join(self.output_dir, f"{site}_features.csv")
                targets_path = os.path.join(self.output_dir, f"{site}_targets.json")
                
                if not os.path.exists(features_path) or not os.path.exists(targets_path):
                    logger.warning(f"Không tìm thấy dữ liệu đã lưu cho vị trí {site}, bỏ qua")
                    continue
                
                features_df = pd.read_csv(features_path)
                
                with open(targets_path, 'r') as f:
                    targets_dict = json.load(f)
                    targets_dict = {k: np.array(v) for k, v in targets_dict.items()}
                
                # Chia dữ liệu thành tập huấn luyện và kiểm tra
                X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
                
                # Chia các đặc trưng
                train_idx, test_idx = train_test_split(
                    np.arange(len(features_df)), 
                    test_size=test_size, 
                    random_state=42
                )
                
                test_features = features_df.iloc[test_idx].reset_index(drop=True)
                
                # Chia các nhãn
                test_targets = {}
                for target_name, target_values in targets_dict.items():
                    test_targets[target_name] = target_values[test_idx]
            
            # Đánh giá mô hình trên tập kiểm tra
            site_evaluation = {}
            
            for target_name in test_targets:
                if target_name in model.models:
                    # Chuẩn hóa dữ liệu
                    if target_name in model.scalers:
                        X_test_scaled = model.scalers[target_name].transform(test_features)
                    else:
                        X_test_scaled = test_features.values
                    
                    # Dự đoán
                    y_pred = model.models[target_name].predict(X_test_scaled)
                    y_true = test_targets[target_name]
                    
                    # Tính các chỉ số
                    mse = np.mean((y_pred - y_true) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_pred - y_true))
                    r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                    
                    site_evaluation[target_name] = {
                        "mse": float(mse),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "r2": float(r2)
                    }
            
            evaluation[site] = site_evaluation
            
            # Vẽ biểu đồ dự đoán vs thực tế
            self._plot_predictions(site, model, test_features, test_targets)
        
        return evaluation
    
    def _plot_predictions(
        self,
        site: str,
        model: KBPModel,
        test_features: pd.DataFrame,
        test_targets: Dict[str, np.ndarray]
    ) -> None:
        """
        Vẽ biểu đồ dự đoán vs giá trị thực.
        
        Args:
            site: Vị trí điều trị
            model: Mô hình KBP
            test_features: Đặc trưng kiểm tra
            test_targets: Nhãn kiểm tra
        """
        output_dir = os.path.join(self.output_dir, site.lower(), "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        for target_name in test_targets:
            if target_name in model.models:
                # Chuẩn hóa dữ liệu
                if target_name in model.scalers:
                    X_test_scaled = model.scalers[target_name].transform(test_features)
                else:
                    X_test_scaled = test_features.values
                
                # Dự đoán
                y_pred = model.models[target_name].predict(X_test_scaled)
                y_true = test_targets[target_name]
                
                # Vẽ biểu đồ
                plt.figure(figsize=(8, 6))
                plt.scatter(y_true, y_pred, alpha=0.7)
                
                # Vẽ đường lý tưởng
                min_val = min(np.min(y_true), np.min(y_pred))
                max_val = max(np.max(y_true), np.max(y_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.xlabel('Giá trị thực')
                plt.ylabel('Dự đoán')
                plt.title(f'Dự đoán vs Thực tế: {target_name}')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Thêm chỉ số đánh giá vào biểu đồ
                mse = np.mean((y_pred - y_true) ** 2)
                r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                
                plt.annotate(f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                # Lưu biểu đồ
                plot_path = os.path.join(output_dir, f"{target_name}_prediction.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Đã lưu biểu đồ dự đoán cho {target_name} tại {plot_path}")
    
    @classmethod
    def load_model(cls, model_dir: str, site: str) -> KBPModel:
        """
        Tải mô hình KBP cho một vị trí điều trị.
        
        Args:
            model_dir: Thư mục chứa mô hình
            site: Vị trí điều trị
            
        Returns:
            KBPModel: Mô hình đã tải
        """
        site_dir = os.path.join(model_dir, site.lower())
        model_name = f"kbp_{site.lower()}"
        
        return KBPModel.load(site_dir, model_name) 