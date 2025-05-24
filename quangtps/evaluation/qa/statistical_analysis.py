"""
QuangTPS Statistical Analysis for QA

Module phân tích thống kê nâng cao cho quality assurance.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import norm, t as t_dist
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings for statistical calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class StatisticalTestResult:
    """Kết quả một statistical test."""

    test_name: str
    test_statistic: float
    p_value: float
    critical_value: float
    degrees_of_freedom: Optional[int] = None
    confidence_level: float = 0.95
    is_significant: bool = False
    interpretation: str = ""
    test_type: str = ""


@dataclass
class ConfidenceInterval:
    """Confidence interval cho một metric."""

    metric_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    sample_size: int = 0
    standard_error: float = 0.0


@dataclass
class DescriptiveStatistics:
    """Thống kê mô tả cho một dataset."""

    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    range_val: float = 0.0

    # Percentiles
    q1: float = 0.0  # 25th percentile
    q3: float = 0.0  # 75th percentile
    iqr: float = 0.0  # Interquartile range

    # Distribution properties
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Outlier detection
    outliers_count: int = 0
    outliers_percentage: float = 0.0


class StatisticalAnalyzer:
    """Analyzer cho các phân tích thống kê QA."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.

        Args:
            confidence_level: Confidence level for intervals and tests
        """
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate_descriptive_statistics(
        self, data: np.ndarray, remove_outliers: bool = False
    ) -> DescriptiveStatistics:
        """
        Tính toán descriptive statistics.

        Args:
            data: Input data array
            remove_outliers: Whether to remove outliers before calculation

        Returns:
            DescriptiveStatistics: Comprehensive descriptive statistics
        """
        try:
            # Remove NaN and infinite values
            clean_data = data[np.isfinite(data)]

            if len(clean_data) == 0:
                return DescriptiveStatistics()

            # Detect outliers using IQR method
            q1 = np.percentile(clean_data, 25)
            q3 = np.percentile(clean_data, 75)
            iqr = q3 - q1

            # Outlier boundaries
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = clean_data[
                (clean_data < lower_bound) | (clean_data > upper_bound)
            ]
            outliers_count = len(outliers)
            outliers_percentage = (outliers_count / len(clean_data)) * 100

            # Remove outliers if requested
            if remove_outliers:
                analysis_data = clean_data[
                    (clean_data >= lower_bound) & (clean_data <= upper_bound)
                ]
            else:
                analysis_data = clean_data

            if len(analysis_data) == 0:
                return DescriptiveStatistics(
                    count=len(clean_data),
                    outliers_count=outliers_count,
                    outliers_percentage=outliers_percentage,
                )

            # Calculate statistics
            count = len(analysis_data)
            mean_val = np.mean(analysis_data)
            median_val = np.median(analysis_data)
            std_val = np.std(analysis_data, ddof=1) if count > 1 else 0.0
            var_val = np.var(analysis_data, ddof=1) if count > 1 else 0.0
            min_val = np.min(analysis_data)
            max_val = np.max(analysis_data)
            range_val = max_val - min_val

            # Distribution properties
            try:
                skewness_val = stats.skew(analysis_data)
                kurtosis_val = stats.kurtosis(analysis_data)
            except:
                skewness_val = 0.0
                kurtosis_val = 0.0

            return DescriptiveStatistics(
                count=count,
                mean=mean_val,
                median=median_val,
                std_dev=std_val,
                variance=var_val,
                minimum=min_val,
                maximum=max_val,
                range_val=range_val,
                q1=q1,
                q3=q3,
                iqr=iqr,
                skewness=skewness_val,
                kurtosis=kurtosis_val,
                outliers_count=outliers_count,
                outliers_percentage=outliers_percentage,
            )

        except Exception as e:
            self.logger.error(f"Error calculating descriptive statistics: {e}")
            return DescriptiveStatistics()

    def calculate_confidence_interval(
        self,
        data: np.ndarray,
        metric_name: str,
        confidence_level: Optional[float] = None,
    ) -> ConfidenceInterval:
        """
        Tính confidence interval cho mean.

        Args:
            data: Input data
            metric_name: Name of the metric
            confidence_level: Confidence level (default uses instance level)

        Returns:
            ConfidenceInterval: Confidence interval information
        """
        try:
            conf_level = confidence_level or self.confidence_level
            clean_data = data[np.isfinite(data)]

            if len(clean_data) < 2:
                return ConfidenceInterval(
                    metric_name=metric_name,
                    point_estimate=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    confidence_level=conf_level,
                )

            n = len(clean_data)
            mean_val = np.mean(clean_data)
            std_val = np.std(clean_data, ddof=1)
            std_error = std_val / np.sqrt(n)

            # Use t-distribution for small samples
            if n < 30:
                df = n - 1
                t_critical = t_dist.ppf((1 + conf_level) / 2, df)
                margin_error = t_critical * std_error
            else:
                # Use normal distribution for large samples
                z_critical = norm.ppf((1 + conf_level) / 2)
                margin_error = z_critical * std_error

            lower_bound = mean_val - margin_error
            upper_bound = mean_val + margin_error

            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=mean_val,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=conf_level,
                sample_size=n,
                standard_error=std_error,
            )

        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            return ConfidenceInterval(
                metric_name=metric_name,
                point_estimate=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence_level=conf_level,
            )

    def one_sample_t_test(
        self, data: np.ndarray, population_mean: float, alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Thực hiện one-sample t-test.

        Args:
            data: Sample data
            population_mean: Population mean to test against
            alternative: "two-sided", "greater", or "less"

        Returns:
            StatisticalTestResult: Test results
        """
        try:
            clean_data = data[np.isfinite(data)]

            if len(clean_data) < 2:
                return StatisticalTestResult(
                    test_name="One-Sample T-Test",
                    test_statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    interpretation="Insufficient data for test",
                    test_type="parametric",
                )

            # Perform t-test
            t_statistic, p_value = stats.ttest_1samp(clean_data, population_mean)

            # Adjust p-value for one-sided tests
            if alternative == "greater":
                p_value = p_value / 2 if t_statistic > 0 else 1 - p_value / 2
            elif alternative == "less":
                p_value = p_value / 2 if t_statistic < 0 else 1 - p_value / 2

            n = len(clean_data)
            df = n - 1
            alpha = 1 - self.confidence_level

            if alternative == "two-sided":
                critical_value = t_dist.ppf(1 - alpha / 2, df)
            else:
                critical_value = t_dist.ppf(1 - alpha, df)

            is_significant = p_value < alpha

            # Create interpretation
            sample_mean = np.mean(clean_data)
            if is_significant:
                interpretation = f"Significant difference between sample mean ({sample_mean:.3f}) and population mean ({population_mean:.3f})"
            else:
                interpretation = f"No significant difference between sample mean ({sample_mean:.3f}) and population mean ({population_mean:.3f})"

            return StatisticalTestResult(
                test_name="One-Sample T-Test",
                test_statistic=float(t_statistic),
                p_value=float(p_value),
                critical_value=float(critical_value),
                degrees_of_freedom=df,
                confidence_level=self.confidence_level,
                is_significant=is_significant,
                interpretation=interpretation,
                test_type="parametric",
            )

        except Exception as e:
            self.logger.error(f"Error in one-sample t-test: {e}")
            return StatisticalTestResult(
                test_name="One-Sample T-Test",
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                interpretation=f"Test failed: {str(e)}",
                test_type="parametric",
            )

    def two_sample_t_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        equal_var: bool = False,
        alternative: str = "two-sided",
    ) -> StatisticalTestResult:
        """
        Thực hiện two-sample t-test.

        Args:
            data1: First sample
            data2: Second sample
            equal_var: Assume equal variances
            alternative: "two-sided", "greater", or "less"

        Returns:
            StatisticalTestResult: Test results
        """
        try:
            clean_data1 = data1[np.isfinite(data1)]
            clean_data2 = data2[np.isfinite(data2)]

            if len(clean_data1) < 2 or len(clean_data2) < 2:
                return StatisticalTestResult(
                    test_name="Two-Sample T-Test",
                    test_statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    interpretation="Insufficient data for test",
                    test_type="parametric",
                )

            # Perform t-test
            t_statistic, p_value = stats.ttest_ind(
                clean_data1, clean_data2, equal_var=equal_var
            )

            # Adjust p-value for one-sided tests
            if alternative == "greater":
                p_value = p_value / 2 if t_statistic > 0 else 1 - p_value / 2
            elif alternative == "less":
                p_value = p_value / 2 if t_statistic < 0 else 1 - p_value / 2

            # Calculate degrees of freedom
            n1, n2 = len(clean_data1), len(clean_data2)
            if equal_var:
                df = n1 + n2 - 2
            else:
                # Welch's formula
                s1_sq = np.var(clean_data1, ddof=1)
                s2_sq = np.var(clean_data2, ddof=1)
                df = (s1_sq / n1 + s2_sq / n2) ** 2 / (
                    (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
                )

            alpha = 1 - self.confidence_level

            if alternative == "two-sided":
                critical_value = t_dist.ppf(1 - alpha / 2, df)
            else:
                critical_value = t_dist.ppf(1 - alpha, df)

            is_significant = p_value < alpha

            # Create interpretation
            mean1, mean2 = np.mean(clean_data1), np.mean(clean_data2)
            if is_significant:
                interpretation = f"Significant difference between groups (means: {mean1:.3f} vs {mean2:.3f})"
            else:
                interpretation = f"No significant difference between groups (means: {mean1:.3f} vs {mean2:.3f})"

            return StatisticalTestResult(
                test_name="Two-Sample T-Test",
                test_statistic=float(t_statistic),
                p_value=float(p_value),
                critical_value=float(critical_value),
                degrees_of_freedom=int(df),
                confidence_level=self.confidence_level,
                is_significant=is_significant,
                interpretation=interpretation,
                test_type="parametric",
            )

        except Exception as e:
            self.logger.error(f"Error in two-sample t-test: {e}")
            return StatisticalTestResult(
                test_name="Two-Sample T-Test",
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                interpretation=f"Test failed: {str(e)}",
                test_type="parametric",
            )

    def wilcoxon_signed_rank_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Thực hiện Wilcoxon signed-rank test (non-parametric paired test).

        Args:
            data1: First sample (paired)
            data2: Second sample (paired)
            alternative: "two-sided", "greater", or "less"

        Returns:
            StatisticalTestResult: Test results
        """
        try:
            # Ensure same length
            min_len = min(len(data1), len(data2))
            clean_data1 = data1[:min_len]
            clean_data2 = data2[:min_len]

            # Remove pairs with NaN or infinite values
            valid_mask = np.isfinite(clean_data1) & np.isfinite(clean_data2)
            clean_data1 = clean_data1[valid_mask]
            clean_data2 = clean_data2[valid_mask]

            if len(clean_data1) < 6:  # Minimum for Wilcoxon test
                return StatisticalTestResult(
                    test_name="Wilcoxon Signed-Rank Test",
                    test_statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    interpretation="Insufficient data for test (minimum 6 pairs required)",
                    test_type="non-parametric",
                )

            # Perform Wilcoxon test
            statistic, p_value = stats.wilcoxon(
                clean_data1, clean_data2, alternative=alternative
            )

            alpha = 1 - self.confidence_level
            is_significant = p_value < alpha

            # Create interpretation
            median1, median2 = np.median(clean_data1), np.median(clean_data2)
            if is_significant:
                interpretation = f"Significant difference between paired samples (medians: {median1:.3f} vs {median2:.3f})"
            else:
                interpretation = f"No significant difference between paired samples (medians: {median1:.3f} vs {median2:.3f})"

            return StatisticalTestResult(
                test_name="Wilcoxon Signed-Rank Test",
                test_statistic=float(statistic),
                p_value=float(p_value),
                critical_value=0.0,  # Critical values not typically used for this test
                confidence_level=self.confidence_level,
                is_significant=is_significant,
                interpretation=interpretation,
                test_type="non-parametric",
            )

        except Exception as e:
            self.logger.error(f"Error in Wilcoxon signed-rank test: {e}")
            return StatisticalTestResult(
                test_name="Wilcoxon Signed-Rank Test",
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                interpretation=f"Test failed: {str(e)}",
                test_type="non-parametric",
            )

    def mann_whitney_u_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Thực hiện Mann-Whitney U test (non-parametric independent samples test).

        Args:
            data1: First independent sample
            data2: Second independent sample
            alternative: "two-sided", "greater", or "less"

        Returns:
            StatisticalTestResult: Test results
        """
        try:
            clean_data1 = data1[np.isfinite(data1)]
            clean_data2 = data2[np.isfinite(data2)]

            if len(clean_data1) < 3 or len(clean_data2) < 3:
                return StatisticalTestResult(
                    test_name="Mann-Whitney U Test",
                    test_statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    interpretation="Insufficient data for test (minimum 3 samples each)",
                    test_type="non-parametric",
                )

            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                clean_data1, clean_data2, alternative=alternative
            )

            alpha = 1 - self.confidence_level
            is_significant = p_value < alpha

            # Create interpretation
            median1, median2 = np.median(clean_data1), np.median(clean_data2)
            if is_significant:
                interpretation = f"Significant difference between independent groups (medians: {median1:.3f} vs {median2:.3f})"
            else:
                interpretation = f"No significant difference between independent groups (medians: {median1:.3f} vs {median2:.3f})"

            return StatisticalTestResult(
                test_name="Mann-Whitney U Test",
                test_statistic=float(statistic),
                p_value=float(p_value),
                critical_value=0.0,  # Critical values not typically used
                confidence_level=self.confidence_level,
                is_significant=is_significant,
                interpretation=interpretation,
                test_type="non-parametric",
            )

        except Exception as e:
            self.logger.error(f"Error in Mann-Whitney U test: {e}")
            return StatisticalTestResult(
                test_name="Mann-Whitney U Test",
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                interpretation=f"Test failed: {str(e)}",
                test_type="non-parametric",
            )

    def normality_test(self, data: np.ndarray) -> StatisticalTestResult:
        """
        Test normality using Shapiro-Wilk test.

        Args:
            data: Data to test for normality

        Returns:
            StatisticalTestResult: Normality test results
        """
        try:
            clean_data = data[np.isfinite(data)]

            if len(clean_data) < 3:
                return StatisticalTestResult(
                    test_name="Shapiro-Wilk Normality Test",
                    test_statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    interpretation="Insufficient data for normality test",
                    test_type="normality",
                )

            if len(clean_data) > 5000:
                # Use Anderson-Darling for large samples
                statistic, critical_values, significance_levels = stats.anderson(
                    clean_data, dist="norm"
                )
                # Use 5% significance level
                alpha = 0.05
                critical_value = critical_values[2]  # 5% level
                is_significant = statistic > critical_value
                p_value = 0.05 if is_significant else 0.1  # Approximate

                test_name = "Anderson-Darling Normality Test"
            else:
                # Use Shapiro-Wilk for smaller samples
                statistic, p_value = stats.shapiro(clean_data)
                alpha = 0.05
                critical_value = 0.05  # p-value threshold
                is_significant = p_value < alpha

                test_name = "Shapiro-Wilk Normality Test"

            if is_significant:
                interpretation = "Data significantly deviates from normal distribution"
            else:
                interpretation = "Data appears to be normally distributed"

            return StatisticalTestResult(
                test_name=test_name,
                test_statistic=float(statistic),
                p_value=float(p_value),
                critical_value=float(critical_value),
                confidence_level=0.95,
                is_significant=is_significant,
                interpretation=interpretation,
                test_type="normality",
            )

        except Exception as e:
            self.logger.error(f"Error in normality test: {e}")
            return StatisticalTestResult(
                test_name="Normality Test",
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                interpretation=f"Test failed: {str(e)}",
                test_type="normality",
            )

    def correlation_analysis(
        self, data1: np.ndarray, data2: np.ndarray, method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Analyze correlation between two datasets.

        Args:
            data1: First dataset
            data2: Second dataset
            method: "pearson", "spearman", or "kendall"

        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        try:
            # Ensure same length and remove invalid values
            min_len = min(len(data1), len(data2))
            clean_data1 = data1[:min_len]
            clean_data2 = data2[:min_len]

            valid_mask = np.isfinite(clean_data1) & np.isfinite(clean_data2)
            clean_data1 = clean_data1[valid_mask]
            clean_data2 = clean_data2[valid_mask]

            if len(clean_data1) < 3:
                return {
                    "method": method,
                    "correlation": 0.0,
                    "p_value": 1.0,
                    "sample_size": len(clean_data1),
                    "interpretation": "Insufficient data for correlation analysis",
                    "confidence_interval": None,
                }

            # Calculate correlation
            if method.lower() == "pearson":
                correlation, p_value = stats.pearsonr(clean_data1, clean_data2)
            elif method.lower() == "spearman":
                correlation, p_value = stats.spearmanr(clean_data1, clean_data2)
            elif method.lower() == "kendall":
                correlation, p_value = stats.kendalltau(clean_data1, clean_data2)
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            # Calculate confidence interval for Pearson correlation
            confidence_interval = None
            if method.lower() == "pearson" and len(clean_data1) > 3:
                try:
                    # Fisher z-transformation
                    z = np.arctanh(correlation)
                    se = 1 / np.sqrt(len(clean_data1) - 3)
                    z_critical = norm.ppf((1 + self.confidence_level) / 2)

                    z_lower = z - z_critical * se
                    z_upper = z + z_critical * se

                    # Transform back
                    lower_bound = np.tanh(z_lower)
                    upper_bound = np.tanh(z_upper)

                    confidence_interval = {
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "confidence_level": self.confidence_level,
                    }
                except:
                    confidence_interval = None

            # Interpret correlation strength
            abs_corr = abs(correlation)
            if abs_corr < 0.1:
                strength = "negligible"
            elif abs_corr < 0.3:
                strength = "weak"
            elif abs_corr < 0.5:
                strength = "moderate"
            elif abs_corr < 0.7:
                strength = "strong"
            else:
                strength = "very strong"

            direction = "positive" if correlation > 0 else "negative"

            alpha = 0.05
            is_significant = p_value < alpha

            if is_significant:
                interpretation = f"Significant {strength} {direction} correlation"
            else:
                interpretation = f"No significant correlation ({strength} {direction})"

            return {
                "method": method,
                "correlation": float(correlation),
                "p_value": float(p_value),
                "sample_size": len(clean_data1),
                "is_significant": is_significant,
                "strength": strength,
                "direction": direction,
                "interpretation": interpretation,
                "confidence_interval": confidence_interval,
            }

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {
                "method": method,
                "correlation": 0.0,
                "p_value": 1.0,
                "sample_size": 0,
                "interpretation": f"Analysis failed: {str(e)}",
                "confidence_interval": None,
            }


__all__ = [
    "StatisticalTestResult",
    "ConfidenceInterval",
    "DescriptiveStatistics",
    "StatisticalAnalyzer",
]
