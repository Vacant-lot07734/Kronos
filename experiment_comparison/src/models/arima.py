import numpy as np
from typing import Dict, List, Tuple, Optional

from statsmodels.tsa.arima.model import ARIMA


class ARIMAForecaster:
    """
    对多维特征逐维拟合 ARIMA：
      输入：最近 lookback 天的历史序列（每维单独）
      输出：未来 pred_len 天的预测（拼成 [pred_len, n_features]）
    注意：ARIMA是单变量模型；这里是“逐维ARIMA”。
    """

    def __init__(self, order: Tuple[int, int, int] = (3, 1, 0), enforce_stationarity: bool = False,
                 enforce_invertibility: bool = False):
        self.order = order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def fit_predict_one_window(self, hist_2d: np.ndarray, pred_len: int) -> np.ndarray:
        """
        hist_2d: [lookback, n_features]
        return:  [pred_len, n_features]
        """
        hist_2d = np.asarray(hist_2d, dtype=np.float64)
        lookback, n_features = hist_2d.shape

        preds = np.zeros((pred_len, n_features), dtype=np.float64)

        for j in range(n_features):
            y = hist_2d[:, j]

            # 处理常见异常：全常数、全0、极短有效值
            if np.allclose(y, y[0]):
                preds[:, j] = y[-1]
                continue

            try:
                model = ARIMA(
                    y,
                    order=self.order,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                )
                res = model.fit()
                fc = res.forecast(steps=pred_len)
                preds[:, j] = np.asarray(fc, dtype=np.float64)
            except Exception:
                # 拟合失败时，退化为“最后值延续”
                preds[:, j] = y[-1]

        return preds
