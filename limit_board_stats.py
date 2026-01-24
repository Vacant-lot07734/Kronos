#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涨跌停板统计脚本 - Tushare版本

核心改进：
1. 使用Tushare Pro接口（更稳定）
2. 智能速率限制：每分钟最多500次API调用，自动等待
3. 正确计算连板所需的历史数据（统计N天需要N+6天数据）
4. 只统计特定板块：10cm(60/00)、20cm(3/688)、30cm(8/9)
5. 11个指标：10cm按连板详细统计、20cm/30cm分首板和连板、跌停不分板块
6. 本地缓存机制：标准格式 ts_code.csv (如: 000001.SZ.csv)
7. 涨跌停价格计算规则：
   - 10cm/20cm: 四舍五入到两位小数
   - 30cm(北交所): 向下取整（去尾）到两位小数
"""

import os
import pandas as pd
import tushare as ts
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import warnings
import time
from datetime import datetime, timedelta
from collections import deque
import threading

warnings.filterwarnings('ignore')

# ================= 配置 =================
HISTORY_DAYS = 300  # 统计最近N个交易日
EXTRA_DAYS = 5  # 为计算连板需要额外获取的天数
MAX_WORKERS = 4
OUTPUT_FILE = "涨跌停板统计.csv"
DETAIL_FILE = "涨跌停板明细.txt"  # 详细股票列表
CACHE_DIR = "cache_data"

# Tushare配置
TUSHARE_TOKEN = "b1483905265967690a3ab468e6170387b24b70d0feb16eb10442608a"  # 请在这里填入你的Tushare Token

# API速率限制配置
API_CALLS_PER_MINUTE = 500  # 每分钟最多调用次数（根据你的Tushare权限调整）
RATE_LIMIT_WINDOW = 60  # 时间窗口（秒）
# 说明：
# - 免费用户通常为 200次/分钟
# - 高级用户可能为 500-2000次/分钟
# - 请根据你的Tushare账户权限调整此值

# 统计指标（11个）
ROW_NAMES = [
    "首板", "2连板", "3连板", "4连板", "5连板", "6连板及以上",
    "跌停板", "20cm首板", "20cm连板", "30cm首板", "30cm连板",
]

# 主要指数列表
MAIN_INDEXES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "899050.BJ": "北证50"
}


# ================= 速率限制器 =================

class RateLimiter:
    """API调用速率限制器"""

    def __init__(self, max_calls: int, time_window: int):
        """
        Args:
            max_calls: 时间窗口内最大调用次数
            time_window: 时间窗口（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """如果需要，等待直到可以进行下次调用"""
        with self.lock:
            now = time.time()

            # 移除时间窗口外的调用记录
            while self.call_times and now - self.call_times[0] >= self.time_window:
                self.call_times.popleft()

            # 如果达到限制，等待
            if len(self.call_times) >= self.max_calls:
                sleep_time = self.time_window - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    print(f"  ⏸️  已达到速率限制，等待 {sleep_time:.1f} 秒...")
                    time.sleep(sleep_time)
                    # 清理旧记录
                    now = time.time()
                    while self.call_times and now - self.call_times[0] >= self.time_window:
                        self.call_times.popleft()

            # 记录本次调用
            self.call_times.append(now)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self.lock:
            now = time.time()
            # 清理过期记录
            while self.call_times and now - self.call_times[0] >= self.time_window:
                self.call_times.popleft()

            return {
                "current_calls": len(self.call_times),
                "max_calls": self.max_calls,
                "time_window": self.time_window
            }


# 全局速率限制器
rate_limiter = RateLimiter(API_CALLS_PER_MINUTE, RATE_LIMIT_WINDOW)


# ================= 初始化 =================

def init_tushare():
    """初始化Tushare"""
    if not TUSHARE_TOKEN:
        print("⚠️  请先设置TUSHARE_TOKEN")
        print("   获取方式: https://tushare.pro/register")
        print("   在代码中设置: TUSHARE_TOKEN = '你的token'")
        return None

    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        # 测试连接（使用速率限制）
        rate_limiter.wait_if_needed()
        test = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240102')

        if test is not None:
            print("✅ Tushare连接成功")
            return pro
        else:
            print("❌ Tushare连接失败")
            return None
    except Exception as e:
        print(f"❌ Tushare初始化失败: {e}")
        return None


# ================= 缓存相关函数 =================

def ensure_cache_dir():
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def get_cache_filename(ts_code: str, start_date: str, end_date: str) -> str:
    """生成缓存文件名（仅按 ts_code 命名，不包含日期范围）"""
    return os.path.join(CACHE_DIR, f"{ts_code}.csv")


def load_from_cache(ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从缓存读取数据（只读取标准格式：带交易所后缀）"""
    cache_file = get_cache_filename(ts_code, start_date, end_date)

    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, encoding='utf-8-sig')
            if not df.empty:
                return df
        except Exception:
            # 如果读取失败，尝试删除损坏的缓存文件
            try:
                os.remove(cache_file)
            except:
                pass

    return None


def save_to_cache(df: pd.DataFrame, ts_code: str, start_date: str, end_date: str):
    """保存数据到缓存（文件名格式：<ts_code>.csv）"""
    if df is None or df.empty:
        return

    try:
        ensure_cache_dir()
        cache_file = get_cache_filename(ts_code, start_date, end_date)
        df.to_csv(cache_file, index=False, encoding='utf-8-sig')
    except Exception:
        pass


# ================= 工具函数 =================

def get_last_n_trade_dates(pro, n: int) -> List[str]:
    """获取以【最后一个实际交易日】为右边界的最近 n 个交易日（升序返回）"""
    try:
        # 速率限制
        rate_limiter.wait_if_needed()

        # 拉长窗口，确保足以覆盖 n 天 + 长假（避免节假日抖动）
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=max(90, n * 6))

        # 取交易日历（仅保留开市日）
        df = pro.trade_cal(
            exchange='SSE',
            start_date=start_dt.strftime('%Y%m%d'),
            end_date=end_dt.strftime('%Y%m%d')
        )
        if df is None or df.empty:
            return []

        # 按日期升序，最后一条天然就是“最后一个实际交易日”
        df = df[df['is_open'] == 1].sort_values('cal_date', ascending=True)
        open_days = df['cal_date'].tolist()
        if not open_days:
            return []

        # 以“最后一个实际交易日”为右边界向前取 n 天
        last_n = open_days[-n:]
        # 若极端情况下仍不够，再扩大窗口兜底
        if len(last_n) < n:
            rate_limiter.wait_if_needed()
            df2 = pro.trade_cal(
                exchange='SSE',
                start_date=(end_dt - timedelta(days=365)).strftime('%Y%m%d'),
                end_date=end_dt.strftime('%Y%m%d')
            )
            df2 = df2[df2['is_open'] == 1].sort_values('cal_date', ascending=True)
            last_n = df2['cal_date'].tolist()[-n:]

        # 20240101 -> 2024-01-01（保持全局使用的格式）
        return [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in last_n]

    except Exception as e:
        print(f"❌ 获取交易日失败: {e}")
        return []

def detect_board_type(code: str) -> Tuple[str, float]:
    """
    判断板块类型

    板块分类：
    - 10cm: 60/00开头（沪深主板，±10%，四舍五入）
    - 20cm: 3/688开头（创业板和科创板，±20%，四舍五入）
    - 30cm: 8/9开头（北交所，±30%，向下取整）
    """
    code = str(code).strip()

    # 去掉可能的交易所后缀
    if '.' in code:
        code = code.split('.')[0]

    # 20cm: 3开头(创业板) 或 688开头(科创板)
    if code.startswith('3') or code.startswith('688'):
        return "20", 20.0

    # 30cm: 北交所(8/9开头，6位数字)
    if len(code) == 6 and code[0] in ('8', '9'):
        return "30", 30.0

    # 10cm: 沪深主板(60/00开头)
    if code.startswith(('60', '00')):
        return "10", 10.0

    # 其他情况默认10cm
    return "10", 10.0


def round_price(price: float, rounding_mode: str = "round") -> Decimal:
    """
    价格取整到2位小数

    Args:
        price: 原始价格
        rounding_mode: 取整模式
            - "round": 四舍五入（沪深主板、创业板、科创板）
            - "floor": 向下取整/去尾（北交所）
    """
    try:
        decimal_price = Decimal(str(price))
        if rounding_mode == "floor":
            # 北交所：向下取整（去尾）
            return decimal_price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        else:
            # 沪深A股：四舍五入
            return decimal_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except:
        return Decimal("0")


def is_limit_price(curr_price: float, prev_close: float, limit_pct: float, board_type: str, is_up: bool = True) -> bool:
    """
    判断是否涨停/跌停

    Args:
        curr_price: 当前收盘价
        prev_close: 前收盘价
        limit_pct: 涨跌幅限制（10/20/30）
        board_type: 板块类型（"10"/"20"/"30"）
        is_up: True=涨停，False=跌停
    """
    if curr_price is None or prev_close is None or prev_close <= 0:
        return False

    try:
        # 全程使用 Decimal，避免 float 误差
        coef = (Decimal("1") + Decimal(str(limit_pct)) / Decimal("100")) if is_up \
               else (Decimal("1") - Decimal(str(limit_pct)) / Decimal("100"))

        rounding_mode = "floor" if board_type == "30" else "round"

        target_price = round_price(Decimal(str(prev_close)) * coef, rounding_mode)
        current_price = round_price(Decimal(str(curr_price)), rounding_mode)
        diff = (current_price - target_price).copy_abs()

        # 若你希望“必须完全等于理论价才算到板”，用严格等于：
        return diff == Decimal("0.00")

    except Exception:
        return False


def fetch_daily_data(pro, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    获取日线数据（带缓存）

    Args:
        pro: Tushare Pro API对象
        ts_code: 股票代码(如: 000001.SZ)
        start_date: 开始日期(格式: 20240101)
        end_date: 结束日期(格式: 20241231)
    """
    # 1. 尝试从缓存读取
    cached_df = load_from_cache(ts_code, start_date, end_date)
    if cached_df is not None:
        return cached_df

    # 2. 从Tushare获取
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            return None

        # 3. 保存到缓存
        save_to_cache(df, ts_code, start_date, end_date)

        # 添加短暂延迟，避免超过API限制
        time.sleep(0.02)

        return df

    except Exception as e:
        return None


def fetch_index_data(pro, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    获取指数数据

    Args:
        pro: Tushare Pro API对象
        ts_code: 指数代码
        start_date: 开始日期
        end_date: 结束日期
    """
    try:
        # 应用速率限制
        rate_limiter.wait_if_needed()

        # 获取指数行情数据
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            return None

        return df
    except Exception as e:
        print(f"获取指数 {ts_code} 数据失败: {e}")
        return None


def get_main_indexes_data(pro, dates: List[str]) -> Dict[str, pd.DataFrame]:
    """
    获取主要指数数据

    Args:
        pro: Tushare Pro API对象
        dates: 交易日期列表

    Returns:
        Dict[str, pd.DataFrame]: 指数数据字典
    """
    if not dates:
        return {}

    print("\n📊 获取主要指数数据...")

    # 确定数据获取范围
    start_date = pd.Timestamp(dates[0])
    end_date = pd.Timestamp(dates[-1])

    # 格式化日期
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')

    index_data = {}

    for ts_code, name in MAIN_INDEXES.items():
        print(f"  正在获取 {name}({ts_code}) 数据...")
        df = fetch_index_data(pro, ts_code, start_date_str, end_date_str)

        if df is not None and not df.empty:
            # 保留指数数据中确实存在的列（不包含volume和amount）
            available_cols = ['trade_date', 'close', 'change', 'pct_chg']
            cols_to_use = [col for col in available_cols if col in df.columns]
            df = df[cols_to_use].copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date')
            index_data[ts_code] = df
            print(f"  ✅ {name} 数据获取成功: {len(df)} 条记录")
        else:
            print(f"  ❌ {name} 数据获取失败")

    return index_data


def get_all_stocks(pro) -> List[Tuple[str, str, str]]:
    """
    获取股票列表（仅包含特定板块）

    只统计：
    - 10cm: 60/00开头
    - 20cm: 3/688开头
    - 30cm: 8/9开头

    Returns:
        List[(ts_code, code, name)]
    """
    print("\n📊 获取股票列表...")

    try:
        # 应用速率限制
        rate_limiter.wait_if_needed()

        # 获取A股列表
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')

        if df is None or df.empty:
            print("❌ 无法获取股票列表")
            return []

        stocks = []
        for _, row in df.iterrows():
            ts_code = row['ts_code']
            code = row['symbol']
            name = row['name']

            # 过滤ST股票
            if 'ST' in name.upper():
                continue

            # 只保留特定板块的股票
            # 10cm: 60/00开头, 20cm: 3/688开头, 30cm: 8/9开头
            if code.startswith(('60', '00', '3', '688')) or (len(code) == 6 and code[0] in ('8', '9')):
                stocks.append((ts_code, code, name))

        print(f"✅ 获取成功: {len(stocks)} 只股票")
        print(f"   (仅包含: 60/00/3/688/8/9开头的股票)")
        return stocks

    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return []


# ================= 核心计算 =================

def compute_limit_boards(pro, dates, stocks):
    if not dates or not stocks:
        return {}, {}

    print(f"\n📅 统计范围:")
    print(f"   统计日期: {dates[0]} 至 {dates[-1]} ({len(dates)} 个交易日)")
    print(f"   股票数量: {len(stocks)} 只")

    stat_start_date = pd.Timestamp(dates[0])
    try:
        rate_limiter.wait_if_needed()
        early_date = (stat_start_date - timedelta(days=EXTRA_DAYS * 2)).strftime('%Y%m%d')
        late_date = (stat_start_date - timedelta(days=1)).strftime('%Y%m%d')  # 排除起始日
        cal_df = pro.trade_cal(exchange='SSE', start_date=early_date, end_date=late_date)
        cal_df = cal_df[cal_df['is_open'] == 1].sort_values('cal_date', ascending=False)
        if len(cal_df) >= EXTRA_DAYS:
            data_start_date = pd.Timestamp(cal_df.iloc[EXTRA_DAYS - 1]['cal_date'])
        else:
            data_start_date = stat_start_date - timedelta(days=EXTRA_DAYS)
    except Exception:
        data_start_date = stat_start_date - timedelta(days=EXTRA_DAYS)

    data_end_date = pd.Timestamp(dates[-1])
    print(f"   数据范围: {data_start_date.strftime('%Y-%m-%d')} 至 {data_end_date.strftime('%Y-%m-%d')}")
    print(f"   (需要额外{EXTRA_DAYS}天历史数据来计算连板)")

    result = {date: {name: 0 for name in ROW_NAMES} for date in dates}
    detail_result = {date: {name: [] for name in ROW_NAMES} for date in dates}
    target_dates = set(pd.Timestamp(d).date() for d in dates)

    def process_stock(ts_code, code, name):
        try:
            board_type, limit_pct = detect_board_type(code)
            df = fetch_daily_data(pro, ts_code,
                                  data_start_date.strftime("%Y%m%d"),
                                  data_end_date.strftime("%Y%m%d"))
            if df is None or df.empty:
                return None
            required_cols = ['trade_date', 'close', 'pre_close']
            if not all(col in df.columns for col in required_cols):
                return None

            df = df[['trade_date', 'close', 'pre_close']].copy()
            df.columns = ['日期', '收盘', '前收盘']
            df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
            df = df.sort_values('日期').dropna(subset=['前收盘'])
            if df.empty:
                return None

            df['涨停'] = df.apply(lambda x: is_limit_price(x['收盘'], x['前收盘'], limit_pct, board_type, True), axis=1)
            df['跌停'] = df.apply(lambda x: is_limit_price(x['收盘'], x['前收盘'], limit_pct, board_type, False), axis=1)

            consecutive_boards = 0
            stock_result = {date: {} for date in dates}
            stock_detail = {date: {} for date in dates}

            for _, row in df.iterrows():
                current_date = row['日期']
                if row['涨停']:
                    consecutive_boards += 1
                else:
                    consecutive_boards = 0

                if current_date.date() not in target_dates:
                    continue
                date_str = current_date.strftime("%Y-%m-%d")

                if row['涨停']:
                    if board_type == "10":
                        key = (
                            '首板' if consecutive_boards == 1 else
                            '2连板' if consecutive_boards == 2 else
                            '3连板' if consecutive_boards == 3 else
                            '4连板' if consecutive_boards == 4 else
                            '5连板' if consecutive_boards == 5 else
                            '6连板及以上'
                        )
                        stock_result[date_str][key] = 1
                        stock_detail[date_str][key] = (ts_code, code, name)
                    elif board_type == "20":
                        key = '20cm首板' if consecutive_boards == 1 else '20cm连板'
                        stock_result[date_str][key] = 1
                        stock_detail[date_str][key] = (ts_code, code, name)
                    elif board_type == "30":
                        key = '30cm首板' if consecutive_boards == 1 else '30cm连板'
                        stock_result[date_str][key] = 1
                        stock_detail[date_str][key] = (ts_code, code, name)

                if row['跌停']:
                    stock_result[date_str]['跌停板'] = 1
                    stock_detail[date_str]['跌停板'] = (ts_code, code, name)

            return stock_result, stock_detail
        except Exception:
            return None

    start_date_str = data_start_date.strftime("%Y%m%d")
    end_date_str = data_end_date.strftime("%Y%m%d")
    cached = 0
    for ts_code, code, name in stocks:
        cached_df = load_from_cache(ts_code, start_date_str, end_date_str)
        if cached_df is not None and all(c in cached_df.columns for c in ['trade_date', 'close', 'pre_close']):
            cached += 1
    if cached > 0:
        print(f"📦 缓存命中: {cached}/{len(stocks)} ({cached * 100 // len(stocks)}%)")
        print(f"   缓存文件名: {stocks[0][0]}.csv")

    stats = rate_limiter.get_stats()
    print(f"🚦 速率限制: {stats['current_calls']}/{stats['max_calls']} 次/分钟")

    processed = success = failed = 0
    print(f"\n⏳ 开始处理...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_stock, ts_code, code, name): (ts_code, code, name)
                   for ts_code, code, name in stocks}
        for future in as_completed(futures):
            processed += 1
            res = future.result()
            if res:
                success += 1
                stock_result, stock_detail = res
                for date_str, counts in stock_result.items():
                    for indicator, value in counts.items():
                        result[date_str][indicator] += value
                for date_str, details in stock_detail.items():
                    for indicator, info in details.items():
                        detail_result[date_str][indicator].append(info)
            else:
                failed += 1
                ts_code_i, code_i, name_i = futures[future]
                print(f"   ❌ 失败股票: {ts_code_i} {code_i} {name_i}")

            if processed % 200 == 0:
                stats = rate_limiter.get_stats()
                print(f"  进度: {processed}/{len(stocks)} ({processed * 100 // len(stocks):2d}%) | "
                      f"✅ {success:4d} | ❌ {failed:4d} | "
                      f"🚦 API: {stats['current_calls']}/{stats['max_calls']}")

    print(f"\n✅ 处理完成！")
    print(f"   成功: {success:5d} ({success * 100 // len(stocks):2.1f}%)")
    print(f"   失败: {failed:5d}")
    final_stats = rate_limiter.get_stats()
    print(f"   API调用: 最近1分钟内 {final_stats['current_calls']} 次")

    return result, detail_result


# ================= 主函数 =================


def check_cache_status():
    """检查缓存状态（新的缓存命名：<ts_code>.csv）"""
    print("\n" + "=" * 70)
    print("【缓存诊断】检查缓存文件")
    print("=" * 70)

    if not os.path.exists(CACHE_DIR):
        print(f"❌ 缓存目录不存在: {CACHE_DIR}")
        return

    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.csv')]

    if not cache_files:
        print(f"❌ 缓存目录为空: {CACHE_DIR}")
        return

    print(f"\n📁 缓存目录: {CACHE_DIR}")
    print(f"📊 缓存文件数: {len(cache_files)}")

    # 只统计标准格式（代码+交易所后缀）
    standard_format = [f for f in cache_files if f.endswith('.SZ.csv') or f.endswith('.SH.csv') or f.endswith('.BJ.csv')]

    print(f"\n📋 标准格式文件 (如 000001.SZ.csv): {len(standard_format)} 个")
    if len(standard_format) < len(cache_files):
        print(f"⚠️  非标准格式文件: {len(cache_files) - len(standard_format)} 个（将被忽略）")

    # 示例文件
    sample_files = standard_format[:5] if standard_format else cache_files[:5]
    print(f"\n📋 示例标准格式文件:")
    for f in sample_files:
        print(f"   {f}")

    # 新格式的文件名不含日期范围
    print("\nℹ️  新的缓存命名不包含日期范围，无法直接从文件名推断覆盖的时间区间。")
    print("    如需确认，请打开 CSV 文件查看 trade_date 列。")

    # 同时给出程序当前需要的日期范围（供用户自查数据是否覆盖）
    pro = init_tushare()
    if pro:
        dates = get_last_n_trade_dates(pro, HISTORY_DAYS)
        if dates:
            stat_start_date = pd.Timestamp(dates[0])

            try:
                rate_limiter.wait_if_needed()
                early_date = (stat_start_date - timedelta(days=EXTRA_DAYS * 2)).strftime('%Y%m%d')
                late_date = (stat_start_date - timedelta(days=1)).strftime('%Y%m%d')  # 排除起始日

                cal_df = pro.trade_cal(exchange='SSE', start_date=early_date, end_date=late_date)
                cal_df = cal_df[cal_df['is_open'] == 1].sort_values('cal_date', ascending=False)

                if len(cal_df) >= EXTRA_DAYS:
                    data_start_date = pd.Timestamp(cal_df.iloc[EXTRA_DAYS - 1]['cal_date'])
                else:
                    data_start_date = stat_start_date - timedelta(days=EXTRA_DAYS)
            except Exception:
                data_start_date = stat_start_date - timedelta(days=EXTRA_DAYS)

            data_end_date = pd.Timestamp(dates[-1])

            need_start = data_start_date.strftime('%Y%m%d')
            need_end = data_end_date.strftime('%Y%m%d')

            print(f"\n🎯 程序需要的日期范围:")
            print(f"   {need_start[:4]}-{need_start[4:6]}-{need_start[6:]} 至 {need_end[:4]}-{need_end[4:6]}-{need_end[6:]}")
            print(f"\n💡 标准缓存文件名格式:")
            print(f"   000001.SZ.csv  (深市主板)")
            print(f"   000002.SZ.csv  (深市主板)")
            print(f"   600000.SH.csv  (沪市主板)")
            print(f"   300001.SZ.csv  (创业板)")
            print(f"   688001.SH.csv  (科创板)")

    print("\n" + "=" * 70)
def test_cache_and_calculation():
    """测试缓存读取和涨跌停计算（调试用）"""
    print("\n" + "=" * 70)
    print("【调试模式】测试单只股票")
    print("=" * 70)

    # 初始化
    ensure_cache_dir()
    pro = init_tushare()
    if pro is None:
        return

    # 获取日期
    dates = get_last_n_trade_dates(pro, 5)  # 只测试最近5天
    if not dates:
        return

    print(f"\n测试日期: {dates}")

    # 测试一只股票
    test_ts_code = "000001.SZ"
    test_code = "000001"
    test_name = "平安银行"

    # 获取数据范围
    stat_start_date = pd.Timestamp(dates[0])
    data_start_date = stat_start_date - timedelta(days=10)
    data_end_date = pd.Timestamp(dates[-1])

    print(f"\n正在获取 {test_name}({test_code}) 的数据...")

    # 获取数据
    df = fetch_daily_data(
        pro,
        test_ts_code,
        data_start_date.strftime("%Y%m%d"),
        data_end_date.strftime("%Y%m%d")
    )

    if df is None or df.empty:
        print("❌ 获取数据失败")
        return

    print(f"✅ 获取成功: {len(df)} 条数据")
    print(f"\n完整数据列名: {df.columns.tolist()}")
    print(f"\n原始数据前5行:")
    print(df.head())

    # 提取需要的列（与主程序相同的逻辑）
    df = df[['trade_date', 'close', 'pre_close']].copy()
    df.columns = ['日期', '收盘', '前收盘']
    # 正确处理整数格式的日期：20250930 -> 2025-09-30
    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
    df = df.sort_values('日期')

    print(f"\n提取并处理后的数据:")
    print(df.head(10))

    # 计算涨停
    board_type, limit_pct = detect_board_type(test_code)
    print(f"\n板块类型: {board_type}cm, 涨停幅度: {limit_pct}%")

    # 说明取整规则
    if board_type == "30":
        print(f"   价格计算规则: 向下取整（去尾）")
    else:
        print(f"   价格计算规则: 四舍五入")

    # 手动计算一个示例
    if len(df) > 0:
        sample_row = df.iloc[-1]
        sample_date = sample_row['日期']
        sample_close = sample_row['收盘']
        sample_pre_close = sample_row['前收盘']

        print(f"\n🔍 详细计算示例 ({sample_date.strftime('%Y-%m-%d')}):")
        print(f"   股票代码: {test_code}")
        print(f"   板块类型: {board_type}cm")
        print(f"   涨跌幅限制: ±{limit_pct}%")
        print(f"   前收盘价: {sample_pre_close}")
        print(f"   今收盘价: {sample_close}")

        # 根据板块类型选择取整方式
        rounding_mode = "floor" if board_type == "30" else "round"

        # 计算涨停价
        up_limit = round_price(sample_pre_close * (1 + limit_pct / 100), rounding_mode)
        down_limit = round_price(sample_pre_close * (1 - limit_pct / 100), rounding_mode)

        print(f"   理论涨停价: {up_limit} ({rounding_mode})")
        print(f"   理论跌停价: {down_limit} ({rounding_mode})")
        print(f"   实际收盘价: {round_price(sample_close, rounding_mode)}")

        diff_up = abs(round_price(sample_close, rounding_mode) - up_limit)
        diff_down = abs(round_price(sample_close, rounding_mode) - down_limit)

        print(f"   与涨停价差: {diff_up} ({'✅涨停' if diff_up == Decimal('0.00') else '❌非涨停'})")
        print(f"   与跌停价差: {diff_down} ({'✅跌停' if diff_down == Decimal('0.00') else '❌非跌停'})")

    df['涨停'] = df.apply(
        lambda x: is_limit_price(x['收盘'], x['前收盘'], limit_pct, board_type, True),
        axis=1
    )
    df['跌停'] = df.apply(
        lambda x: is_limit_price(x['收盘'], x['前收盘'], limit_pct, board_type, False),
        axis=1
    )

    print(f"\n涨跌停计算结果:")
    print(df[['日期', '收盘', '前收盘', '涨停', '跌停']].tail(10))

    # 统计涨跌停
    zhangting_count = df['涨停'].sum()
    dieting_count = df['跌停'].sum()

    print(f"\n统计结果:")
    print(f"  涨停天数: {zhangting_count}")
    print(f"  跌停天数: {dieting_count}")

    if zhangting_count == 0 and dieting_count == 0:
        print("\n⚠️  该股票近期无涨跌停")
        print("   这可能是正常的（大盘股通常很少涨跌停）")
        print("\n💡 建议:")
        print("   可以修改代码测试其他股票，例如:")
        print("   - 小盘股更容易涨跌停")
        print("   - 查看9月25日前后的新闻，选择热门股票测试")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print(" " * 18 + "涨跌停板统计程序")
    print(" " * 17 + "Tushare版本 v5.0")
    print("=" * 70)

    print("\n本版本特点:")
    print("  ✅ 使用Tushare Pro接口（稳定可靠）")
    print("  ✅ 自动缓存行情数据到本地")
    print("  ✅ 正确计算连板（获取足够的历史数据）")
    print("  ✅ 只统计: 60/00/3/688/8/9开头的股票")
    print("  ✅ 板块定义: 10cm(60/00) 20cm(3/688) 30cm(8/9)")
    print("  ✅ 价格计算: 10cm/20cm四舍五入, 30cm向下取整")
    print("  ✅ 11个指标: 10cm连板+20cm+30cm+跌停")
    print(f"  ✅ 智能速率限制: {API_CALLS_PER_MINUTE}次/分钟")
    print("  ✅ 包含主要指数数据: 上证指数、深证成指、创业板指、北证50")

    print("\n💡 提示:")
    print("  如果结果全为0，请运行:")
    print("  python limit_board_stats.py --check-cache  # 检查缓存日期范围")
    print("  python limit_board_stats.py --test         # 调试单只股票")

    # 初始化
    ensure_cache_dir()
    pro = init_tushare()
    if pro is None:
        return

    # 1. 获取交易日
    print("\n" + "=" * 70)
    print("【步骤1】获取交易日历")
    print("-" * 70)
    dates = get_last_n_trade_dates(pro, HISTORY_DAYS)
    if not dates:
        print("❌ 无法获取交易日")
        return
    print(f"✅ {dates[0]} 至 {dates[-1]}")

    # 2. 获取股票列表
    print("\n" + "=" * 70)
    print("【步骤2】获取股票列表")
    print("-" * 70)
    stocks = get_all_stocks(pro)
    if not stocks:
        print("❌ 无法获取股票列表")
        return

    # 3. 计算统计
    print("\n" + "=" * 70)
    print("【步骤3】计算涨跌停统计")
    print("-" * 70)
    result, detail_result = compute_limit_boards(pro, dates, stocks)

    # 4. 获取主要指数数据
    print("\n" + "=" * 70)
    print("【步骤4】获取主要指数数据")
    print("-" * 70)
    index_data = get_main_indexes_data(pro, dates)

    # 5. 生成涨跌停统计DataFrame
    print("\n" + "=" * 70)
    print("【步骤5】生成统计表格")
    print("-" * 70)

    # 创建涨跌停统计数据
    df = pd.DataFrame(index=ROW_NAMES)
    for date in dates:
        counts = result.get(date, {})
        df[date] = [counts.get(name, 0) for name in ROW_NAMES]

    # 6. 合并指数数据和涨跌停数据到同一个CSV
    if index_data:
        print(f"\n💾 保存合并数据到: {OUTPUT_FILE}")
        # 创建包含指数涨跌幅和涨跌停数据的DataFrame
        combined_data = {}

        # 添加涨跌停数据
        for date in dates:
            combined_data[date] = {}
            # 添加涨跌停统计数据
            for indicator in ROW_NAMES:
                combined_data[date][indicator] = result[date].get(indicator, 0)

            # 添加指数涨跌幅数据
            date_obj = pd.Timestamp(date)
            for ts_code, index_df in index_data.items():
                # 查找对应日期的数据
                day_data = index_df[index_df['trade_date'] == date_obj]
                if not day_data.empty:
                    combined_data[date][f"{MAIN_INDEXES[ts_code]}_涨跌幅(%)"] = day_data['pct_chg'].iloc[0]
                else:
                    combined_data[date][f"{MAIN_INDEXES[ts_code]}_涨跌幅(%)"] = None

        # 创建合并后的DataFrame
        combined_df = pd.DataFrame.from_dict(combined_data, orient='index')

        # 重新排列列顺序：先放涨跌停数据，再放指数数据
        columns_order = ROW_NAMES.copy()
        for ts_code, name in MAIN_INDEXES.items():
            columns_order.append(f"{name}_涨跌幅(%)")

        # 确保所有列都存在
        final_columns = [col for col in columns_order if col in combined_df.columns]
        combined_df = combined_df[final_columns]

        combined_df.to_csv(OUTPUT_FILE, encoding="utf-8-sig")
        print("✅ 合并数据已保存")
    else:
        print("⚠️  指数数据获取失败，只保存涨跌停数据")
        df.to_csv(OUTPUT_FILE, encoding="utf-8-sig")
        print("✅ 涨跌停数据已保存")

    # 7. 保存明细文件
    print(f"\n💾 保存涨跌停明细到: {DETAIL_FILE}")
    with open(DETAIL_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 30 + "涨跌停板明细清单\n")
        f.write("=" * 80 + "\n\n")

        for date in dates:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"日期: {date}\n")
            f.write(f"{'=' * 80}\n")

            total_count = sum(result[date].values())
            f.write(f"涨跌停总数: {total_count}\n\n")

            # 按指标输出
            for indicator in ROW_NAMES:
                count = result[date].get(indicator, 0)
                stocks_list = detail_result[date].get(indicator, [])

                if count > 0:
                    f.write(f"\n【{indicator}】共 {count} 只\n")
                    f.write("-" * 80 + "\n")

                    # 按股票代码排序
                    stocks_list_sorted = sorted(stocks_list, key=lambda x: x[1])

                    for idx, (ts_code, code, name) in enumerate(stocks_list_sorted, 1):
                        f.write(f"  {idx:3d}. {code:8s}  {name:20s}  ({ts_code})\n")

            f.write("\n")

    print("✅ 已保存明细文件")

    # 8. 显示结果
    print("\n" + "=" * 70)
    print(" " * 28 + "统计结果")
    print("=" * 70)
    if index_data:
        # 显示合并后的数据
        print(combined_df.to_string())
    else:
        # 只显示涨跌停数据
        print(df.to_string())

    # 9. 数据检查
    print("\n" + "=" * 70)
    print(" " * 28 + "数据检查")
    print("=" * 70)

    all_zero = True
    total_limit_boards = 0
    for date in dates:
        total = sum(result[date].values())
        total_limit_boards += total
        if total > 0:
            all_zero = False
            # 显示详细分类
            zhangting = result[date]['首板'] + result[date]['2连板'] + result[date]['3连板'] + \
                        result[date]['4连板'] + result[date]['5连板'] + result[date]['6连板及以上'] + \
                        result[date]['20cm首板'] + result[date]['20cm连板'] + \
                        result[date]['30cm首板'] + result[date]['30cm连板']
            dieting = result[date]['跌停板']
            print(f"✅ {date}: {total:4d} 个 (涨停: {zhangting:3d}, 跌停: {dieting:3d})")
        else:
            print(f"⚠️  {date}: {total:4d} 个涨跌停板")

    if all_zero:
        print("\n⚠️  所有日期统计为0！")
        print("\n可能的原因:")
        print("  1. 缓存文件的日期范围与程序需要的不匹配")
        print("  2. 数据获取失败 - 检查网络连接和Tushare Token")
        print("  3. 这段时间确实没有涨跌停（极少见）")
        print("\n建议操作:")
        print("  1. 检查缓存: python limit_board_stats.py --check-cache")
        print("  2. 调试模式: python limit_board_stats.py --test")
        print("  3. 删除缓存重新获取: rm -rf cache_data && python limit_board_stats.py")
    else:
        print(f"\n✅ 数据正常！")
        print(f"   统计期间共 {total_limit_boards} 个涨跌停板")
        print(f"   日均 {total_limit_boards / len(dates):.1f} 个")

    print("\n" + "=" * 70)
    print(f"✅ 完成！")
    print(f"   统计文件: {OUTPUT_FILE}")
    print(f"   明细文件: {DETAIL_FILE}")
    print(f"   缓存目录: {CACHE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_cache_and_calculation()
        elif sys.argv[1] == '--check-cache':
            check_cache_status()
        else:
            print("用法:")
            print("  python limit_board_stats.py           # 正常运行")
            print("  python limit_board_stats.py --test    # 调试单只股票")
            print("  python limit_board_stats.py --check-cache  # 检查缓存状态")
    else:
        main()
