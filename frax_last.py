import os
import time
import requests
from tqdm import tqdm
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import logging
import psutil
import subprocess
import re
from bs4 import BeautifulSoup
from itertools import combinations


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Настройки
MIN_APR = 8  # Порог для ставки Lend APR
BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"
CHAT_IDS = [6192278046, 306507209]
CHECK_INTERVAL = 60 * 60  # Интервал проверки в секундах (1 час)
MAX_PAIRS = 100  # Максимальное количество пар
MAX_ITERATION_TIME = 1200  # Максимальное время на итерацию (20 минут)
BLACKLISTED_PAIRS = set()


@dataclass
class InterestRateParams:
    max_full_util_rate: float = 3.164468e-8
    min_full_util_rate: float = 1.580586e-9
    zero_util_rate: float = 1.584231e-10
    rate_half_life: float = 172800.0
    max_target_util: float = 0.85
    min_target_util: float = 0.75
    vertex_util: float = 0.875
    vertex_rate_percent: float = 0.36
    util_precision: float = 1e5
    rate_precision: float = 1e18


@dataclass
class TimeWeightedInterestRateParams:
    protocol_fee: float = 0.1
    min_apr: float = 0.005
    max_apr: float = 100.0131
    min_utilization: float = 0.75
    max_utilization: float = 0.85
    rate_half_life_days: float = 0.5
    util_precision: float = 1e5
    rate_precision: float = 1e18


class VariableInterestRate:
    def __init__(self, params: InterestRateParams, suffix: str = "[0.5 0.2@.875 5-10k] 2 days (.75-.85)"):
        self.params = params
        self.suffix = suffix
        self.seconds_per_year = 365.24 * 24 * 3600

    def calculate_old_full_utilization_interest(self, old_lend_apr: float, old_utilization: float) -> float:
        old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
        old_borrow_rate_per_sec = old_borrow_apr / (self.seconds_per_year * 100)
        term = (
                       old_borrow_rate_per_sec - self.params.zero_util_rate) * self.params.vertex_util / old_utilization if old_utilization != 0 else 0
        vertex_interest = term + self.params.zero_util_rate
        full_utilization_interest = ((
                                             vertex_interest - self.params.zero_util_rate) / self.params.vertex_rate_percent) + self.params.zero_util_rate
        return full_utilization_interest

    def get_full_utilization_interest(self, delta_time: float, utilization: float,
                                      full_utilization_interest: float) -> float:
        if utilization < self.params.min_target_util:
            delta_utilization = ((
                                         self.params.min_target_util - utilization) * self.params.rate_precision) / self.params.min_target_util
            decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
            new_full_utilization_interest = (full_utilization_interest * (
                    self.params.rate_half_life * 1e36)) / decay_growth
        elif utilization > self.params.max_target_util:
            delta_utilization = ((utilization - self.params.max_target_util) * self.params.rate_precision) / (
                    self.params.util_precision - self.params.max_target_util)
            decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
            new_full_utilization_interest = (full_utilization_interest * decay_growth) / (
                    self.params.rate_half_life * 1e36)
        else:
            new_full_utilization_interest = full_utilization_interest
        new_full_utilization_interest = min(new_full_utilization_interest, self.params.max_full_util_rate)
        new_full_utilization_interest = max(new_full_utilization_interest, self.params.min_full_util_rate)
        return new_full_utilization_interest

    def get_new_rate(self, delta_time: float, utilization: float, old_full_utilization_interest: float) -> Tuple[
        float, float]:
        new_full_utilization_interest = self.get_full_utilization_interest(delta_time, utilization,
                                                                           old_full_utilization_interest)
        vertex_interest = (((
                                    new_full_utilization_interest - self.params.zero_util_rate) * self.params.vertex_rate_percent) + self.params.zero_util_rate)
        if utilization < self.params.vertex_util:
            new_rate_per_sec = (self.params.zero_util_rate + (
                    utilization * (vertex_interest - self.params.zero_util_rate)) / self.params.vertex_util)
        else:
            new_rate_per_sec = (vertex_interest + (
                    (utilization - self.params.vertex_util) * (new_full_utilization_interest - vertex_interest)) / (
                                        1.0 - self.params.vertex_util))
        return new_rate_per_sec, new_full_utilization_interest


class TimeWeightedVariableInterestRate:
    def __init__(self, params: TimeWeightedInterestRateParams):
        self.params = params
        self.seconds_per_year = 365.24 * 24 * 3600
        self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600

    def calculate_rate_per_sec(self, apr: float) -> float:
        return apr / 100 / self.seconds_per_year

    def calculate_new_full_utilization_rate(self, old_utilization: float, old_lend_apr: float,
                                            delta_time: float) -> float:
        old_rate_per_sec = self.calculate_rate_per_sec(old_lend_apr)
        old_full_util_rate = old_rate_per_sec / (
                old_utilization * (1 - self.params.protocol_fee)) if old_utilization != 0 else 0

        if old_utilization < self.params.min_utilization:
            delta_utilization = ((self.params.min_utilization - old_utilization) *
                                 self.params.rate_precision) / self.params.min_utilization
            decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
            new_full_util_rate = (old_full_util_rate * (self.rate_half_life_secs * 1e36)) / decay_growth
        elif old_utilization > self.params.max_utilization:
            delta_utilization = ((old_utilization - self.params.max_utilization) *
                                 self.params.rate_precision) / (
                                        self.params.util_precision - self.params.max_utilization)
            decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
            new_full_util_rate = (old_full_util_rate * decay_growth) / (self.rate_half_life_secs * 1e36)
        else:
            new_full_util_rate = old_full_util_rate

        min_rate_per_sec = self.calculate_rate_per_sec(self.params.min_apr)
        max_rate_per_sec = self.calculate_rate_per_sec(self.params.max_apr)
        new_full_util_rate = min(max(new_full_util_rate, min_rate_per_sec), max_rate_per_sec)
        return new_full_util_rate

    def get_new_lend_apr(self, delta_time: float, current_utilization: float, old_lend_apr: float,
                         old_utilization: float) -> float:
        full_utilization_rate = self.calculate_new_full_utilization_rate(
            old_utilization, old_lend_apr, delta_time)
        lend_rate_per_sec = full_utilization_rate * current_utilization * (
                1 - self.params.protocol_fee) if current_utilization != 0 else 0
        lend_apr = lend_rate_per_sec * self.seconds_per_year * 100
        return lend_apr


def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
    if amount_str == "N/A" or not amount_str:
        logger.warning(f"Невозможно распарсить сумму: {amount_str}, возвращается 0.0")
        return 0.0
    try:
        cleaned = re.sub(r'[^\d.]', '', amount_str)
        value = float(cleaned)
        if 'm' in amount_str.lower():
            return value * 1_000_000
        elif 'k' in amount_str.lower():
            return value * 1_000
        return value
    except ValueError:
        logger.error(f"Ошибка парсинга суммы: {amount_str}, возвращается 0.0")
        return 0.0


def get_max_investment_for_v2_utilization(data: Dict) -> float:
    """
    Рассчитывает максимальное вложение для Variable V2, чтобы new_utilization > 0.76.
    """
    if data.get("Rate Type") != "Variable V2":
        return float('inf')
    available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
    reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
    max_investment = (0.24 * reserve_size - available_liquidity) / 0.76
    return max_investment if max_investment > 0 else 0


def calculate_profit_for_project(data: Dict, investment: float, v1_model, v2_model, delta_time: float = 86400.0,
                                 enforce_v2_utilization: bool = False) -> Tuple[
    Optional[float], Optional[float], Optional[float]]:
    """
    Рассчитывает дневную прибыль, новую ставку и утилизацию для заданного вложения.
    """
    try:
        lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
        utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
        lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
        utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
        available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
        reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
        rate_type = data.get("Rate Type", "N/A")
        logger.debug(
            f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, "
            f"Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}"
        )
    except Exception as e:
        logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
        return None, None, None

    # Фильтрация
    if lend_apr <= MIN_APR:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_APR}%)")
        return None, None, None
    if utilization >= 1.01:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization * 100}% >= 101%)")
        return None, None, None
    if reserve_size == 0:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
        return None, None, None

    # Проверка начальной утилизации для Variable V2 в варианте 2
    if enforce_v2_utilization and rate_type == "Variable V2" and utilization < 0.76:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Initial Utilization={utilization * 100}% < 76%)")
        return None, None, None

    # Расчет новой утилизации
    new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
    logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")

    # Проверка new_utilization > 0.76 для Variable V2
    if enforce_v2_utilization and rate_type == "Variable V2" and new_utilization <= 0.76:
        logger.debug(f"Пропущено (V2): new_utilization={new_utilization:.4f} <= 0.76")
        return None, None, None

    # Расчет новой ставки и прибыли
    seconds_per_year = 365.24 * 24 * 3600
    if rate_type == "Variable V2":
        old_full_utilization_interest = v2_model.calculate_old_full_utilization_interest(lend_apr, utilization)
        new_rate_per_sec, _ = v2_model.get_new_rate(delta_time, new_utilization, old_full_utilization_interest)
        new_lend_apr = new_rate_per_sec * seconds_per_year * new_utilization * 100
        daily_profit = (investment * new_lend_apr / 100) / 365.24
    elif rate_type == "Variable V1":
        new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
        daily_profit = (investment * new_lend_apr / 100) / 365.24
    else:
        logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
        return None, None, None

    return daily_profit, new_lend_apr, new_utilization

def calculate_optimal_investment(projects_data: List[Dict], v1_model, v2_model, delta_time: float = 86400.0, max_total_investment: float = 200000) -> List[Dict]:
    """
    Распределяет капитал между 1–4 проектами с учетом сравнительной доходности, возвращая результаты для двух вариантов:
    1. Без ограничения на утилизацию.
    2. Для Variable V2: начальная утилизация >= 76%, new_utilization > 0.76.
    Args:
        projects_data: Список словарей с данными по проектам.
        v1_model: Модель для Variable V1.
        v2_model: Модель для Variable V2.
        delta_time: Временной интервал в секундах.
        max_total_investment: Максимальная сумма для инвестиций.
    Returns:
        List[Dict]: Список словарей с результатами для каждой пары в двух вариантах.
    """
    # Фильтрация проектов
    valid_projects_default = []
    valid_projects_constrained = []
    for data in projects_data:
        profit, lend_apr, utilization = calculate_profit_for_project(data, 0, v1_model, v2_model, delta_time, enforce_v2_utilization=False)
        if profit is not None:
            valid_projects_default.append(data)
        profit, lend_apr, utilization = calculate_profit_for_project(data, 0, v1_model, v2_model, delta_time, enforce_v2_utilization=True)
        if profit is not None:
            valid_projects_constrained.append(data)

    # Логирование валидных проектов
    logger.info(f"Valid projects for default mode: {len(valid_projects_default)}")
    for project in valid_projects_default:
        logger.info(f"Project: {project['Link']}, Lend APR: {project.get('Lend APR', 'N/A')}, Utilization: {project.get('Utilization Rate', 'N/A')}")
    logger.info(f"Valid projects for constrained mode: {len(valid_projects_constrained)}")

    if not valid_projects_default:
        logger.info("Ни один проект не прошел фильтрацию для варианта 1")
        return []

    step = 5000  # Уменьшенный шаг для более точного распределения
    results = []

    def distribute_capital(projects: List[Dict], enforce_v2_utilization: bool) -> Tuple[float, Dict]:
        best_total_profit = 0
        best_investments = {}
        for num_projects in range(1, min(5, len(projects) + 1)):
            for combo in combinations(projects, num_projects):
                current_investments = {data['Link']: 0 for data in combo}
                remaining_capital = max_total_investment
                total_profit = 0
                while remaining_capital >= step:
                    best_additional_profit = 0
                    best_project = None
                    best_new_investment = None
                    for data in combo:
                        current_investment = current_investments[data['Link']]
                        new_investment = current_investment + step
                        max_investment = get_max_investment_for_v2_utilization(data) if enforce_v2_utilization else max_total_investment
                        max_investment = min(max_investment, parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True) * 2)  # Ограничение по резерву
                        if new_investment > max_investment or new_investment > remaining_capital:
                            continue
                        profit, lend_apr, utilization = calculate_profit_for_project(data, new_investment, v1_model, v2_model, delta_time, enforce_v2_utilization)
                        if profit is None:
                            continue
                        current_profit, _, _ = calculate_profit_for_project(data, current_investment, v1_model, v2_model, delta_time, enforce_v2_utilization) or (0, 0, 0)
                        additional_profit = profit - current_profit
                        logger.debug(f"Project: {data['Link']}, Investment: {new_investment}, Additional Profit: {additional_profit}, New APR: {lend_apr}, New Utilization: {utilization}")
                        if additional_profit > best_additional_profit:
                            best_additional_profit = additional_profit
                            best_project = data
                            best_new_investment = new_investment
                    if best_project is None:
                        break
                    current_investments[best_project['Link']] = best_new_investment
                    remaining_capital -= step
                    total_profit += best_additional_profit
                if total_profit > best_total_profit:
                    best_total_profit = total_profit
                    best_investments = current_investments
        return best_total_profit, best_investments

    total_profit_default, investments_default = distribute_capital(valid_projects_default, enforce_v2_utilization=False)
    total_profit_constrained, investments_constrained = distribute_capital(valid_projects_constrained, enforce_v2_utilization=True)

    # Формирование результатов для всех пар
    for data in projects_data:
        link = data['Link']
        default_result = investments_default.get(link, {'investment': 0, 'daily_profit': 0, 'lend_apr': 0, 'utilization': 0})
        constrained_result = investments_constrained.get(link, {'investment': 0, 'daily_profit': 0, 'lend_apr': 0, 'utilization': 0})
        results.append({
            'Link': link,
            'default': {
                'optimal_investment': default_result['investment'],
                'max_profit': default_result['daily_profit'],
                'optimal_lend_apr': default_result['lend_apr'],
                'optimal_utilization': default_result['utilization'],
                'total_profit': total_profit_default
            },
            'v2_utilization_constrained': {
                'optimal_investment': constrained_result['investment'],
                'max_profit': constrained_result['daily_profit'],
                'optimal_lend_apr': constrained_result['lend_apr'],
                'optimal_utilization': constrained_result['utilization'],
                'total_profit': total_profit_constrained
            }
        })

    logger.info(f"Результаты распределения: {results}")
    return results

# def calculate_optimal_investment(projects_data: List[Dict], v1_model, v2_model, delta_time: float = 86400.0,
#                                  max_total_investment: float = 200000) -> List[Dict]:
#     """
#     Распределяет капитал между 1–4 проектами с учетом сравнительной доходности, возвращая результаты для двух вариантов:
#     1. Без ограничения на утилизацию.
#     2. Для Variable V2: начальная утилизация >= 76%, new_utilization > 0.76.Args:
#     projects_data: Список словарей с данными по проектам.
#     v1_model: Модель для Variable V1.
#     v2_model: Модель для Variable V2.
#     delta_time: Временной интервал в секундах.
#     max_total_investment: Максимальная сумма для инвестиций.
#
# Returns:
#     List[Dict]: Список словарей с результатами для каждой пары в двух вариантах.
# """
# # Фильтрация проектов
# valid_projects_default = []
# valid_projects_constrained = []
# for data in projects_data:
#     profit, lend_apr, utilization = calculate_profit_for_project(data, 0, v1_model, v2_model, delta_time,
#                                                                  enforce_v2_utilization=False)
#     if profit is not None:
#         valid_projects_default.append(data)
#     profit, lend_apr, utilization = calculate_profit_for_project(data, 0, v1_model, v2_model, delta_time,
#                                                                  enforce_v2_utilization=True)
#     if profit is not None:
#         valid_projects_constrained.append(data)
#
# if not valid_projects_default:
#     logger.info("Ни один проект не прошел фильтрацию для варианта 1")
#     return []
#
# step = 5000
# results = []
#
# def distribute_capital(projects: List[Dict], enforce_v2_utilization: bool) -> Tuple[float, Dict]:
#     best_total_profit = 0
#     best_investments = {}
#
#     # Перебираем комбинации от 1 до 4 проектов
#     for num_projects in range(1, min(5, len(projects) + 1)):
#         for combo in combinations(projects, num_projects):
#             current_investments = {data['Link']: 0 for data in combo}
#             remaining_capital = max_total_investment
#             total_profit = 0
#
#             while remaining_capital >= step:
#                 best_additional_profit = 0
#                 best_project = None
#                 best_new_investment = None
#
#                 # Проверяем прирост прибыли
#                 for data in combo:
#                     current_investment = current_investments[data['Link']]
#                     new_investment = current_investment + step
#                     max_investment = get_max_investment_for_v2_utilization(
#                         data) if enforce_v2_utilization else max_total_investment
#                     if new_investment > max_investment:
#                         continue
#                     profit, lend_apr, utilization = calculate_profit_for_project(data, new_investment, v1_model,
#                                                                                  v2_model, delta_time,
#                                                                                  enforce_v2_utilization)
#                     if profit is None:
#                         continue
#                     current_profit, _, _ = calculate_profit_for_project(data, current_investment, v1_model,
#                                                                         v2_model, delta_time,
#                                                                         enforce_v2_utilization) or (0, 0, 0)
#                     additional_profit = profit - current_profit
#                     if additional_profit > best_additional_profit:
#                         best_additional_profit = additional_profit
#                         best_project = data
#                         best_new_investment = new_investment
#
#                 if best_project is None:
#                     break
#
#                 current_investments[best_project['Link']] = best_new_investment
#                 remaining_capital -= step
#                 total_profit += best_additional_profit
#
#             # Формируем результат для комбинации
#             combo_investments = {}
#             for data in combo:
#                 investment = current_investments[data['Link']]
#                 if investment > 0:
#                     profit, lend_apr, utilization = calculate_profit_for_project(data, investment, v1_model,
#                                                                                  v2_model, delta_time,
#                                                                                  enforce_v2_utilization)
#                     combo_investments[data['Link']] = {
#                         'investment': investment,
#                         'daily_profit': profit,
#                         'lend_apr': lend_apr,
#                         'utilization': utilization
#                     }
#
#             if total_profit > best_total_profit:
#                 best_total_profit = total_profit
#                 best_investments = combo_investments
#
#     return best_total_profit, best_investments





def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            payload = {
                "chat_id": chat_id,
                "text": message[:4096]
            }
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
            else:
                logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
        except requests.RequestException as e:
            logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {type(e).__name__}: {e}")


def kill_chromedriver():
    """Принудительно завершает все процессы chromedriver"""
    try:
        subprocess.run(['pkill', '-f', 'chromedriver'], check=True)
        logger.info("Все процессы chromedriver завершены")
    except subprocess.CalledProcessError:
        logger.info("Процессы chromedriver не найдены")
    except Exception as e:
        logger.error(f"Ошибка при завершении chromedriver: {type(e).__name__}: {e}")


def get_driver(chromedriver_path: str):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    )
    driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
    return driver


# def get_pair_links(driver, max_retries=3):
#     url = "https://facts.frax.finance/fraxlend/pairs"
#     logger.info(f"Попытка загрузки страницы: {url}")
#     for attempt in range(max_retries):
#         try:
#             response = requests.get(url, timeout=10)
#             logger.info(f"Статус HTTP-запроса к {url}: {response.status_code}")
#             driver.get(url)
#             WebDriverWait(driver, 10).until(
#                 EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']"))
#             )
#             elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']")
#             links = set()
#             for e in elems:
#                 href = e.get_attribute("href")
#                 if href and '/fraxlend/pairs/' in href:
#                     if not href.startswith('http'):
#                         href = f"https://facts.frax.finance{href}"
#                     if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
#                         links.add(href)
#             logger.info(f"Найдено ссылок на пары: {len(links)}")
#             return list(links)[:MAX_PAIRS]
#         except TimeoutException as e:
#             logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при ожидании элементов на {url}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error("Не удалось загрузить ссылки после всех попыток.")
#                 send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
#                 return []
#             time.sleep(2)
#         except Exception as e:
#             logger.error(f"Попытка {attempt + 1}/{max_retries}: Ошибка при получении ссылок: {type(e).__name__}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error("Не удалось загрузить ссылки после всех попыток.")
#                 send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
#                 return []
#             time.sleep(2)
#     return []

def get_pair_links(driver, max_retries=3):
    url = "https://facts.frax.finance/fraxlend/pairs"
    logger.info(f"Попытка загрузки страницы: {url}")
    for attempt in range(max_retries):
        try:
            # Pre-check website availability
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Сайт недоступен, статус: {response.status_code}")
                send_to_telegram(f"Ошибка: Сайт {url} недоступен, статус: {response.status_code}")
                time.sleep(2)
                continue

            logger.info(f"Статус HTTP-запроса к {url}: {response.status_code}")
            driver.get(url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']"))
            )
            elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']")
            links = set()
            for e in elems:
                href = e.get_attribute("href")
                if href and '/fraxlend/pairs/' in href:
                    if not href.startswith('http'):
                        href = f"https://facts.frax.finance{href}"
                    if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
                        links.add(href)
            logger.info(f"Найдено ссылок на пары: {len(links)}")
            return list(links)[:MAX_PAIRS]
        except TimeoutException as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при ожидании элементов на {url}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ссылки после всех попыток.")
                send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары из-за тайм-аута.")
                return []
            time.sleep(2)
        except WebDriverException as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: WebDriverException при загрузке {url}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ссылки после всех попыток.")
                send_to_telegram(f"Ошибка: Не удалось загрузить ссылки на пары из-за сбоя WebDriver: {e}")
                return []
            logger.info("Перезапуск WebDriver")
            try:
                driver.quit()
            except:
                pass
            kill_chromedriver()
            driver = get_driver(os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
            time.sleep(2)
        except Exception as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Ошибка при получении ссылок: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ссылки после всех попыток.")
                send_to_telegram(f"Ошибка: Не удалось загрузить ссылки на пары: {type(e).__name__}: {e}")
                return []
            time.sleep(2)
    return []

def fetch_metrics(driver, url, timeout=15, max_retries=2):
    logger.info(f"Загрузка страницы: {url}")
    if url in BLACKLISTED_PAIRS:
        logger.info(f"Пара {url} в черном списке, пропускаем")
        return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
                "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            process = psutil.Process()
            peak_memory = process.memory_info().rss / 1024 / 1024
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
            if "Frax Facts" not in driver.page_source:
                raise Exception("Страница не содержит ожидаемого содержимого 'Frax Facts'")
            with open(f"page_{url.split('/')[-1]}.html", "w") as f:
                f.write(driver.page_source)
            logger.info(f"Сохранён HTML страницы {url} для отладки")
            labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type",
                      "Collateral"]
            data = {"Link": url}
            for label in labels:
                try:
                    if label == "Collateral":
                        possible_labels = ["Collateral", "Collateral Token", "Collateral Asset"]
                        for possible_label in possible_labels:
                            try:
                                el = driver.find_element(By.XPATH,
                                                         f"//div[contains(text(), '{possible_label}')]/following-sibling::div")
                                data[label] = el.text.strip()
                                logger.info(f"Найдено значение для {possible_label}: {data[label]}")
                                break
                            except:
                                continue
                        else:
                            data[label] = "N/A"
                            logger.warning(f"Не удалось найти значение для Collateral на {url}")
                    else:
                        el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
                        data[label] = el.text.strip()
                    logger.info(f"Извлечено значение для {label}: {data[label]}")
                except Exception as e:
                    data[label] = "N/A"
                    logger.warning(f"Не удалось извлечь {label} для {url}: {type(e).__name__}: {e}")
            logger.info(f"Данные для {url}: {data}")
            logger.info(f"Время обработки {url}: {time.time() - start_time:.2f} секунд")
            peak_memory = max(peak_memory, process.memory_info().rss / 1024 / 1024)
            logger.info(f"Пиковая память во время обработки {url}: {peak_memory:.2f} MB")
            return data
        except TimeoutException as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при загрузке данных для {url}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Пара {url} не загрузилась после {max_retries} попыток, добавляем в черный список")
                BLACKLISTED_PAIRS.add(url)
                send_to_telegram(f"Пара {url} добавлена в черный список после {max_retries} неудачных попыток загрузки")
                return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
                        "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
            time.sleep(2)
        except Exception as e:
            logger.error(
                f"Попытка {attempt + 1}/{max_retries}: Ошибка при загрузке данных для {url}: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Пара {url} не загрузилась после {max_retries} попыток, добавляем в черный список")
                BLACKLISTED_PAIRS.add(url)
                send_to_telegram(f"Пара {url} добавлена в черный список после ошибки: {type(e).__name__}: {e}")
                return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
                        "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
            time.sleep(2)


def get_fraxlend_fxs_lower_bound(driver) -> float:
    url = "https://app.frax.finance/staking/overview"
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Fraxlend V1 FRAX/FXS')]"))
        )
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        target_text = re.compile(r"Fraxlend\s*V1\s*FRAX/FXS", re.IGNORECASE)
        fees_range = None
        for element in soup.find_all(string=target_text):
            parent = element.find_parent()
            if parent:
                for sibling in parent.find_all_next(string=True, limit=10):
                    if re.search(r'\d+\.\d+%\s*-\s*\d+\.\d+%', sibling):
                        fees_range = sibling.strip()
                        break
                if fees_range:
                    break
        if fees_range:
            match = re.search(r'(\d+\.\d+)%\s*-\s*\d+\.\d+%', fees_range)
            if match:
                return float(match.group(1))
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        return 0.0


def process_pairs():
    logger.info(
        f"Запуск функции process_pairs, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    send_to_telegram("Тест: Сервер запущен, начинаем парсинг")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1280,720')
    options.add_argument('--disable-extensions')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.binary_location = '/usr/bin/chromium'

    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    logger.info(f"Используемый путь к chromedriver: {chromedriver_path}")

    v1_params = TimeWeightedInterestRateParams()
    v2_params = InterestRateParams()
    v1_model = TimeWeightedVariableInterestRate(v1_params)
    v2_model = VariableInterestRate(v2_params)
    iteration_count = 0

    while True:
        iteration_start_time = time.time()
        iteration_count += 1
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

        processed_urls = set()
        logger.info(f"Сброс processed_urls для цикла #{iteration_count}")

        logger.info(f"Текущий черный список: {BLACKLISTED_PAIRS}")
        send_to_telegram(f"Начало цикла #{iteration_count}. Текущий черный список: {BLACKLISTED_PAIRS}")

        if iteration_count % 24 == 0:
            logger.info("Сброс черного списка")
            BLACKLISTED_PAIRS.clear()
            send_to_telegram("Черный список сброшен")

        try:
            logger.info(
                f"Начало парсинга пар, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            driver = None
            for attempt in range(3):
                try:
                    logger.info(f"Попытка {attempt + 1}/3: Инициализация WebDriver для списка пар")
                    driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
                    logger.info("WebDriver успешно инициализирован для списка пар")
                    break
                except Exception as e:
                    logger.error(f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver: {type(e).__name__}: {e}")
                    if attempt == 2:
                        logger.error("Не удалось инициализировать WebDriver после 3 попыток.")
                        send_to_telegram("Ошибка: Не удалось инициализировать WebDriver. Проверьте конфигурацию.")
                        return
                    time.sleep(2)

            pair_links = get_pair_links(driver)
            bonus = get_fraxlend_fxs_lower_bound(driver)

            logger.info(f"Bonus: {bonus}")
            logger.info(f"Полученные пары: {pair_links}")
            if driver:
                try:
                    driver.quit()
                except WebDriverException:
                    logger.info("Игнорируется WebDriverException при закрытии WebDriver для списка пар")
                kill_chromedriver()
                logger.info(
                    f"WebDriver закрыт после получения списка пар, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            logger.info(f"Найдено пар: {len(pair_links)}")
            if not pair_links:
                logger.warning("Список пар пуст. Пропускаем итерацию.")
                send_to_telegram("Предупреждение: Список пар пуст. Проверьте сайт или селектор.")
                time.sleep(CHECK_INTERVAL)
                continue

            projects_data = []
            skipped_pairs = []

            for url in tqdm(pair_links, desc="Обработка пар"):
                if time.time() - iteration_start_time > MAX_ITERATION_TIME:
                    logger.warning(
                        f"Превышено максимальное время итерации ({MAX_ITERATION_TIME} секунд). Пропускаем оставшиеся пары.")
                    send_to_telegram(
                        f"Превышено время итерации ({MAX_ITERATION_TIME} секунд). Пропущено {len(pair_links) - pair_links.index(url)} пар.")
                    break

                if url in processed_urls or url in BLACKLISTED_PAIRS:
                    logger.info(f"Пропущена пара (уже обработана или в черном списке): {url}")
                    skipped_pairs.append(url)
                    continue

                logger.info(f"Обработка пары: {url}")
                driver = None
                for attempt in range(3):
                    try:
                        logger.info(f"Попытка {attempt + 1}/3: Инициализация WebDriver для {url}")
                        driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
                        logger.info(f"WebDriver успешно инициализирован для {url}")
                        break
                    except Exception as e:
                        logger.error(
                            f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver для {url}: {type(e).__name__}: {e}")
                        if attempt == 2:
                            logger.error(f"Не удалось инициализировать WebDriver для {url} после 3 попыток.")
                            send_to_telegram(f"Ошибка: Не удалось инициализировать WebDriver для {url}.")
                            break
                        time.sleep(2)

                if driver:
                    data = fetch_metrics(driver, url)
                    logger.info(f"Полученные данные для {url}: {data}")
                    if all(data.get(label, "N/A") == "N/A" for label in
                           ["Available Liquidity", "Utilization Rate", "Lend APR", "Reserve Size"]):
                        logger.info(f"Пропущена пара из-за некорректных данных: {url}")
                        processed_urls.add(url)
                        send_to_telegram(
                            f"Пропущена пара {url} (Collateral: {data.get('Collateral', 'N/A')}) из-за некорректных данных: {data}")
                    else:
                        projects_data.append(data)
                        processed_urls.add(url)

                    peak_memory = max(peak_memory, psutil.Process().memory_info().rss / 1024 / 1024)
                    logger.info(
                        f"Память после обработки {url}: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
                    logger.info(f"Пиковая память в итерации: {peak_memory:.2f} MB")

                    try:
                        driver.quit()
                    except WebDriverException:
                        logger.info(f"Игнорируется WebDriverException при закрытии WebDriver для {url}")
                    kill_chromedriver()
                    logger.info(
                        f"WebDriver закрыт после обработки {url}, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            if skipped_pairs:
                send_to_telegram(f"Пропущенные пары в цикле #{iteration_count}: {', '.join(skipped_pairs)}")
#№№№№№

#
#pасчет оптимального распределения
            if projects_data:
                results = calculate_optimal_investment(projects_data, v1_model, v2_model)
                # Отправка сообщений для варианта 1 (default)
# Отправка сообщений для варианта 1 (default)
                send_to_telegram("=== Результаты для варианта 1 (без ограничений на утилизацию) ===")
                for result in results:
                    link = result['Link']
                    pair_data = next((data for data in projects_data if data['Link'] == link), {})
                    collateral = pair_data.get('Collateral', 'N/A')
                    pair_address = link.split('/')[-1]

                    optimal_investment = result['default']['optimal_investment']
                    max_profit = result['default']['max_profit']
                    optimal_lend_apr = result['default']['optimal_lend_apr']
                    optimal_utilization = result['default']['optimal_utilization']
                    total_profit = result['default']['total_profit']

                    if optimal_investment > 0:
                        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        b = ""
                        s = 0
                        if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
                            b = "+" + str(bonus)
                            s = (optimal_investment * bonus / 100) / 365.24
                            optimal_lend_apr += bonus
                            max_profit += s
                        send_to_telegram(
                            f"Вариант 1 (без ограничений на утилизацию)\n"
                            f"Пара: {collateral} ({link})\n"
                            f"Timestamp (UTC): {timestamp} +3 часа\n"
                            f"Старая Lend APR: {pair_data.get('Lend APR', 'N/A')}\n"
                            f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%{b}\n"
                            f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
                            f"Максимальный доход за 1 день: ${max_profit + s:.2f}\n"
                            f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
                            f"Available Liquidity: {pair_data.get('Available Liquidity', 'N/A')}\n"
                            f"Utilization Rate: {pair_data.get('Utilization Rate', 'N/A')}\n"
                            f"Borrow APR: {pair_data.get('Borrow APR', 'N/A')}\n"
                            f"Reserve Size: {pair_data.get('Reserve Size', 'N/A')}\n"
                            f"Rate Type: {pair_data.get('Rate Type', 'N/A')}\n"
                            f"Общая дневная прибыль (все пары): ${total_profit:.2f}"
                        )

                # Отправка сообщений для варианта 2 (v2_utilization_constrained)
                send_to_telegram("=== Результаты для варианта 2 (ограничение утилизации V2 > 76%) ===")
                for result in results:
                    link = result['Link']
                    pair_data = next((data for data in projects_data if data['Link'] == link), {})
                    collateral = pair_data.get('Collateral', 'N/A')
                    pair_address = link.split('/')[-1]

                    optimal_investment = result['v2_utilization_constrained']['optimal_investment']
                    max_profit = result['v2_utilization_constrained']['max_profit']
                    optimal_lend_apr = result['v2_utilization_constrained']['optimal_lend_apr']
                    optimal_utilization = result['v2_utilization_constrained']['optimal_utilization']
                    total_profit = result['v2_utilization_constrained']['total_profit']

                    if optimal_investment > 0:
                        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        b = ""
                        s = 0
                        if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
                            b = "+" + str(bonus)
                            s = (optimal_investment * bonus / 100) / 365.24
                            optimal_lend_apr += bonus
                            max_profit += s
                        send_to_telegram(
                            f"Вариант 2 (ограничение утилизации V2 > 76%)\n"
                            f"Пара: {collateral} ({link})\n"
                            f"Timestamp (UTC): {timestamp} +3 часа\n"
                            f"Старая Lend APR: {pair_data.get('Lend APR', 'N/A')}\n"
                            f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%{b}\n"
                            f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
                            f"Максимальный доход за 1 день: ${max_profit + s:.2f}\n"
                            f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
                            f"Available Liquidity: {pair_data.get('Available Liquidity', 'N/A')}\n"
                            f"Utilization Rate: {pair_data.get('Utilization Rate', 'N/A')}\n"
                            f"Borrow APR: {pair_data.get('Borrow APR', 'N/A')}\n"
                            f"Reserve Size: {pair_data.get('Reserve Size', 'N/A')}\n"
                            f"Rate Type: {pair_data.get('Rate Type', 'N/A')}\n"
                            f"Общая дневная прибыль (все пары): ${total_profit:.2f}"
                        )


        except Exception as e:
            logger.error(f"Ошибка обработки пар: {type(e).__name__}: {e}")
            send_to_telegram(f"Ошибка при обработке пар: {type(e).__name__}: {e}")

        finally:
            if driver:
                try:
                    driver.quit()
                except WebDriverException:
                    logger.info("Игнорируется WebDriverException при закрытии WebDriver в finally")
                kill_chromedriver()
                logger.info(
                    f"WebDriver закрыт в finally, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            elapsed_time = time.time() - iteration_start_time
            logger.info(f"Итерация завершена за {elapsed_time:.2f} секунд")
            logger.info(f"Пиковая память в итерации: {peak_memory:.2f} MB")
            logger.info(f"Ожидание {CHECK_INTERVAL} секунд перед следующей итерацией")
            remaining_time = CHECK_INTERVAL
            while remaining_time > 0:
                sleep_time = min(30, remaining_time)
                time.sleep(sleep_time)
                remaining_time -= sleep_time
                peak_memory = max(peak_memory, psutil.Process().memory_info().rss / 1024 / 1024)


def main():
    logger.info("Запуск Background Worker")
    send_to_telegram("Тест: Background Worker запущен")
    process_pairs()


if __name__ == "__main__":
    main()