from bs4 import BeautifulSoup
import os
import re
import time
import requests
from tqdm import tqdm
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import logging
import psutil
import subprocess
from urllib3.exceptions import NewConnectionError

MIN_LEND_APR_THRESHOLD = 15  # Пороговая ставка Lend APR для фильтрации (в процентах)


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Настройки Telegram
#BOT_TOKEN = "8207805821:AAGcdowmtfkoGTeOpBfFggIl56fQQEB4f4A"
BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"

CHAT_IDS = [6192278046, 306507209]
CHECK_INTERVAL = 60*60  # Интервал проверки в секундах (1 час)
MAX_PAIRS = 100  # Максимальное количество пар
MAX_ITERATION_TIME = 1200  # Максимальное время на итерацию (10 минут)

# Blacklist for problematic pairs
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

    def calculate_old_full_utilization_interest(self, old_lend_apr: float, old_utilization: float) -> float:
        seconds_per_year = 365.24 * 24 * 3600
        old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
        old_borrow_rate_per_sec = old_borrow_apr / (seconds_per_year * 100)
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
        return apr / 100

    def calculate_new_full_utilization_rate(self,
                                            old_utilization: float,
                                            old_lend_apr: float,
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

    def get_new_lend_apr(self,
                         delta_time: float,
                         current_utilization: float,
                         old_lend_apr: float,
                         old_utilization: float) -> float:
        full_utilization_rate = self.calculate_new_full_utilization_rate(
            old_utilization, old_lend_apr, delta_time)
        lend_rate_per_sec = full_utilization_rate * current_utilization * (
            1 - self.params.protocol_fee) if current_utilization != 0 else 0
        lend_apr = lend_rate_per_sec * 100
        return lend_apr

def get_pair_links(driver, max_retries=3):
    url = "https://facts.frax.finance/fraxlend/pairs"
    logger.info(f"Попытка загрузки страницы: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"Статус HTTP-запроса к {url}: {response.status_code}")
            driver.get(url)
            WebDriverWait(driver, 10).until(
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
                send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
                return []
            time.sleep(2)

        except Exception as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Ошибка при получении ссылок: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ссылки после всех попыток.")
                send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
                return []
            time.sleep(2)

    return []

def fetch_fraxlend_v1_frax_fxs_rate(driver, max_retries=3):
    url = "https://app.frax.finance/staking/overview"
    logger.info(f"Попытка загрузки страницы для получения ставки Fraxlend V1 FRAX/FXS: {url}")

    for attempt in range(max_retries):
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Fraxlend V1 FRAX/FXS')]"))
            )
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Ищем элемент, содержащий текст "Fraxlend V1 FRAX/FXS"
            fraxlend_element = soup.find(string=re.compile('Fraxlend V1 FRAX/FXS'))
            if not fraxlend_element:
                logger.error("Не найден элемент с текстом 'Fraxlend V1 FRAX/FXS'")
                return 0.0

            # Ищем родительский элемент, содержащий диапазон процентов
            parent = fraxlend_element.find_parent()
            rate_text = None
            for sibling in parent.find_next_siblings():
                if '[FXS:' in sibling.text:
                    rate_text = sibling.text
                    break

            if not rate_text:
                logger.error("Не найден диапазон процентов для Fraxlend V1 FRAX/FXS")
                return 0.0

            # Извлекаем нижнюю границу из текста вида [FXS: X% - Y%]
            match = re.search(r'\[FXS:\s*([\d.]+)%\s*-\s*([\d.]+)%\]', rate_text)
            if match:
                lower_bound = float(match.group(1))
                logger.info(f"Извлечена нижняя граница ставки: {lower_bound}%")
                return lower_bound
            else:
                logger.error(f"Не удалось распарсить диапазон процентов из текста: {rate_text}")
                return 0.0

        except TimeoutException as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при загрузке {url}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ставку после всех попыток")
                send_to_telegram("Ошибка: Не удалось загрузить ставку Fraxlend V1 FRAX/FXS")
                return 0.0
            time.sleep(2)

        except Exception as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Ошибка при получении ставки: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ставку после всех попыток")
                send_to_telegram(f"Ошибка при получении ставки Fraxlend V1 FRAX/FXS: {type(e).__name__}: {e}")
                return 0.0
            time.sleep(2)

    return 0.0


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

            # Сохраняем HTML для отладки
            with open(f"page_{url.split('/')[-1]}.html", "w") as f:
                f.write(driver.page_source)
            logger.info(f"Сохранён HTML страницы {url} для отладки")

            labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type",
                      "Collateral"]
            data = {"Link": url}
            for label in labels:
                try:
                    # Пробуем разные варианты написания для Collateral
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

def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
    if amount_str == "N/A" or not amount_str:
        logger.warning(f"Невозможно распарсить сумму: {amount_str}, возвращается 0.0")
        return 0.0
    try:
        cleaned = re.sub(r'[^\d.]', '', amount_str)
        value = float(cleaned)
        if is_reserve_size and 'm' in amount_str.lower():
            return value * 1_000_000
        elif 'k' in amount_str.lower():
            return value * 1_000
        return value
    except ValueError:
        logger.error(f"Ошибка парсинга суммы: {amount_str}, возвращается 0.0")
        return 0.0

def calculate_optimal_investment(data, v1_model, v2_model, driver, delta_time=86400.0):
    try:
        lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
        utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
        lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
        utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
        available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
        reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
        rate_type = data.get("Rate Type", "N/A")
        pair_address = data.get("Link", "").split("/")[-1].lower()  # Приводим к нижнему регистру

        # Для пары 0xDbe88DBAc39263c47629ebbA02b3eF4cf0752A72 добавляем ставку Fraxlend V1 FRAX/FXS
        fraxlend_rate = 0.0
        if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":  # Сравниваем в нижнем регистре
            fraxlend_rate = fetch_fraxlend_v1_frax_fxs_rate(driver)
            lend_apr += fraxlend_rate
            logger.info(f"Добавлена ставка Fraxlend V1 FRAX/FXS ({fraxlend_rate:.2f}%) к Lend APR для пары {pair_address}. Новый Lend APR: {lend_apr:.2f}%")
            if fraxlend_rate == 0.0:
                logger.warning(f"Ставка Fraxlend V1 FRAX/FXS не получена для пары {pair_address}")
                send_to_telegram(f"Предупреждение: Не удалось получить ставку Fraxlend V1 FRAX/FXS для пары {pair_address}")

        #logger.info(
            #f"Распарсенные данные: Lend APR={lend_apr:.2f}, Utilization={utilization:.4f}, Available Liquidity={available _(available_liquidity), Reserve Size={reserve_size}, Rate Type={rate_type}, Fraxlend Rate={fraxlend_rate:.2f}%")
    except Exception as e:
        logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
        send_to_telegram(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
        return None, None, None, None, 0.0

    # Фильтрация: Lend APR > MIN_LEND_APR_THRESHOLD, Utilization Rate < 101%, Reserve Size != 0
    if lend_apr <= MIN_LEND_APR_THRESHOLD:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr:.2f} <= {MIN_LEND_APR_THRESHOLD}%)")
        return None, None, None, None, fraxlend_rate
    if utilization >= 1.01:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization*100:.2f}% >= 101%)")
        return None, None, None, None, fraxlend_rate
    if reserve_size == 0:
        logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
        return None, None, None, None, fraxlend_rate

    seconds_per_year = 365.24 * 24 * 3600
    max_investment = 200000
    step = 5000
    investments = range(3000, int(max_investment) + 1, step)

    max_profit = 0
    optimal_investment = 0
    optimal_lend_apr = 0
    optimal_utilization = 0
    valid_investments = 0

    if rate_type == "Variable V2":
        old_full_utilization_interest = v2_model.calculate_old_full_utilization_interest(lend_apr, utilization)
        for investment in investments:
            new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
            logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
            if new_utilization < 0.76:
                logger.debug(f"Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
                continue

            valid_investments += 1
            new_rate_per_sec, _ = v2_model.get_new_rate(delta_time, new_utilization, old_full_utilization_interest)
            new_lend_apr = new_rate_per_sec * seconds_per_year * new_utilization * 100
            # Добавляем Fraxlend ставку к новому Lend APR для указанной пары
            if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
                new_lend_apr += fraxlend_rate
            daily_profit = (investment * new_lend_apr / 100) / 365.24

            if daily_profit > max_profit:
                max_profit = daily_profit
                optimal_investment = investment
                optimal_lend_apr = new_lend_apr
                optimal_utilization = new_utilization

    elif rate_type == "Variable V1":
        for investment in investments:
            new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
            logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
            valid_investments += 1
            new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
            # Добавляем Fraxlend ставку к новому Lend APR для указанной пары
            if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
                new_lend_apr += fraxlend_rate
            daily_profit = (investment * new_lend_apr / 100) / 365.24

            if daily_profit > max_profit:
                max_profit = daily_profit
                optimal_investment = investment
                optimal_lend_apr = new_lend_apr
                optimal_utilization = new_utilization

    else:
        logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
        return None, None, None, None, fraxlend_rate

    if max_profit == 0:
        logger.info(f"Не найдено допустимых вложений для пары, valid_investments={valid_investments}")
        return None, None, None, None, fraxlend_rate

    return optimal_investment, max_profit, optimal_lend_apr, optimal_utilization, fraxlend_rate

# def calculate_optimal_investment(data, v1_model, v2_model, driver, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
#         utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         pair_address = data.get("Link", "").split("/")[-1]
#
#         # Для пары 0xDbe88DBAc39263c47629ebbA02b3eF4cf0752A72 добавляем ставку Fraxlend V1 FRAX/FXS
#         if pair_address == "0xDbe88DBAc39263c47629ebbA02b3eF4cf0752A72":
#             fraxlend_rate = fetch_fraxlend_v1_frax_fxs_rate(driver)
#             lend_apr += fraxlend_rate
#             logger.info(f"Добавлена ставка Fraxlend V1 FRAX/FXS ({fraxlend_rate}%) к Lend APR для пары {pair_address}. Новый Lend APR: {lend_apr}%")
#
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         send_to_telegram(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         return None, None, None, None
#
#     # Фильтрация: Lend APR > MIN_LEND_APR_THRESHOLD, Utilization Rate < 101%, Reserve Size != 0
#     if lend_apr <= MIN_LEND_APR_THRESHOLD:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_LEND_APR_THRESHOLD}%)")
#         return None, None, None, None
#     if utilization >= 1.01:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization*100}% >= 101%)")
#         return None, None, None, None
#     if reserve_size == 0:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
#         return None, None, None, None
#
#     seconds_per_year = 365.24 * 24 * 3600
#     max_investment = 200000
#     step = 5000
#     investments = range(3000, int(max_investment) + 1, step)
#
#     max_profit = 0
#     optimal_investment = 0
#     optimal_lend_apr = 0
#     optimal_utilization = 0
#     valid_investments = 0
#
#     if rate_type == "Variable V2":
#         old_full_utilization_interest = v2_model.calculate_old_full_utilization_interest(lend_apr, utilization)
#         for investment in investments:
#             new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#             logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
#             if new_utilization < 0.76:
#                 logger.debug(f"Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
#                 continue
#
#             valid_investments += 1
#             new_rate_per_sec, _ = v2_model.get_new_rate(delta_time, new_utilization, old_full_utilization_interest)
#             new_lend_apr = new_rate_per_sec * seconds_per_year * new_utilization * 100
#             # Добавляем Fraxlend ставку к новому Lend APR для указанной пары
#             if pair_address == "0xDbe88DBAc39263c47629ebbA02b3eF4cf0752A72":
#                 new_lend_apr += fraxlend_rate
#             daily_profit = (investment * new_lend_apr / 100) / 365.24
#
#             if daily_profit > max_profit:
#                 max_profit = daily_profit
#                 optimal_investment = investment
#                 optimal_lend_apr = new_lend_apr
#                 optimal_utilization = new_utilization
#
#     elif rate_type == "Variable V1":
#         for investment in investments:
#             new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#             logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
#             valid_investments += 1
#             new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
#             # Добавляем Fraxlend ставку к новому Lend APR для указанной пары
#             if pair_address == "0xDbe88DBAc39263c47629ebbA02b3eF4cf0752A72":
#                 new_lend_apr += fraxlend_rate
#             daily_profit = (investment * new_lend_apr / 100) / 365.24
#
#             if daily_profit > max_profit:
#                 max_profit = daily_profit
#                 optimal_investment = investment
#                 optimal_lend_apr = new_lend_apr
#                 optimal_utilization = new_utilization
#
#     else:
#         logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return None, None, None, None
#
#     if max_profit == 0:
#         logger.info(f"Не найдено допустимых вложений для пары, valid_investments={valid_investments}")
#         return None, None, None, None
#
#     return optimal_investment, max_profit, optimal_lend_apr, optimal_utilization

# def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
#         utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         send_to_telegram(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         return None, None, None, None
#
#     # Фильтрация: Lend APR > 20%, Utilization Rate < 101%, Reserve Size != 0
#     if lend_apr <= 20:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= 20%)")
#         return None, None, None, None
#     if utilization >= 1.01:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization*100}% >= 101%)")
#         return None, None, None, None
#     if reserve_size == 0:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
#         return None, None, None, None
#
#     seconds_per_year = 365.24 * 24 * 3600
#     max_investment = 200000
#     step = 5000
#     investments = range(3000, int(max_investment) + 1, step)
#
#     max_profit = 0
#     optimal_investment = 0
#     optimal_lend_apr = 0
#     optimal_utilization = 0
#     valid_investments = 0
#
#     if rate_type == "Variable V2":
#         old_full_utilization_interest = v2_model.calculate_old_full_utilization_interest(lend_apr, utilization)
#         for investment in investments:
#             new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#             logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
#             if new_utilization < 0.76:
#                 logger.debug(f"Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
#                 continue
#
#             valid_investments += 1
#             new_rate_per_sec, _ = v2_model.get_new_rate(delta_time, new_utilization, old_full_utilization_interest)
#             new_lend_apr = new_rate_per_sec * seconds_per_year * new_utilization * 100
#             daily_profit = (investment * new_lend_apr / 100) / 365.24
#
#             if daily_profit > max_profit:
#                 max_profit = daily_profit
#                 optimal_investment = investment
#                 optimal_lend_apr = new_lend_apr
#                 optimal_utilization = new_utilization
#
#     elif rate_type == "Variable V1":
#         for investment in investments:
#             new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#             logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
#             valid_investments += 1
#             new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
#             daily_profit = (investment * new_lend_apr / 100) / 365.24
#
#             if daily_profit > max_profit:
#                 max_profit = daily_profit
#                 optimal_investment = investment
#                 optimal_lend_apr = new_lend_apr
#                 optimal_utilization = new_utilization
#
#     else:
#         logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return None, None, None, None
#
#     if max_profit == 0:
#         logger.info(f"Не найдено допустимых вложений для пары, valid_investments={valid_investments}")
#         return None, None, None, None
#
#     return optimal_investment, max_profit, optimal_lend_apr, optimal_utilization

def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": message
        }
        try:
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
            else:
                logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
        except Exception as e:
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


def process_pairs():
    logger.info(f"Запуск функции process_pairs, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    send_to_telegram("Тест: Сервер запущен, начинаем парсинг")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1280,720')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-images')
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
            logger.info(f"Начало парсинга пар, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
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
            logger.info(f"Полученные пары: {pair_links}")
            if driver:
                try:
                    driver.quit()
                except NewConnectionError:
                    logger.info("Игнорируется NewConnectionError при закрытии WebDriver для списка пар")
                kill_chromedriver()
                logger.info(f"WebDriver закрыт после получения списка пар, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            logger.info(f"Найдено пар: {len(pair_links)}")
            if not pair_links:
                logger.warning("Список пар пуст. Пропускаем итерацию.")
                send_to_telegram("Предупреждение: Список пар пуст. Проверьте сайт или селектор.")
                time.sleep(CHECK_INTERVAL)
                continue

            skipped_pairs = []

            for url in tqdm(pair_links, desc="Обработка пар"):
                if time.time() - iteration_start_time > MAX_ITERATION_TIME:
                    logger.warning(f"Превышено максимальное время итерации ({MAX_ITERATION_TIME} секунд). Пропускаем оставшиеся пары.")
                    send_to_telegram(f"Превышено время итерации ({MAX_ITERATION_TIME} секунд). Пропущено {len(pair_links) - pair_links.index(url)} пар.")
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
                        logger.error(f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver для {url}: {type(e).__name__}: {e}")
                        if attempt == 2:
                            logger.error(f"Не удалось инициализировать WebDriver для {url} после 3 попыток.")
                            send_to_telegram(f"Ошибка: Не удалось инициализировать WebDriver для {url}.")
                            break
                        time.sleep(2)

                if driver:
                    data = fetch_metrics(driver, url)
                    logger.info(f"Полученные данные для {url}: {data}")
                    if all(data.get(label, "N/A") == "N/A" for label in ["Available Liquidity", "Utilization Rate", "Lend APR", "Reserve Size"]):
                        logger.info(f"Пропущена пара из-за некорректных данных: {url}")
                        processed_urls.add(url)
                        send_to_telegram(f"Пропущена пара {url} (Collateral: {data.get('Collateral', 'N/A')}) из-за некорректных данных: {data}")
                    else:
                        # Передаем driver в calculate_optimal_investment
                        optimal_investment, max_profit, optimal_lend_apr, optimal_utilization, fraxlend_rate = calculate_optimal_investment(
                            data, v1_model, v2_model, driver
                        )

                        rate_type = data.get("Rate Type", "N/A")
                        collateral = data.get("Collateral", "N/A")
                        pair_address = url.split("/")[-1].lower()  # Приводим к нижнему регистру
                        if optimal_investment is not None:
                            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                            message = (
                                f"📄 Пара: {collateral} ({url}, {rate_type})\n"
                                f"Timestamp (UTC): {timestamp} +3 часа\n"
                                f"Старая Lend APR: {data.get('Lend APR')}\n"
                                f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
                            )
                            if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
                                message += f"Fraxlend V1 FRAX/FXS Rate: {fraxlend_rate:.2f}%\n"
                                logger.info(f"Добавлена строка Fraxlend Rate ({fraxlend_rate:.2f}%) в сообщение для пары {pair_address}")
                            message += (
                                f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
                                f"Максимальный доход за 1 день: ${max_profit:,.2f}\n"
                                f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
                                f"Available Liquidity: {data.get('Available Liquidity')}\n"
                                f"Utilization Rate: {data.get('Utilization Rate')}\n"
                                f"Borrow APR: {data.get('Borrow APR')}\n"
                                f"Reserve Size: {data.get('Reserve Size')}\n"
                                f"Rate Type: {rate_type}"
                            )
                            send_to_telegram(message)
                            processed_urls.add(url)
                        else:
                            logger.info(f"Пара {url} не прошла фильтры: {data}")
                            if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
                                logger.info(f"Пара {pair_address} не прошла фильтры, но Fraxlend Rate={fraxlend_rate:.2f}%")

                    peak_memory = max(peak_memory, psutil.Process().memory_info().rss / 1024 / 1024)
                    logger.info(f"Память после обработки {url}: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
                    logger.info(f"Пиковая память в итерации: {peak_memory:.2f} MB")

                    try:
                        driver.quit()
                    except NewConnectionError:
                        logger.info(f"Игнорируется NewConnectionError при закрытии WebDriver для {url}")
                    kill_chromedriver()
                    logger.info(f"WebDriver закрыт после обработки {url}, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            if skipped_pairs:
                send_to_telegram(f"Пропущенные пары в цикле #{iteration_count}: {', '.join(skipped_pairs)}")

        except Exception as e:
            logger.error(f"Ошибка обработки пар: {type(e).__name__}: {e}")
            send_to_telegram(f"Ошибка при обработке пар: {type(e).__name__}: {e}")

        finally:
            if driver:
                try:
                    driver.quit()
                except NewConnectionError:
                    logger.info("Игнорируется NewConnectionError при закрытии WebDriver в finally")
                kill_chromedriver()
                logger.info(f"WebDriver закрыт в finally, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

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
                logger.info(
                    f"Ожидание, осталось {remaining_time} секунд, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB, пиковая память: {peak_memory:.2f} MB")
def main():
    logger.info("Запуск Background Worker")
    send_to_telegram("Тест: Background Worker запущен")
    process_pairs()

if __name__ == "__main__":
    main()