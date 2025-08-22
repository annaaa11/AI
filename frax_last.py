# import os
# import time
# import requests
# from tqdm import tqdm
# from datetime import datetime
# from dataclasses import dataclass
# from typing import Tuple, List, Dict
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, WebDriverException
# import logging
# import psutil
# import subprocess
# import re
# from bs4 import BeautifulSoup
#
# # Константы
# MIN_APR = 5  # Порог для ставки Lend APR
# TOTAL_INVESTMENT = 200_000  # Общая сумма для инвестиций
# INVESTMENT_STEP = 5_000  # Шаг инвестиций для оптимизации
# MIN_INVESTMENT = 3_000  # Минимальная сумма инвестиций
# SECONDS_PER_YEAR = 365.24 * 24 * 3600  # Глобальная константа
# CHECK_INTERVAL = 60 * 60  # Интервал проверки в секундах (1 час)
# MAX_PAIRS = 100  # Максимальное количество пар
# MAX_ITERATION_TIME = 1200  # Максимальное время на итерацию (10 минут)
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)
#
# # Настройки Telegram
# BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"
# CHAT_IDS = [6192278046, 306507209]
#
# # Blacklist for problematic pairs
# BLACKLISTED_PAIRS = set()
#
#
# @dataclass
# class InterestRateParams:
#     max_full_util_rate: float = 3.164468e-8
#     min_full_util_rate: float = 1.580586e-9
#     zero_util_rate: float = 1.584231e-10
#     rate_half_life: float = 172800.0
#     max_target_util: float = 0.85
#     min_target_util: float = 0.75
#     vertex_util: float = 0.875
#     vertex_rate_percent: float = 0.36
#     util_precision: float = 1e5
#     rate_precision: float = 1e18
#
#
# @dataclass
# class TimeWeightedInterestRateParams:
#     protocol_fee: float = 0.1
#     min_apr: float = 0.005
#     max_apr: float = 100.0131
#     min_utilization: float = 0.75
#     max_utilization: float = 0.85
#     rate_half_life_days: float = 0.5
#     util_precision: float = 1e5
#     rate_precision: float = 1e18
#
#
# class VariableInterestRate:
#     def __init__(self, params: InterestRateParams, suffix: str = "[0.5 0.2@.875 5-10k] 2 days (.75-.85)"):
#         self.params = params
#         self.suffix = suffix
#
#     def calculate_old_full_utilization_interest(self, old_lend_apr: float, old_utilization: float) -> float:
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (SECONDS_PER_YEAR * 100)
#         term = (
#                        old_borrow_rate_per_sec - self.params.zero_util_rate) * self.params.vertex_util / old_utilization if old_utilization != 0 else 0
#         vertex_interest = term + self.params.zero_util_rate
#         full_utilization_interest = ((
#                                              vertex_interest - self.params.zero_util_rate) / self.params.vertex_rate_percent) + self.params.zero_util_rate
#         return full_utilization_interest
#
#     def get_full_utilization_interest(self, delta_time: float, utilization: float,
#                                       full_utilization_interest: float) -> float:
#         if utilization < self.params.min_target_util:
#             delta_utilization = ((
#                                          self.params.min_target_util - utilization) * self.params.rate_precision) / self.params.min_target_util
#             decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * (
#                     self.params.rate_half_life * 1e36)) / decay_growth
#         elif utilization > self.params.max_target_util:
#             delta_utilization = ((utilization - self.params.max_target_util) * self.params.rate_precision) / (
#                     self.params.util_precision - self.params.max_target_util)
#             decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * decay_growth) / (
#                     self.params.rate_half_life * 1e36)
#         else:
#             new_full_utilization_interest = full_utilization_interest
#         new_full_utilization_interest = min(new_full_utilization_interest, self.params.max_full_util_rate)
#         new_full_utilization_interest = max(new_full_utilization_interest, self.params.min_full_util_rate)
#         return new_full_utilization_interest
#
#     def get_new_rate(self, delta_time: float, utilization: float, old_full_utilization_interest: float) -> Tuple[
#         float, float]:
#         new_full_utilization_interest = self.get_full_utilization_interest(delta_time, utilization,
#                                                                            old_full_utilization_interest)
#         vertex_interest = (((
#                                     new_full_utilization_interest - self.params.zero_util_rate) * self.params.vertex_rate_percent) + self.params.zero_util_rate)
#         if utilization < self.params.vertex_util:
#             new_rate_per_sec = (self.params.zero_util_rate + (
#                     utilization * (vertex_interest - self.params.zero_util_rate)) / self.params.vertex_util)
#         else:
#             new_rate_per_sec = (vertex_interest + (
#                     (utilization - self.params.vertex_util) * (new_full_utilization_interest - vertex_interest)) / (
#                                         1.0 - self.params.vertex_util))
#         return new_rate_per_sec, new_full_utilization_interest
#
#
# class TimeWeightedVariableInterestRate:
#     def __init__(self, params: TimeWeightedInterestRateParams):
#         self.params = params
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100 / SECONDS_PER_YEAR
#
#     def calculate_new_full_utilization_rate(self, old_utilization: float, old_lend_apr: float,
#                                             delta_time: float) -> float:
#         old_rate_per_sec = self.calculate_rate_per_sec(old_lend_apr)
#         old_full_util_rate = old_rate_per_sec / (
#                 old_utilization * (1 - self.params.protocol_fee)) if old_utilization != 0 else 0
#
#         if old_utilization < self.params.min_utilization:
#             delta_utilization = ((self.params.min_utilization - old_utilization) *
#                                  self.params.rate_precision) / self.params.min_utilization
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * (self.rate_half_life_secs * 1e36)) / decay_growth
#         elif old_utilization > self.params.max_utilization:
#             delta_utilization = ((old_utilization - self.params.max_utilization) *
#                                  self.params.rate_precision) / (
#                                         self.params.util_precision - self.params.max_utilization)
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * decay_growth) / (self.rate_half_life_secs * 1e36)
#         else:
#             new_full_util_rate = old_full_util_rate
#
#         min_rate_per_sec = self.calculate_rate_per_sec(self.params.min_apr)
#         max_rate_per_sec = self.calculate_rate_per_sec(self.params.max_apr)
#         new_full_util_rate = min(max(new_full_util_rate, min_rate_per_sec), max_rate_per_sec)
#         return new_full_util_rate
#
#     def get_new_lend_apr(self, delta_time: float, current_utilization: float, old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(
#             old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (
#                 1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * SECONDS_PER_YEAR * 100
#         return lend_apr
#
#
# def get_pair_links(driver, max_retries=3) -> List[str]:
#     """Получает список ссылок на пары с сайта Fraxlend."""
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
#
#
# def fetch_metrics(driver, url: str, timeout=15, max_retries=2) -> Dict:
#     """Извлекает метрики для указанной пары."""
#     logger.info(f"Загрузка страницы: {url}")
#     if url in BLACKLISTED_PAIRS:
#         logger.info(f"Пара {url} в черном списке, пропускаем")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
#     for attempt in range(max_retries):
#         try:
#             start_time = time.time()
#             process = psutil.Process()
#             peak_memory = process.memory_info().rss / 1024 / 1024
#             driver.get(url)
#             WebDriverWait(driver, timeout).until(
#                 EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#             if "Frax Facts" not in driver.page_source:
#                 raise Exception("Страница не содержит ожидаемого содержимого 'Frax Facts'")
#             with open(f"page_{url.split('/')[-1]}.html", "w") as f:
#                 f.write(driver.page_source)
#             logger.info(f"Сохранён HTML страницы {url} для отладки")
#             labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type",
#                       "Collateral"]
#             data = {"Link": url}
#             for label in labels:
#                 try:
#                     if label == "Collateral":
#                         possible_labels = ["Collateral", "Collateral Token", "Collateral Asset"]
#                         for possible_label in possible_labels:
#                             try:
#                                 el = driver.find_element(By.XPATH,
#                                                          f"//div[contains(text(), '{possible_label}')]/following-sibling::div")
#                                 data[label] = el.text.strip()
#                                 logger.info(f"Найдено значение для {possible_label}: {data[label]}")
#                                 break
#                             except:
#                                 continue
#                         else:
#                             data[label] = "N/A"
#                             logger.warning(f"Не удалось найти значение для Collateral на {url}")
#                     else:
#                         el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                         data[label] = el.text.strip()
#                     logger.info(f"Извлечено значение для {label}: {data[label]}")
#                 except Exception as e:
#                     data[label] = "N/A"
#                     logger.warning(f"Не удалось извлечь {label} для {url}: {type(e).__name__}: {e}")
#             logger.info(f"Данные для {url}: {data}")
#             logger.info(f"Время обработки {url}: {time.time() - start_time:.2f} секунд")
#             peak_memory = max(peak_memory, process.memory_info().rss / 1024 / 1024)
#             logger.info(f"Пиковая память во время обработки {url}: {peak_memory:.2f} MB")
#             return data
#         except TimeoutException as e:
#             logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при загрузке данных для {url}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error(f"Пара {url} не загрузилась после {max_retries} попыток, добавляем в черный список")
#                 BLACKLISTED_PAIRS.add(url)
#                 send_to_telegram(f"Пара {url} добавлена в черный список после {max_retries} неудачных попыток загрузки")
#                 return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                         "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
#             time.sleep(2)
#         except Exception as e:
#             logger.error(
#                 f"Попытка {attempt + 1}/{max_retries}: Ошибка при загрузке данных для {url}: {type(e).__name__}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error(f"Пара {url} не загрузилась после {max_retries} попыток, добавляем в черный список")
#                 BLACKLISTED_PAIRS.add(url)
#                 send_to_telegram(f"Пара {url} добавлена в черный список после ошибки: {type(e).__name__}: {e}")
#                 return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                         "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
#             time.sleep(2)
#
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     """Парсит строковое представление суммы в долларах."""
#     if amount_str == "N/A" or not amount_str:
#         logger.warning(f"Невозможно распарсить сумму: {amount_str}, возвращается 0.0")
#         return 0.0
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if 'm' in amount_str.lower():
#             return value * 1_000_000  # Предполагается, что 'm' всегда означает миллионы
#         elif 'k' in amount_str.lower():
#             return value * 1_000
#         return value
#     except ValueError:
#         logger.error(f"Ошибка парсинга суммы: {amount_str}, возвращается 0.0")
#         return 0.0
#
#
# def calculate_pair_profit(data: Dict, v1_model, v2_model, investment: float, bonus: float = 0.0,
#                           delta_time: float = 86400.0) -> Tuple[float, float, float, Dict]:
#     """Рассчитывает прибыль, новую Lend APR и утилизацию для пары при заданной инвестиции."""
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
#         utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         pair_address = data.get("Link", "").split("/")[-1].lower()
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         send_to_telegram(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         return 0.0, 0.0, 0.0, data
#
#     if lend_apr <= MIN_APR:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_APR}%)")
#         return 0.0, 0.0, 0.0, data
#     if utilization >= 1.01:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization * 100}% >= 101%)")
#         return 0.0, 0.0, 0.0, data
#     if reserve_size == 0:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
#         return 0.0, 0.0, 0.0, data
#     if investment > available_liquidity:
#         logger.info(
#             f"Пара отфильтрована: {data.get('Link')} (Investment={investment} > Available Liquidity={available_liquidity})")
#         return 0.0, 0.0, 0.0, data
#
#     new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#     logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
#
#     if rate_type == "Variable V2":
#         if new_utilization < 0.76:
#             logger.debug(f"Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
#             return 0.0, 0.0, 0.0, data
#         old_full_utilization_interest = v2_model.calculate_old_full_utilization_interest(lend_apr, utilization)
#         new_rate_per_sec, _ = v2_model.get_new_rate(delta_time, new_utilization, old_full_utilization_interest)
#         new_lend_apr = new_rate_per_sec * SECONDS_PER_YEAR * new_utilization * 100
#         if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
#             new_lend_apr += bonus
#         daily_profit = (investment * new_lend_apr / 100) / 365.24
#     elif rate_type == "Variable V1":
#         new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
#         if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
#             new_lend_apr += bonus
#         daily_profit = (investment * new_lend_apr / 100) / 365.24
#     else:
#         logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return 0.0, 0.0, 0.0, data
#
#     return daily_profit, new_lend_apr, new_utilization, data
#
#
# def optimize_investment_distribution(pairs: List[Dict], v1_model, v2_model, total_investment: float, bonus: float) -> \
# List[Dict]:
#     """Оптимизирует распределение инвестиций между парами для максимизации дневной прибыли."""
#     pair_profits = []
#     for data in pairs:
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         max_investment = min(total_investment, available_liquidity)
#         investments = range(MIN_INVESTMENT, int(max_investment) + 1, INVESTMENT_STEP)
#
#         for investment in investments:
#             daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(data, v1_model, v2_model,
#                                                                                       investment, bonus)
#             if daily_profit > 0:
#                 pair_profits.append({
#                     "data": data,
#                     "daily_profit": daily_profit,
#                     "new_lend_apr": new_lend_apr,
#                     "new_utilization": new_utilization,
#                     "investment": investment,
#                     "profit_per_dollar": daily_profit / investment if investment > 0 else 0
#                 })
#
#     if not pair_profits:
#         logger.info("Нет подходящих пар для инвестиций")
#         return []
#
#     # Инициализируем лучшее распределение
#     best_allocation = []
#     max_total_profit = 0.0
#
#     # Проверяем вложение всей суммы в одну пару
#     for data in pairs:
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         if available_liquidity >= total_investment:
#             daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(data, v1_model, v2_model,
#                                                                                       total_investment, bonus)
#             if daily_profit > max_total_profit:
#                 max_total_profit = daily_profit
#                 best_allocation = [{
#                     "data": data,
#                     "investment": total_investment,
#                     "daily_profit": daily_profit,
#                     "new_lend_apr": new_lend_apr,
#                     "new_utilization": new_utilization
#                 }]
#                 logger.info(
#                     f"Обновлено лучшее вложение: ${total_investment:,.2f} в {data['Link']}, прибыль: ${daily_profit:,.2f}")
#
#     # Итеративное распределение по парам
#     remaining_investment = total_investment
#     allocated_pairs = []
#     used_urls = set()
#
#     while remaining_investment >= MIN_INVESTMENT and pair_profits:
#         # Сортируем по прибыли на доллар, исключая уже использованные пары
#         valid_profits = [p for p in pair_profits if p["data"]["Link"] not in used_urls]
#         if not valid_profits:
#             break
#         valid_profits.sort(key=lambda x: x["profit_per_dollar"], reverse=True)
#         best_option = valid_profits[0]
#         pair_url = best_option["data"]["Link"]
#         available_liquidity = parse_dollar_amount(best_option["data"].get("Available Liquidity", "0"))
#
#         # Проверяем разные суммы инвестиций для этой пары
#         max_investment = min(remaining_investment, available_liquidity)
#         best_investment = 0
#         best_profit = 0
#         best_lend_apr = 0
#         best_utilization = 0
#         for investment in range(MIN_INVESTMENT, int(max_investment) + 1, INVESTMENT_STEP):
#             daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(
#                 best_option["data"], v1_model, v2_model, investment, bonus
#             )
#             if daily_profit > best_profit:
#                 best_profit = daily_profit
#                 best_investment = investment
#                 best_lend_apr = new_lend_apr
#                 best_utilization = new_utilization
#
#         if best_profit > 0:
#             allocated_pairs.append({
#                 "data": best_option["data"],
#                 "investment": best_investment,
#                 "daily_profit": best_profit,
#                 "new_lend_apr": best_lend_apr,
#                 "new_utilization": best_utilization
#             })
#             remaining_investment -= best_investment
#             used_urls.add(pair_url)
#             logger.info(f"Инвестировано ${best_investment:,.2f} в {pair_url}, прибыль: ${best_profit:,.2f}")
#         else:
#             break
#
#     # Сравниваем с лучшим вложением в одну пару
#     total_profit = sum(p["daily_profit"] for p in allocated_pairs)
#     if total_profit > max_total_profit:
#         best_allocation = allocated_pairs
#         max_total_profit = total_profit
#         logger.info(f"Обновлено лучшее распределение: {len(best_allocation)} пар, прибыль: ${total_profit:,.2f}")
#
#     if not best_allocation:
#         logger.info("Не найдено подходящих комбинаций для инвестиций")
#         return []
#
#     remaining_investment = total_investment - sum(p["investment"] for p in best_allocation)
#     if remaining_investment > 0:
#         logger.info(f"Остаток нераспределенных средств: ${remaining_investment:,.2f}")
#         send_to_telegram(f"Остаток нераспределенных средств: ${remaining_investment:,.2f}")
#
#     return best_allocation
#
#
# def send_to_telegram(message: str):
#     """Отправляет сообщение в Telegram."""
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         try:
#             payload = {
#                 "chat_id": chat_id,
#                 "text": message[:4096]
#             }
#             response = requests.post(url, data=payload, timeout=10)
#             if response.status_code != 200:
#                 logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#             else:
#                 logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
#         except requests.RequestException as e:
#             logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {type(e).__name__}: {e}")
#
#
# def kill_chromedriver():
#     """Принудительно завершает все процессы chromedriver."""
#     try:
#         subprocess.run(['pkill', '-f', 'chromedriver'], check=True)
#         logger.info("Все процессы chromedriver завершены")
#     except subprocess.CalledProcessError:
#         logger.info("Процессы chromedriver не найдены")
#     except Exception as e:
#         logger.error(f"Ошибка при завершении chromedriver: {type(e).__name__}: {e}")
#
#
# def get_driver(chromedriver_path: str):
#     """Инициализирует WebDriver с заданными настройками."""
#     options = Options()
#     options.add_argument("--headless=new")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--window-size=1920,1080")
#     options.add_argument(
#         "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) "
#         "Chrome/115.0 Safari/537.36"
#     )
#     driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
#     return driver
#
#
# def get_fraxlend_fxs_lower_bound(driver) -> float:
#     """Получает нижнюю границу APR для Fraxlend V1 FRAX/FXS."""
#     url = "https://app.frax.finance/staking/overview"
#     logger.info(f"Загрузка страницы для получения бонуса: {url}")
#     try:
#         driver.get(url)
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Fraxlend V1 FRAX/FXS')]"))
#         )
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         target_text = re.compile(r"Fraxlend\s*V1\s*FRAX/FXS", re.IGNORECASE)
#         fees_range = None
#
#         for element in soup.find_all(string=target_text):
#             parent = element.find_parent()
#             if parent:
#                 for sibling in parent.find_all_next(string=True, limit=10):
#                     if re.search(r'\d+\.\d+%\s*-\s*\d+\.\d+%', sibling):
#                         fees_range = sibling.strip()
#                         break
#                 if fees_range:
#                     break
#
#         if fees_range:
#             match = re.search(r'(\d+\.\d+)%\s*-\s*\d+\.\d+%', fees_range)
#             if match:
#                 logger.info(f"Извлечена нижняя граница APR: {match.group(1)}%")
#                 return float(match.group(1))
#             else:
#                 logger.warning(f"Не удалось извлечь нижнюю границу из диапазона: {fees_range}")
#                 return 0.0
#         else:
#             logger.warning(f"Не найдено 'Fraxlend V1 FRAX/FXS' или связанный диапазон APR на {url}")
#             return 0.0
#
#     except Exception as e:
#         logger.error(f"Ошибка получения бонуса для Fraxlend V1 FRAX/FXS: {type(e).__name__}: {e}")
#         return 0.0
#
#
# def process_pairs():
#     """Обрабатывает пары Fraxlend и распределяет инвестиции для максимизации прибыли."""
#     logger.info(
#         f"Запуск функции process_pairs, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#     send_to_telegram("Сервер запущен, начинаем парсинг")
#     chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
#     logger.info(f"Используемый путь к chromedriver: {chromedriver_path}")
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     iteration_count = 0
#
#     while True:
#         iteration_start_time = time.time()
#         iteration_count += 1
#         peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
#
#         processed_urls = set()
#         logger.info(f"Сброс processed_urls для цикла #{iteration_count}")
#
#         logger.info(f"Текущий черный список: {BLACKLISTED_PAIRS}")
#         send_to_telegram(f"Начало цикла #{iteration_count}. Черный список: {BLACKLISTED_PAIRS}")
#
#         if iteration_count % 24 == 0:
#             logger.info("Сброс черного списка")
#             BLACKLISTED_PAIRS.clear()
#             send_to_telegram("Черный список сброшен")
#
#         driver = None
#         try:
#             logger.info(f"Инициализация WebDriver для цикла #{iteration_count}")
#             kill_chromedriver()
#             driver = get_driver(chromedriver_path)
#             logger.info("WebDriver успешно инициализирован")
#
#             pair_links = get_pair_links(driver)
#             logger.info(f"Полученные пары: {pair_links}")
#
#             if not pair_links:
#                 logger.warning("Список пар пуст. Пропускаем итерацию.")
#                 send_to_telegram("Предупреждение: Список пар пуст. Проверьте сайт или селектор.")
#                 time.sleep(CHECK_INTERVAL)
#                 continue
#
#             frax_fxs_pair = "https://facts.frax.finance/fraxlend/pairs/0xdbe88dbac39263c47629ebba02b3ef4cf0752a72"
#             bonus = get_fraxlend_fxs_lower_bound(driver) if frax_fxs_pair in pair_links else 0.0
#             logger.info(f"Bonus for FRAX/FXS: {bonus}%")
#
#             pairs_data = []
#             skipped_pairs = []
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if time.time() - iteration_start_time > MAX_ITERATION_TIME:
#                     logger.warning(
#                         f"Превышено максимальное время итерации ({MAX_ITERATION_TIME} секунд). Пропускаем оставшиеся пары.")
#                     send_to_telegram(
#                         f"Превышено время итерации ({MAX_ITERATION_TIME} секунд). Пропущено {len(pair_links) - pair_links.index(url)} пар.")
#                     break
#
#                 if url in processed_urls or url in BLACKLISTED_PAIRS:
#                     logger.info(f"Пропущена пара (уже обработана или в черном списке): {url}")
#                     skipped_pairs.append(url)
#                     continue
#
#                 logger.info(f"Обработка пары: {url}")
#                 data = fetch_metrics(driver, url)
#                 logger.info(f"Полученные данные для {url}: {data}")
#
#                 if all(data.get(label, "N/A") == "N/A" for label in
#                        ["Available Liquidity", "Utilization Rate", "Lend APR", "Reserve Size"]):
#                     logger.info(f"Пропущена пара из-за некорректных данных: {url}")
#                     processed_urls.add(url)
#                     send_to_telegram(
#                         f"Пропущена пара {url} (Collateral: {data.get('Collateral', 'N/A')}) из-за некорректных данных: {data}")
#                     continue
#
#                 pairs_data.append(data)
#                 processed_urls.add(url)
#
#             if pairs_data:
#                 allocated_pairs = optimize_investment_distribution(pairs_data, v1_model, v2_model, TOTAL_INVESTMENT,
#                                                                    bonus)
#                 if allocated_pairs:
#                     total_daily_profit = sum(p["daily_profit"] for p in allocated_pairs)
#                     message = f"Результаты распределения инвестиций (${TOTAL_INVESTMENT:,.2f}):\n"
#                     for pair in allocated_pairs:
#                         data = pair["data"]
#                         message += (
#                             f"\nПара: {data.get('Collateral', 'N/A')} ({data.get('Rate Type', 'N/A')})\n"
#                             f"Ссылка: {data.get('Link')}\n"
#                             f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
#                             f"Текущая Lend APR: {data.get('Lend APR')} {'+' + str(bonus) if data.get('Link').split('/')[-1].lower() == '0xdbe88dbac39263c47629ebba02b3ef4cf0752a72' else ''}\n"
#                             f"Новая Lend APR: {pair['new_lend_apr']:,.2f}%\n"
#                             f"Инвестировано: ${pair['investment']:,.2f}\n"
#                             f"Доход за день: ${pair['daily_profit']:,.2f}\n"
#                             f"Новая утилизация: {pair['new_utilization'] * 100:.2f}%\n"
#                             f"Ликвидность: {data.get('Available Liquidity')}\n"
#                             f"Текущая утилизация: {data.get('Utilization Rate')}\n"
#                             f"Borrow APR: {data.get('Borrow APR')}\n"
#                             f"Резерв: {data.get('Reserve Size')}\n"
#                         )
#                     message += f"\nОбщий доход за день: ${total_daily_profit:,.2f}"
#                     send_to_telegram(message)
#                 else:
#                     send_to_telegram("Не найдено подходящих пар для инвестиций.")
#             else:
#                 send_to_telegram("Нет данных о пар для обработки.")
#
#             if skipped_pairs:
#                 send_to_telegram(f"Пропущенные пары в цикле #{iteration_count}: {', '.join(skipped_pairs)}")
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки пар: {type(e).__name__}: {e}")
#             send_to_telegram(f"Ошибка при обработке пар: {type(e).__name__}: {e}")
#
#         finally:
#             if driver:
#                 try:
#                     driver.quit()
#                 except WebDriverException:
#                     logger.info("Игнорируется WebDriverException при закрытии WebDriver")
#                 kill_chromedriver()
#                 logger.info(f"WebDriver закрыт, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#
#             elapsed_time = time.time() - iteration_start_time
#             logger.info(f"Итерация завершена за {elapsed_time:.2f} секунд")
#             logger.info(f"Пиковая память в итерации: {peak_memory:.2f} MB")
#             logger.info(f"Ожидание {CHECK_INTERVAL} секунд перед следующей итерацией")
#             remaining_time = CHECK_INTERVAL
#             while remaining_time > 0:
#                 sleep_time = min(30, remaining_time)
#                 time.sleep(sleep_time)
#                 remaining_time -= sleep_time
#                 peak_memory = max(peak_memory, psutil.Process().memory_info().rss / 1024 / 1024)
#                 logger.info(
#                     f"Ожидание, осталось {remaining_time} секунд, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB, пиковая память: {peak_memory:.2f} MB")
#
#
# def main():
#     """Основная функция для запуска обработки пар."""
#     logger.info("Запуск Background Worker")
#     send_to_telegram("Background Worker запущен")
#     process_pairs()
#
#
# if __name__ == "__main__":
#     main()
#
# import math
# from dataclasses import dataclass
# from typing import Dict, Tuple
# import logging
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Константы
# SECONDS_PER_YEAR = 365.24 * 24 * 3600
# MIN_APR = 0.01  # Минимальная APR для фильтрации
#
# @dataclass
# class VariableRateV2Params:
#     MIN_TARGET_UTIL: float
#     MAX_TARGET_UTIL: float
#     VERTEX_UTILIZATION: float
#     ZERO_UTIL_RATE: float
#     MIN_FULL_UTIL_RATE: float
#     MAX_FULL_UTIL_RATE: float
#     RATE_HALF_LIFE: float
#     VERTEX_RATE_PERCENT: float
#
# def apr_to_per_sec(apr: float) -> float:
#     return apr / SECONDS_PER_YEAR
#
# def per_sec_to_apr(rate_per_sec: float) -> float:
#     return rate_per_sec * SECONDS_PER_YEAR
#
# # === 1) Восстановить старый full_util_rate из наблюдаемых (old_util, old_lend_apr) ===
# def infer_old_full_from_observation(params: VariableRateV2Params,
#                                     old_util: float,
#                                     old_lend_apr: float) -> float:
#     """
#     На вход:
#       old_util        — старая утилизация (доля)
#       old_lend_apr    — наблюдаемый Lend APR при old_util (годовой, доля)
#     Возвращает:
#       old_full_rate_per_sec — искомая ставка при 100% util (в сек^-1)
#     """
#     old_borrow_apr = old_lend_apr / max(old_util, 1e-12)
#     r_obs = apr_to_per_sec(old_borrow_apr)
#
#     z = apr_to_per_sec(params.ZERO_UTIL_RATE)
#     p = params.VERTEX_RATE_PERCENT
#     u = old_util
#     u_v = params.VERTEX_UTILIZATION
#
#     if u < u_v:
#         denom = (u / u_v) * p
#         x = z + (r_obs - z) / max(denom, 1e-18)
#     else:
#         a = (u - u_v) / max(1 - u_v, 1e-18)
#         coef_x = (p + a * (1 - p))
#         const = (1 - p) * z * (1 - a)
#         x = (r_obs - const) / max(coef_x, 1e-18)
#
#     x = min(max(x, apr_to_per_sec(params.MIN_FULL_UTIL_RATE)),
#             apr_to_per_sec(params.MAX_FULL_UTIL_RATE))
#     return x
#
# # === 2) Обновление full_util_rate по half-life + расхождению утилы от целевого коридора ===
# def get_full_util_rate(params: VariableRateV2Params,
#                        delta_time: float,
#                        utilization: float,
#                        old_full_rate_per_sec: float) -> float:
#     if utilization < params.MIN_TARGET_UTIL:
#         delta_u = params.MIN_TARGET_UTIL - utilization
#         decay_growth = params.RATE_HALF_LIFE + (delta_u ** 2) * delta_time
#         new_full_rate = old_full_rate_per_sec * (params.RATE_HALF_LIFE / decay_growth)
#     elif utilization > params.MAX_TARGET_UTIL:
#         delta_u = utilization - params.MAX_TARGET_UTIL
#         decay_growth = params.RATE_HALF_LIFE + (delta_u ** 2) * delta_time
#         new_full_rate = old_full_rate_per_sec * (decay_growth / params.RATE_HALF_LIFE)
#     else:
#         new_full_rate = old_full_rate_per_sec
#
#     return min(max(new_full_rate, apr_to_per_sec(params.MIN_FULL_UTIL_RATE)),
#                apr_to_per_sec(params.MAX_FULL_UTIL_RATE))
#
# # === 3) Borrow per second по piecewise-линейке; затем Lend = Borrow * Util ===
# def get_new_rates(params: VariableRateV2Params,
#                   delta_time: float,
#                   utilization: float,
#                   old_full_rate_per_sec: float):
#     new_full = get_full_util_rate(params, delta_time, utilization, old_full_rate_per_sec)
#
#     z = apr_to_per_sec(params.ZERO_UTIL_RATE)
#     p = params.VERTEX_RATE_PERCENT
#     u_v = params.VERTEX_UTILIZATION
#
#     v = ((new_full - z) * p) + z  # vertex rate per sec
#
#     if utilization < u_v:
#         r = z + (utilization / u_v) * (v - z)
#     else:
#         r = v + ((utilization - u_v) / max(1 - u_v, 1e-18)) * (new_full - v)
#
#     borrow_apr = per_sec_to_apr(r)
#     lend_apr = borrow_apr * utilization
#     return lend_apr, borrow_apr, per_sec_to_apr(new_full)
#
# # === 4) Удобная обёртка под ваш сценарий ===
# def compute_new_lend_apr(params: VariableRateV2Params,
#                          old_util: float,
#                          old_lend_apr: float,
#                          new_util: float,
#                          delta_t: float):
#     old_full_sec = infer_old_full_from_observation(params, old_util, old_lend_apr)
#     new_lend_apr, new_borrow_apr, new_full_apr = get_new_rates(params, delta_t, new_util, old_full_sec)
#     return {
#         "new_lend_apr": new_lend_apr,
#         "new_borrow_apr": new_borrow_apr,
#         "new_full_util_apr": new_full_apr
#     }
#
# # === 5) Функция расчёта прибыли для вашей логики ===
# def calculate_pair_profit(data: Dict, v1_model, v2_params, investment: float, bonus: float = 0.0, delta_time: float = 86400.0) -> Tuple[float, float, float, Dict]:
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
#         utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
#         available_liquidity = float(data.get("Available Liquidity", "0").replace("k", "000").replace("$", "").replace(",", ""))
#         reserve_size = float(data.get("Reserve Size", "0").replace("k", "000").replace("$", "").replace(",", ""))
#         rate_type = data.get("Rate Type", "N/A")
#         pair_address = data.get("Link", "").split("/")[-1].lower()
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         return 0.0, 0.0, 0.0, data
#
#     if lend_apr <= MIN_APR or utilization >= 1.01 or reserve_size == 0 or investment > available_liquidity:
#         logger.info(f"Пара отфильтрована: {data.get('Link')}")
#         return 0.0, 0.0, 0.0, data
#
#     new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#
#     if rate_type == "Variable V2":
#         res = compute_new_lend_apr(v2_params, utilization, lend_apr, new_utilization, delta_time)
#         new_lend_apr = res["new_lend_apr"]
#         if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
#             new_lend_apr += bonus
#         daily_profit = (investment * new_lend_apr / 100) / 365.24
#     elif rate_type == "Variable V1":
#         new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
#         if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
#             new_lend_apr += bonus
#         daily_profit = (investment * new_lend_apr / 100) / 365.24
#     else:
#         logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return 0.0, 0.0, 0.0, data
#
#     return daily_profit, new_lend_apr, new_utilization, data
#
# # === 6) Пример использования и расчёт ===
# if __name__ == "__main__":
#     # Инициализация параметров
#     v2_params = VariableRateV2Params(
#         MIN_TARGET_UTIL=0.75,
#         MAX_TARGET_UTIL=0.85,
#         VERTEX_UTILIZATION=0.875,
#         ZERO_UTIL_RATE=0.005,
#         MIN_FULL_UTIL_RATE=0.0499,
#         MAX_FULL_UTIL_RATE=99.8779,
#         RATE_HALF_LIFE=172800,
#         VERTEX_RATE_PERCENT=0.2039
#     )
#
#     # Ваши данные
#     data = {
#         "Lend APR": "13.59",
#         "Utilization Rate": "96.27",
#         "Available Liquidity": "17630",  # 17.63k frxUSD
#         "Reserve Size": "472080",       # 472.08k frxUSD
#         "Rate Type": "Variable V2",
#         "Link": "https://facts.frax.finance/fraxlend/pairs/0x8e5f09de0cd7841239410f929a905e214443d9e0"
#     }
#     investment = 13000.0
#     delta_time = 86400.0  # 1 день
#
#     # Расчёт
#     daily_profit, new_lend_apr, new_utilization, updated_data = calculate_pair_profit(data, None, v2_params, investment, delta_time=delta_time)
#
#     # Вывод результатов
#     print(f"Текущая утилизация: {float(data['Utilization Rate'])}%")
#     print(f"Текущая Lend APR: {float(data['Lend APR'])}%")
#     print(f"Новая утилизация: {new_utilization * 100:.2f}%")
#     print(f"Новая Lend APR: {new_lend_apr * 100:.2f}%")
#     print(f"Инвестировано: ${investment:.2f}")
#     print(f"Доход за день: ${daily_profit:.2f}")
#
#     # Дополнительно для утилизации 100%
#     res_100 = compute_new_lend_apr(v2_params, 0.9627, 0.1359, 1.0, delta_time)
#     print(f"\nПри утилизации 100%:")
#     print(f"Новая Lend APR: {res_100['new_lend_apr'] * 100:.2f}%")
#     print(f"Новая Borrow APR: {res_100['new_borrow_apr'] * 100:.2f}%")
#     print(f"Новая Full Util APR: {res_100['new_full_util_apr'] * 100:.2f}%")

# import os
# import time
# import requests
# from tqdm import tqdm
# from datetime import datetime
# from dataclasses import dataclass
# from typing import Tuple, List, Dict
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, WebDriverException
# import logging
# import psutil
# import subprocess
# import re
# from bs4 import BeautifulSoup
#
# # Константы
# MIN_APR = 5  # Порог для ставки Lend APR
# TOTAL_INVESTMENT = 200_000  # Общая сумма для инвестиций
# INVESTMENT_STEP = 5_000  # Шаг инвестиций для оптимизации
# MIN_INVESTMENT = 3_000  # Минимальная сумма инвестиций
# SECONDS_PER_YEAR = 365.24 * 24 * 3600  # Глобальная константа
# CHECK_INTERVAL = 60 * 60  # Интервал проверки в секундах (1 час)
# MAX_PAIRS = 100  # Максимальное количество пар
# MAX_ITERATION_TIME = 1200  # Максимальное время на итерацию (10 минут)
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)
#
# # Настройки Telegram
# BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"
# CHAT_IDS = [6192278046, 306507209]
#
# # Blacklist for problematic pairs
# BLACKLISTED_PAIRS = set()
#
# # === Обновленные параметры для Variable Rate V2 ===
# @dataclass
# class InterestRateParams:
#     MIN_TARGET_UTIL: float = 0.75
#     MAX_TARGET_UTIL: float = 0.85
#     VERTEX_UTILIZATION: float = 0.875
#     ZERO_UTIL_RATE: float = 0.005  # 0.5%
#     MIN_FULL_UTIL_RATE: float = 0.0499  # 4.99%
#     MAX_FULL_UTIL_RATE: float = 99.8779  # 9,987.79%
#     RATE_HALF_LIFE: float = 172800.0  # 2 дня
#     VERTEX_RATE_PERCENT: float = 0.2039  # 20.39%
#
# @dataclass
# class TimeWeightedInterestRateParams:
#     protocol_fee: float = 0.1
#     min_apr: float = 0.005
#     max_apr: float = 100.0131
#     min_utilization: float = 0.75
#     max_utilization: float = 0.85
#     rate_half_life_days: float = 0.5
#     util_precision: float = 1e5
#     rate_precision: float = 1e18
#
# def apr_to_per_sec(apr: float) -> float:
#     return apr / SECONDS_PER_YEAR
#
# def per_sec_to_apr(rate_per_sec: float) -> float:
#     return rate_per_sec * SECONDS_PER_YEAR
#
# # === 1) Восстановить старый full_util_rate из наблюдаемых (old_util, old_lend_apr) ===
# def infer_old_full_from_observation(params: InterestRateParams,
#                                     old_util: float,
#                                     old_lend_apr: float) -> float:
#     old_borrow_apr = old_lend_apr / max(old_util, 1e-12)
#     r_obs = apr_to_per_sec(old_borrow_apr)
#     z = apr_to_per_sec(params.ZERO_UTIL_RATE)
#     p = params.VERTEX_RATE_PERCENT
#     u = old_util
#     u_v = params.VERTEX_UTILIZATION
#
#     if u < u_v:
#         denom = (u / u_v) * p
#         x = z + (r_obs - z) / max(denom, 1e-18)
#     else:
#         a = (u - u_v) / max(1 - u_v, 1e-18)
#         coef_x = (p + a * (1 - p))
#         const = (1 - p) * z * (1 - a)
#         x = (r_obs - const) / max(coef_x, 1e-18)
#
#     x = min(max(x, apr_to_per_sec(params.MIN_FULL_UTIL_RATE)),
#             apr_to_per_sec(params.MAX_FULL_UTIL_RATE))
#     return x
#
# # === 2) Обновление full_util_rate по half-life + расхождению утилы от целевого коридора ===
# def get_full_util_rate(params: InterestRateParams,
#                        delta_time: float,
#                        utilization: float,
#                        old_full_rate_per_sec: float) -> float:
#     if utilization < params.MIN_TARGET_UTIL:
#         delta_u = params.MIN_TARGET_UTIL - utilization
#         decay_growth = params.RATE_HALF_LIFE + (delta_u ** 2) * delta_time
#         new_full_rate = old_full_rate_per_sec * (params.RATE_HALF_LIFE / decay_growth)
#     elif utilization > params.MAX_TARGET_UTIL:
#         delta_u = utilization - params.MAX_TARGET_UTIL
#         decay_growth = params.RATE_HALF_LIFE + (delta_u ** 2) * delta_time
#         new_full_rate = old_full_rate_per_sec * (decay_growth / params.RATE_HALF_LIFE)
#     else:
#         new_full_rate = old_full_rate_per_sec
#     return min(max(new_full_rate, apr_to_per_sec(params.MIN_FULL_UTIL_RATE)),
#                apr_to_per_sec(params.MAX_FULL_UTIL_RATE))
#
# # === 3) Borrow per second по piecewise-линейке; затем Lend = Borrow * Util ===
# def get_new_rates(params: InterestRateParams,
#                   delta_time: float,
#                   utilization: float,
#                   old_full_rate_per_sec: float):
#     new_full = get_full_util_rate(params, delta_time, utilization, old_full_rate_per_sec)
#     z = apr_to_per_sec(params.ZERO_UTIL_RATE)
#     p = params.VERTEX_RATE_PERCENT
#     u_v = params.VERTEX_UTILIZATION
#     v = ((new_full - z) * p) + z
#     if utilization < u_v:
#         r = z + (utilization / u_v) * (v - z)
#     else:
#         r = v + ((utilization - u_v) / max(1 - u_v, 1e-18)) * (new_full - v)
#     borrow_apr = per_sec_to_apr(r)
#     lend_apr = borrow_apr * utilization
#     return lend_apr, borrow_apr, per_sec_to_apr(new_full)
#
# # === 4) Удобная обёртка под ваш сценарий ===
# def compute_new_lend_apr(params: InterestRateParams,
#                          old_util: float,
#                          old_lend_apr: float,
#                          new_util: float,
#                          delta_t: float):
#     old_full_sec = infer_old_full_from_observation(params, old_util, old_lend_apr)
#     new_lend_apr, new_borrow_apr, new_full_apr = get_new_rates(params, delta_t, new_util, old_full_sec)
#     return {
#         "new_lend_apr": new_lend_apr,
#         "new_borrow_apr": new_borrow_apr,
#         "new_full_util_apr": new_full_apr
#     }
#
# class VariableInterestRate:
#     def __init__(self, params: InterestRateParams):
#         self.params = params
#
#     def calculate_old_full_utilization_interest(self, old_lend_apr: float, old_utilization: float) -> float:
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (SECONDS_PER_YEAR * 100)
#         term = ((old_borrow_rate_per_sec - apr_to_per_sec(self.params.ZERO_UTIL_RATE)) *
#                 self.params.VERTEX_UTILIZATION / old_utilization if old_utilization != 0 else 0)
#         vertex_interest = term + apr_to_per_sec(self.params.ZERO_UTIL_RATE)
#         full_utilization_interest = ((vertex_interest - apr_to_per_sec(self.params.ZERO_UTIL_RATE)) /
#                                     self.params.VERTEX_RATE_PERCENT) + apr_to_per_sec(self.params.ZERO_UTIL_RATE)
#         return per_sec_to_apr(full_utilization_interest)
#
#     def get_full_utilization_interest(self, delta_time: float, utilization: float,
#                                       full_utilization_interest: float) -> float:
#         if utilization < self.params.MIN_TARGET_UTIL:
#             delta_utilization = ((self.params.MIN_TARGET_UTIL - utilization) * 1e18) / self.params.MIN_TARGET_UTIL
#             decay_growth = (self.params.RATE_HALF_LIFE * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * (self.params.RATE_HALF_LIFE * 1e36)) / decay_growth
#         elif utilization > self.params.MAX_TARGET_UTIL:
#             delta_utilization = ((utilization - self.params.MAX_TARGET_UTIL) * 1e18) / (1 - self.params.MAX_TARGET_UTIL)
#             decay_growth = (self.params.RATE_HALF_LIFE * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * decay_growth) / (self.params.RATE_HALF_LIFE * 1e36)
#         else:
#             new_full_utilization_interest = full_utilization_interest
#         return min(max(new_full_utilization_interest, self.params.MIN_FULL_UTIL_RATE), self.params.MAX_FULL_UTIL_RATE)
#
#     def get_new_rate(self, delta_time: float, utilization: float, old_full_utilization_interest: float) -> Tuple[float, float]:
#         new_full_utilization_interest = self.get_full_utilization_interest(delta_time, utilization, old_full_utilization_interest)
#         vertex_interest = ((apr_to_per_sec(new_full_utilization_interest) - apr_to_per_sec(self.params.ZERO_UTIL_RATE)) *
#                           self.params.VERTEX_RATE_PERCENT) + apr_to_per_sec(self.params.ZERO_UTIL_RATE)
#         if utilization < self.params.VERTEX_UTILIZATION:
#             new_rate_per_sec = (apr_to_per_sec(self.params.ZERO_UTIL_RATE) +
#                                (utilization / self.params.VERTEX_UTILIZATION) * (vertex_interest - apr_to_per_sec(self.params.ZERO_UTIL_RATE)))
#         else:
#             new_rate_per_sec = (vertex_interest + ((utilization - self.params.VERTEX_UTILIZATION) /
#                                                   (1.0 - self.params.VERTEX_UTILIZATION)) *
#                                (apr_to_per_sec(new_full_utilization_interest) - vertex_interest))
#         return new_rate_per_sec, new_full_utilization_interest
#
# class TimeWeightedVariableInterestRate:
#     def __init__(self, params: TimeWeightedInterestRateParams):
#         self.params = params
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100 / SECONDS_PER_YEAR
#
#     def calculate_new_full_utilization_rate(self, old_utilization: float, old_lend_apr: float,
#                                             delta_time: float) -> float:
#         old_rate_per_sec = self.calculate_rate_per_sec(old_lend_apr)
#         old_full_util_rate = old_rate_per_sec / (old_utilization * (1 - self.params.protocol_fee)) if old_utilization != 0 else 0
#
#         if old_utilization < self.params.min_utilization:
#             delta_utilization = ((self.params.min_utilization - old_utilization) *
#                                 self.params.rate_precision) / self.params.min_utilization
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * (self.rate_half_life_secs * 1e36)) / decay_growth
#         elif old_utilization > self.params.max_utilization:
#             delta_utilization = ((old_utilization - self.params.max_utilization) *
#                                 self.params.rate_precision) / (self.params.util_precision - self.params.max_utilization)
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * decay_growth) / (self.rate_half_life_secs * 1e36)
#         else:
#             new_full_util_rate = old_full_util_rate
#
#         min_rate_per_sec = self.calculate_rate_per_sec(self.params.min_apr)
#         max_rate_per_sec = self.calculate_rate_per_sec(self.params.max_apr)
#         return min(max(new_full_util_rate, min_rate_per_sec), max_rate_per_sec)
#
#     def get_new_lend_apr(self, delta_time: float, current_utilization: float, old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * SECONDS_PER_YEAR * 100
#         return lend_apr
#
# def get_pair_links(driver, max_retries=3) -> List[str]:
#     """Получает список ссылок на пары с сайта Fraxlend."""
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
#
# def fetch_metrics(driver, url: str, timeout=15, max_retries=2) -> Dict:
#     """Извлекает метрики для указанной пары."""
#     logger.info(f"Загрузка страницы: {url}")
#     if url in BLACKLISTED_PAIRS:
#         logger.info(f"Пара {url} в черном списке, пропускаем")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
#     for attempt in range(max_retries):
#         try:
#             start_time = time.time()
#             process = psutil.Process()
#             peak_memory = process.memory_info().rss / 1024 / 1024
#             driver.get(url)
#             WebDriverWait(driver, timeout).until(
#                 EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#             if "Frax Facts" not in driver.page_source:
#                 raise Exception("Страница не содержит ожидаемого содержимого 'Frax Facts'")
#             with open(f"page_{url.split('/')[-1]}.html", "w") as f:
#                 f.write(driver.page_source)
#             logger.info(f"Сохранён HTML страницы {url} для отладки")
#             labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type",
#                       "Collateral"]
#             data = {"Link": url}
#             for label in labels:
#                 try:
#                     if label == "Collateral":
#                         possible_labels = ["Collateral", "Collateral Token", "Collateral Asset"]
#                         for possible_label in possible_labels:
#                             try:
#                                 el = driver.find_element(By.XPATH,
#                                                        f"//div[contains(text(), '{possible_label}')]/following-sibling::div")
#                                 data[label] = el.text.strip()
#                                 logger.info(f"Найдено значение для {possible_label}: {data[label]}")
#                                 break
#                             except:
#                                 continue
#                         else:
#                             data[label] = "N/A"
#                             logger.warning(f"Не удалось найти значение для Collateral на {url}")
#                     else:
#                         el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                         data[label] = el.text.strip()
#                     logger.info(f"Извлечено значение для {label}: {data[label]}")
#                 except Exception as e:
#                     data[label] = "N/A"
#                     logger.warning(f"Не удалось извлечь {label} для {url}: {type(e).__name__}: {e}")
#             logger.info(f"Данные для {url}: {data}")
#             logger.info(f"Время обработки {url}: {time.time() - start_time:.2f} секунд")
#             peak_memory = max(peak_memory, process.memory_info().rss / 1024 / 1024)
#             logger.info(f"Пиковая память во время обработки {url}: {peak_memory:.2f} MB")
#             return data
#         except TimeoutException as e:
#             logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при загрузке данных для {url}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error(f"Пара {url} не загрузилась после {max_retries} попыток, добавляем в черный список")
#                 BLACKLISTED_PAIRS.add(url)
#                 send_to_telegram(f"Пара {url} добавлена в черный список после {max_retries} неудачных попыток загрузки")
#                 return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                         "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
#             time.sleep(2)
#         except Exception as e:
#             logger.error(
#                 f"Попытка {attempt + 1}/{max_retries}: Ошибка при загрузке данных для {url}: {type(e).__name__}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error(f"Пара {url} не загрузилась после {max_retries} попыток, добавляем в черный список")
#                 BLACKLISTED_PAIRS.add(url)
#                 send_to_telegram(f"Пара {url} добавлена в черный список после ошибки: {type(e).__name__}: {e}")
#                 return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                         "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A", "Collateral": "N/A"}
#             time.sleep(2)
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     """Парсит строковое представление суммы в долларах."""
#     if amount_str == "N/A" or not amount_str:
#         logger.warning(f"Невозможно распарсить сумму: {amount_str}, возвращается 0.0")
#         return 0.0
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if 'm' in amount_str.lower():
#             return value * 1_000_000  # Предполагается, что 'm' всегда означает миллионы
#         elif 'k' in amount_str.lower():
#             return value * 1_000
#         return value
#     except ValueError:
#         logger.error(f"Ошибка парсинга суммы: {amount_str}, возвращается 0.0")
#         return 0.0
#
# def calculate_pair_profit(data: Dict, v1_model, v2_params, investment: float, bonus: float = 0.0,
#                           delta_time: float = 86400.0) -> Tuple[float, float, float, Dict]:
#     """Рассчитывает прибыль, новую Lend APR и утилизацию для пары при заданной инвестиции."""
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
#         utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         pair_address = data.get("Link", "").split("/")[-1].lower()
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         send_to_telegram(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         return 0.0, 0.0, 0.0, data
#
#     if lend_apr <= MIN_APR:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_APR}%)")
#         return 0.0, 0.0, 0.0, data
#     if utilization >= 1.01:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization * 100}% >= 101%)")
#         return 0.0, 0.0, 0.0, data
#     if reserve_size == 0:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
#         return 0.0, 0.0, 0.0, data
#     if investment > available_liquidity:
#         logger.info(
#             f"Пара отфильтрована: {data.get('Link')} (Investment={investment} > Available Liquidity={available_liquidity})")
#         return 0.0, 0.0, 0.0, data
#
#     new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
#     logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")
#
#     if rate_type == "Variable V2":
#         if new_utilization < 0.76:
#             logger.debug(f"Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
#             return 0.0, 0.0, 0.0, data
#         res = compute_new_lend_apr(v2_params, utilization, lend_apr, new_utilization, delta_time)
#         new_lend_apr = res["new_lend_apr"]
#         if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
#             new_lend_apr += bonus
#         daily_profit = (investment * new_lend_apr / 100) / 365.24
#     elif rate_type == "Variable V1":
#         new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
#         if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
#             new_lend_apr += bonus
#         daily_profit = (investment * new_lend_apr / 100) / 365.24
#     else:
#         logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return 0.0, 0.0, 0.0, data
#
#     return daily_profit, new_lend_apr, new_utilization, data
#
#
# def optimize_investment_distribution(pairs: List[Dict], v1_model, v2_model, total_investment: float, bonus: float) -> List[Dict]:
#     """Оптимизирует распределение инвестиций между парами для максимизации дневной прибыли."""
#     pair_profits = []
#     for data in pairs:
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         max_investment = min(total_investment, available_liquidity)
#         investments = range(MIN_INVESTMENT, int(max_investment) + 1, INVESTMENT_STEP)
#
#         for investment in investments:
#             daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(data, v1_model, v2_model,
#                                                                                       investment, bonus)
#             if daily_profit > 0:
#                 pair_profits.append({
#                     "data": data,
#                     "daily_profit": daily_profit,
#                     "new_lend_apr": new_lend_apr,
#                     "new_utilization": new_utilization,
#                     "investment": investment,
#                     "profit_per_dollar": daily_profit / investment if investment > 0 else 0
#                 })
#
#     if not pair_profits:
#         logger.info("Нет подходящих пар для инвестиций")
#         return []
#
#     # Инициализируем лучшее распределение
#     best_allocation = []
#     max_total_profit = 0.0
#
#     # Проверяем вложение всей суммы в одну пару
#     for data in pairs:
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         if available_liquidity >= total_investment:
#             daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(data, v1_model, v2_model,
#                                                                                       total_investment, bonus)
#             if daily_profit > max_total_profit:
#                 max_total_profit = daily_profit
#                 best_allocation = [{
#                     "data": data,
#                     "investment": total_investment,
#                     "daily_profit": daily_profit,
#                     "new_lend_apr": new_lend_apr,
#                     "new_utilization": new_utilization
#                 }]
#                 logger.info(
#                     f"Обновлено лучшее вложение: ${total_investment:,.2f} в {data['Link']}, прибыль: ${daily_profit:,.2f}")
#
#     # Итеративное распределение по парам
#     remaining_investment = total_investment
#     allocated_pairs = []
#     used_urls = set()
#
#     while remaining_investment >= MIN_INVESTMENT and pair_profits:
#         # Сортируем по прибыли на доллар, исключая уже использованные пары
#         valid_profits = [p for p in pair_profits if p["data"]["Link"] not in used_urls]
#         if not valid_profits:
#             break
#         valid_profits.sort(key=lambda x: x["profit_per_dollar"], reverse=True)
#         best_option = valid_profits[0]
#         pair_url = best_option["data"]["Link"]
#         available_liquidity = parse_dollar_amount(best_option["data"].get("Available Liquidity", "0"))
#
#         # Проверяем разные суммы инвестиций для этой пары
#         max_investment = min(remaining_investment, available_liquidity)
#         best_investment = 0
#         best_profit = 0
#         best_lend_apr = 0
#         best_utilization = 0
#         for investment in range(MIN_INVESTMENT, int(max_investment) + 1, INVESTMENT_STEP):
#             daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(
#                 best_option["data"], v1_model, v2_model, investment, bonus
#             )
#             if daily_profit > best_profit:
#                 best_profit = daily_profit
#                 best_investment = investment
#                 best_lend_apr = new_lend_apr
#                 best_utilization = new_utilization
#
#         if best_profit > 0:
#             allocated_pairs.append({
#                 "data": best_option["data"],
#                 "investment": best_investment,
#                 "daily_profit": best_profit,
#                 "new_lend_apr": best_lend_apr,
#                 "new_utilization": best_utilization
#             })
#             remaining_investment -= best_investment
#             used_urls.add(pair_url)
#             logger.info(f"Инвестировано ${best_investment:,.2f} в {pair_url}, прибыль: ${best_profit:,.2f}")
#         else:
#             break
#
#     # Сравниваем с лучшим вложением в одну пару
#     total_profit = sum(p["daily_profit"] for p in allocated_pairs)
#     if total_profit > max_total_profit:
#         best_allocation = allocated_pairs
#         max_total_profit = total_profit
#         logger.info(f"Обновлено лучшее распределение: {len(best_allocation)} пар, прибыль: ${total_profit:,.2f}")
#
#     if not best_allocation:
#         logger.info("Не найдено подходящих комбинаций для инвестиций")
#         return []
#
#     remaining_investment = total_investment - sum(p["investment"] for p in best_allocation)
#     if remaining_investment > 0:
#         logger.info(f"Остаток нераспределенных средств: ${remaining_investment:,.2f}")
#         send_to_telegram(f"Остаток нераспределенных средств: ${remaining_investment:,.2f}")
#
#     return best_allocation
#
#
# def send_to_telegram(message: str):
#     """Отправляет сообщение в Telegram."""
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         try:
#             payload = {
#                 "chat_id": chat_id,
#                 "text": message[:4096]
#             }
#             response = requests.post(url, data=payload, timeout=10)
#             if response.status_code != 200:
#                 logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#             else:
#                 logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
#         except requests.RequestException as e:
#             logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {type(e).__name__}: {e}")
#
#
# def kill_chromedriver():
#     """Принудительно завершает все процессы chromedriver."""
#     try:
#         subprocess.run(['pkill', '-f', 'chromedriver'], check=True)
#         logger.info("Все процессы chromedriver завершены")
#     except subprocess.CalledProcessError:
#         logger.info("Процессы chromedriver не найдены")
#     except Exception as e:
#         logger.error(f"Ошибка при завершении chromedriver: {type(e).__name__}: {e}")
#
#
# def get_driver(chromedriver_path: str):
#     """Инициализирует WebDriver с заданными настройками."""
#     options = Options()
#     options.add_argument("--headless=new")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--window-size=1920,1080")
#     options.add_argument(
#         "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) "
#         "Chrome/115.0 Safari/537.36"
#     )
#     driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
#     return driver
#
#
# def get_fraxlend_fxs_lower_bound(driver) -> float:
#     """Получает нижнюю границу APR для Fraxlend V1 FRAX/FXS."""
#     url = "https://app.frax.finance/staking/overview"
#     logger.info(f"Загрузка страницы для получения бонуса: {url}")
#     try:
#         driver.get(url)
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Fraxlend V1 FRAX/FXS')]"))
#         )
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         target_text = re.compile(r"Fraxlend\s*V1\s*FRAX/FXS", re.IGNORECASE)
#         fees_range = None
#
#         for element in soup.find_all(string=target_text):
#             parent = element.find_parent()
#             if parent:
#                 for sibling in parent.find_all_next(string=True, limit=10):
#                     if re.search(r'\d+\.\d+%\s*-\s*\d+\.\d+%', sibling):
#                         fees_range = sibling.strip()
#                         break
#                 if fees_range:
#                     break
#
#         if fees_range:
#             match = re.search(r'(\d+\.\d+)%\s*-\s*\d+\.\d+%', fees_range)
#             if match:
#                 logger.info(f"Извлечена нижняя граница APR: {match.group(1)}%")
#                 return float(match.group(1))
#             else:
#                 logger.warning(f"Не удалось извлечь нижнюю границу из диапазона: {fees_range}")
#                 return 0.0
#         else:
#             logger.warning(f"Не найдено 'Fraxlend V1 FRAX/FXS' или связанный диапазон APR на {url}")
#             return 0.0
#
#     except Exception as e:
#         logger.error(f"Ошибка получения бонуса для Fraxlend V1 FRAX/FXS: {type(e).__name__}: {e}")
#         return 0.0
#
#
# def process_pairs():
#     """Обрабатывает пары Fraxlend и распределяет инвестиции для максимизации прибыли."""
#     logger.info(
#         f"Запуск функции process_pairs, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#     send_to_telegram("Сервер запущен, начинаем парсинг")
#     chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
#     logger.info(f"Используемый путь к chromedriver: {chromedriver_path}")
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     iteration_count = 0
#
#     while True:
#         iteration_start_time = time.time()
#         iteration_count += 1
#         peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
#
#         processed_urls = set()
#         logger.info(f"Сброс processed_urls для цикла #{iteration_count}")
#
#         logger.info(f"Текущий черный список: {BLACKLISTED_PAIRS}")
#         send_to_telegram(f"Начало цикла #{iteration_count}. Черный список: {BLACKLISTED_PAIRS} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#
#         if iteration_count % 24 == 0:
#             logger.info("Сброс черного списка")
#             BLACKLISTED_PAIRS.clear()
#             send_to_telegram("Черный список сброшен")
#
#         driver = None
#         try:
#             logger.info(f"Инициализация WebDriver для цикла #{iteration_count}")
#             kill_chromedriver()
#             driver = get_driver(chromedriver_path)
#             logger.info("WebDriver успешно инициализирован")
#
#             pair_links = get_pair_links(driver)
#             logger.info(f"Полученные пары: {pair_links}")
#
#             if not pair_links:
#                 logger.warning("Список пар пуст. Пропускаем итерацию.")
#                 send_to_telegram(f"Предупреждение: Список пар пуст. Проверьте сайт или селектор. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#                 time.sleep(CHECK_INTERVAL)
#                 continue
#
#             frax_fxs_pair = "https://facts.frax.finance/fraxlend/pairs/0xdbe88dbac39263c47629ebba02b3ef4cf0752a72"
#             bonus = get_fraxlend_fxs_lower_bound(driver) if frax_fxs_pair in pair_links else 0.0
#             logger.info(f"Bonus for FRAX/FXS: {bonus}%")
#
#             pairs_data = []
#             skipped_pairs = []
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if time.time() - iteration_start_time > MAX_ITERATION_TIME:
#                     logger.warning(
#                         f"Превышено максимальное время итерации ({MAX_ITERATION_TIME} секунд). Пропускаем оставшиеся пары.")
#                     send_to_telegram(
#                         f"Превышено время итерации ({MAX_ITERATION_TIME} секунд). Пропущено {len(pair_links) - pair_links.index(url)} пар. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#                     break
#
#                 if url in processed_urls or url in BLACKLISTED_PAIRS:
#                     logger.info(f"Пропущена пара (уже обработана или в черном списке): {url}")
#                     skipped_pairs.append(url)
#                     continue
#
#                 logger.info(f"Обработка пары: {url}")
#                 data = fetch_metrics(driver, url)
#                 logger.info(f"Полученные данные для {url}: {data}")
#
#                 if all(data.get(label, "N/A") == "N/A" for label in
#                        ["Available Liquidity", "Utilization Rate", "Lend APR", "Reserve Size"]):
#                     logger.info(f"Пропущена пара из-за некорректных данных: {url}")
#                     processed_urls.add(url)
#                     send_to_telegram(
#                         f"Пропущена пара {url} (Collateral: {data.get('Collateral', 'N/A')}) из-за некорректных данных: {data} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#                     continue
#
#                 pairs_data.append(data)
#                 processed_urls.add(url)
#
#             if pairs_data:
#                 allocated_pairs = optimize_investment_distribution(pairs_data, v1_model, v2_model, TOTAL_INVESTMENT,
#                                                                    bonus)
#                 if allocated_pairs:
#                     total_daily_profit = sum(p["daily_profit"] for p in allocated_pairs)
#                     message = f"Результаты распределения инвестиций (${TOTAL_INVESTMENT:,.2f}):\n"
#                     for pair in allocated_pairs:
#                         data = pair["data"]
#                         message += (
#                             f"\nПара: {data.get('Collateral', 'N/A')} ({data.get('Rate Type', 'N/A')})\n"
#                             f"Ссылка: {data.get('Link')}\n"
#                             f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST\n"
#                             f"Текущая Lend APR: {data.get('Lend APR')} {'+' + str(bonus) + '%' if data.get('Link').split('/')[-1].lower() == '0xdbe88dbac39263c47629ebba02b3ef4cf0752a72' else ''}\n"
#                             f"Новая Lend APR: {pair['new_lend_apr'] * 100:.2f}%\n"
#                             f"Инвестировано: ${pair['investment']:,.2f}\n"
#                             f"Доход за день: ${pair['daily_profit']:,.2f}\n"
#                             f"Новая утилизация: {pair['new_utilization'] * 100:.2f}%\n"
#                             f"Ликвидность: {data.get('Available Liquidity')}\n"
#                             f"Текущая утилизация: {data.get('Utilization Rate')}\n"
#                             f"Borrow APR: {data.get('Borrow APR')}\n"
#                             f"Резерв: {data.get('Reserve Size')}\n"
#                         )
#                     message += f"\nОбщий доход за день: ${total_daily_profit:,.2f}"
#                     send_to_telegram(message)
#                 else:
#                     send_to_telegram("Не найдено подходящих пар для инвестиций. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#             else:
#                 send_to_telegram("Нет данных о пар для обработки. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#
#             if skipped_pairs:
#                 send_to_telegram(f"Пропущенные пары в цикле #{iteration_count}: {', '.join(skipped_pairs)} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки пар: {type(e).__name__}: {e}")
#             send_to_telegram(f"Ошибка при обработке пар: {type(e).__name__}: {e} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
#
#         finally:
#             if driver:
#                 try:
#                     driver.quit()
#                 except WebDriverException:
#                     logger.info("Игнорируется WebDriverException при закрытии WebDriver")
#                 kill_chromedriver()
#                 logger.info(f"WebDriver закрыт, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#
#             elapsed_time = time.time() - iteration_start_time
#             logger.info(f"Итерация завершена за {elapsed_time:.2f} секунд")
#             logger.info(f"Пиковая память в итерации: {peak_memory:.2f} MB")
#             logger.info(f"Ожидание {CHECK_INTERVAL} секунд перед следующей итерацией")
#             remaining_time = CHECK_INTERVAL
#             while remaining_time > 0:
#                 sleep_time = min(30, remaining_time)
#                 time.sleep(sleep_time)
#                 remaining_time -= sleep_time
#                 peak_memory = max(peak_memory, psutil.Process().memory_info().rss / 1024 / 1024)
#                 logger.info(
#                     f"Ожидание, осталось {remaining_time} секунд, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB, пиковая память: {peak_memory:.2f} MB")
#
#
# def main():
#     """Основная функция для запуска обработки пар."""
#     logger.info("Запуск Background Worker")
#     send_to_telegram("Background Worker запущен (Время: 07:12 PM EEST, Thursday, August 21, 2025)")
#     process_pairs()
#
#
# if __name__ == "__main__":
#     main()

import os
import time
import requests
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchDriverException
import logging
import psutil
import subprocess
import re
from bs4 import BeautifulSoup

# Constants
MIN_APR = 5  # Threshold for Lend APR
TOTAL_INVESTMENT = 200_000  # Total investment amount
INVESTMENT_STEP = 5_000  # Investment step for optimization
MIN_INVESTMENT = 3_000  # Minimum investment amount
SECONDS_PER_YEAR = 365.24 * 24 * 3600  # Global constant
CHECK_INTERVAL = 60 * 60  # Check interval in seconds (1 hour)
MAX_PAIRS = 100  # Maximum number of pairs
MAX_ITERATION_TIME = 1200  # Maximum iteration time (10 minutes)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Telegram settings
BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"
CHAT_IDS = [6192278046, 306507209]

# Blacklist for problematic pairs
BLACKLISTED_PAIRS = set()

# === Updated parameters for Variable Rate V2 ===
@dataclass
class InterestRateParams:
    MIN_TARGET_UTIL: float = 0.75
    MAX_TARGET_UTIL: float = 0.85
    VERTEX_UTILIZATION: float = 0.875
    ZERO_UTIL_RATE: float = 0.005  # 0.5%
    MIN_FULL_UTIL_RATE: float = 0.0499  # 4.99%
    MAX_FULL_UTIL_RATE: float = 99.8779  # 9,987.79%
    RATE_HALF_LIFE: float = 172800.0  # 2 days
    VERTEX_RATE_PERCENT: float = 0.2039  # 20.39%

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

def apr_to_per_sec(apr: float) -> float:
    return apr / SECONDS_PER_YEAR

def per_sec_to_apr(rate_per_sec: float) -> float:
    return rate_per_sec * SECONDS_PER_YEAR

# === 1) Restore old full_util_rate from observed (old_util, old_lend_apr) ===
def infer_old_full_from_observation(params: InterestRateParams,
                                    old_util: float,
                                    old_lend_apr: float) -> float:
    old_borrow_apr = old_lend_apr / max(old_util, 1e-12)
    r_obs = apr_to_per_sec(old_borrow_apr)
    z = apr_to_per_sec(params.ZERO_UTIL_RATE)
    p = params.VERTEX_RATE_PERCENT
    u = old_util
    u_v = params.VERTEX_UTILIZATION

    if u < u_v:
        denom = (u / u_v) * p
        x = z + (r_obs - z) / max(denom, 1e-18)
    else:
        a = (u - u_v) / max(1 - u_v, 1e-18)
        coef_x = (p + a * (1 - p))
        const = (1 - p) * z * (1 - a)
        x = (r_obs - const) / max(coef_x, 1e-18)

    x = min(max(x, apr_to_per_sec(params.MIN_FULL_UTIL_RATE)),
            apr_to_per_sec(params.MAX_FULL_UTIL_RATE))
    return x

# === 2) Update full_util_rate based on half-life + utilization deviation from target range ===
def get_full_util_rate(params: InterestRateParams,
                       delta_time: float,
                       utilization: float,
                       old_full_rate_per_sec: float) -> float:
    if utilization < params.MIN_TARGET_UTIL:
        delta_u = params.MIN_TARGET_UTIL - utilization
        decay_growth = params.RATE_HALF_LIFE + (delta_u ** 2) * delta_time
        new_full_rate = old_full_rate_per_sec * (params.RATE_HALF_LIFE / decay_growth)
    elif utilization > params.MAX_TARGET_UTIL:
        delta_u = utilization - params.MAX_TARGET_UTIL
        decay_growth = params.RATE_HALF_LIFE + (delta_u ** 2) * delta_time
        new_full_rate = old_full_rate_per_sec * (decay_growth / params.RATE_HALF_LIFE)
    else:
        new_full_rate = old_full_rate_per_sec
    return min(max(new_full_rate, apr_to_per_sec(params.MIN_FULL_UTIL_RATE)),
               apr_to_per_sec(params.MAX_FULL_UTIL_RATE))

# === 3) Borrow per second using piecewise-linear model; then Lend = Borrow * Util ===
def get_new_rates(params: InterestRateParams,
                  delta_time: float,
                  utilization: float,
                  old_full_rate_per_sec: float):
    new_full = get_full_util_rate(params, delta_time, utilization, old_full_rate_per_sec)
    z = apr_to_per_sec(params.ZERO_UTIL_RATE)
    p = params.VERTEX_RATE_PERCENT
    u_v = params.VERTEX_UTILIZATION
    v = ((new_full - z) * p) + z
    if utilization < u_v:
        r = z + (utilization / u_v) * (v - z)
    else:
        r = v + ((utilization - u_v) / max(1 - u_v, 1e-18)) * (new_full - v)
    borrow_apr = per_sec_to_apr(r)
    lend_apr = borrow_apr * utilization
    return lend_apr, borrow_apr, per_sec_to_apr(new_full)

# === 4) Wrapper for your scenario ===
def compute_new_lend_apr(params: InterestRateParams,
                         old_util: float,
                         old_lend_apr: float,
                         new_util: float,
                         delta_t: float):
    old_full_sec = infer_old_full_from_observation(params, old_util, old_lend_apr)
    new_lend_apr, new_borrow_apr, new_full_apr = get_new_rates(params, delta_t, new_util, old_full_sec)
    return {
        "new_lend_apr": new_lend_apr,
        "new_borrow_apr": new_borrow_apr,
        "new_full_util_apr": new_full_apr
    }

class TimeWeightedVariableInterestRate:
    def __init__(self, params: TimeWeightedInterestRateParams):
        self.params = params
        self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600

    def calculate_rate_per_sec(self, apr: float) -> float:
        return apr / 100 / SECONDS_PER_YEAR

    def calculate_new_full_utilization_rate(self, old_utilization: float, old_lend_apr: float,
                                            delta_time: float) -> float:
        old_rate_per_sec = self.calculate_rate_per_sec(old_lend_apr)
        old_full_util_rate = old_rate_per_sec / (old_utilization * (1 - self.params.protocol_fee)) if old_utilization != 0 else 0

        if old_utilization < self.params.min_utilization:
            delta_utilization = ((self.params.min_utilization - old_utilization) *
                                self.params.rate_precision) / self.params.min_utilization
            decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
            new_full_util_rate = (old_full_util_rate * (self.rate_half_life_secs * 1e36)) / decay_growth
        elif old_utilization > self.params.max_utilization:
            delta_utilization = ((old_utilization - self.params.max_utilization) *
                                self.params.rate_precision) / (self.params.util_precision - self.params.max_utilization)
            decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
            new_full_util_rate = (old_full_util_rate * decay_growth) / (self.rate_half_life_secs * 1e36)
        else:
            new_full_util_rate = old_full_util_rate

        min_rate_per_sec = self.calculate_rate_per_sec(self.params.min_apr)
        max_rate_per_sec = self.calculate_rate_per_sec(self.params.max_apr)
        return min(max(new_full_util_rate, min_rate_per_sec), max_rate_per_sec)

    def get_new_lend_apr(self, delta_time: float, current_utilization: float, old_lend_apr: float,
                         old_utilization: float) -> float:
        full_utilization_rate = self.calculate_new_full_utilization_rate(old_utilization, old_lend_apr, delta_time)
        lend_rate_per_sec = full_utilization_rate * current_utilization * (1 - self.params.protocol_fee) if current_utilization != 0 else 0
        lend_apr = lend_rate_per_sec * SECONDS_PER_YEAR * 100
        return lend_apr

def get_pair_links(driver, max_retries=3) -> List[str]:
    """Получает список ссылок на пары с сайта Fraxlend."""
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

def fetch_metrics(driver, url: str, timeout=15, max_retries=2) -> Dict:
    """Извлекает метрики для указанной пары."""
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

def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
    """Parses a string representation of a dollar amount."""
    if amount_str == "N/A" or not amount_str:
        logger.warning(f"Unable to parse amount: {amount_str}, returning 0.0")
        return 0.0
    try:
        cleaned = re.sub(r'[^\d.]', '', amount_str)
        value = float(cleaned)
        if 'm' in amount_str.lower():
            return value * 1_000_000  # Assumes 'm' always means millions
        elif 'k' in amount_str.lower():
            return value * 1_000
        return value
    except ValueError:
        logger.error(f"Error parsing amount: {amount_str}, returning 0.0")
        return 0.0

def calculate_pair_profit(data: Dict, v1_model, v2_params, investment: float, bonus: float = 0.0,
                          delta_time: float = 86400.0) -> Tuple[float, float, float, Dict]:
    """Calculates profit, new Lend APR, and utilization for a pair with the given investment."""
    try:
        lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
        utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
        lend_apr = float(lend_apr_str) if lend_apr_str != "N/A" else 0.0
        utilization = float(utilization_str) / 100 if utilization_str != "N/A" else 0.0
        available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
        reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
        rate_type = data.get("Rate Type", "N/A")
        pair_address = data.get("Link", "").split("/")[-1].lower()
        logger.info(
            f"Parsed data: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
    except Exception as e:
        logger.error(f"Error parsing data for {data.get('Link')}: {e}")
        send_to_telegram(f"Error parsing data for {data.get('Link')}: {e}")
        return 0.0, 0.0, 0.0, data

    if lend_apr <= MIN_APR:
        logger.info(f"Pair filtered: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_APR}%)")
        return 0.0, 0.0, 0.0, data
    if utilization >= 1.01:
        logger.info(f"Pair filtered: {data.get('Link')} (Utilization Rate={utilization * 100}% >= 101%)")
        return 0.0, 0.0, 0.0, data
    if reserve_size == 0:
        logger.info(f"Pair filtered: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
        return 0.0, 0.0, 0.0, data
    if investment > available_liquidity:
        logger.info(
            f"Pair filtered: {data.get('Link')} (Investment={investment} > Available Liquidity={available_liquidity})")
        return 0.0, 0.0, 0.0, data

    new_utilization = 1 - (available_liquidity + investment) / (reserve_size + investment)
    logger.debug(f"Investment={investment}, new_utilization={new_utilization:.4f}")

    if rate_type == "Variable V2":
        if new_utilization < 0.76:
            logger.debug(f"Skipped (V2): new_utilization={new_utilization:.4f} < 0.76")
            return 0.0, 0.0, 0.0, data
        res = compute_new_lend_apr(v2_params, utilization, lend_apr, new_utilization, delta_time)
        new_lend_apr = res["new_lend_apr"]
        if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
            new_lend_apr += bonus
        daily_profit = (investment * new_lend_apr / 100) / 365.24
    elif rate_type == "Variable V1":
        new_lend_apr = v1_model.get_new_lend_apr(delta_time, new_utilization, lend_apr, utilization)
        if pair_address == "0xdbe88dbac39263c47629ebba02b3ef4cf0752a72":
            new_lend_apr += bonus
        daily_profit = (investment * new_lend_apr / 100) / 365.24
    else:
        logger.info(f"Pair filtered: Unsupported Rate Type={rate_type}")
        return 0.0, 0.0, 0.0, data

    return daily_profit, new_lend_apr, new_utilization, data


def optimize_investment_distribution(pairs: List[Dict], v1_model, v2_params, total_investment: float, bonus: float) -> List[Dict]:
    """Optimizes investment distribution among pairs to maximize daily profit."""
    pair_profits = []
    for data in pairs:
        available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
        max_investment = min(total_investment, available_liquidity)
        investments = range(MIN_INVESTMENT, int(max_investment) + 1, INVESTMENT_STEP)

        for investment in investments:
            daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(data, v1_model, v2_params,
                                                                                      investment, bonus)
            if daily_profit > 0:
                pair_profits.append({
                    "data": data,
                    "daily_profit": daily_profit,
                    "new_lend_apr": new_lend_apr,
                    "new_utilization": new_utilization,
                    "investment": investment,
                    "profit_per_dollar": daily_profit / investment if investment > 0 else 0
                })

    if not pair_profits:
        logger.info("No suitable pairs for investment")
        return []

    # Initialize best allocation
    best_allocation = []
    max_total_profit = 0.0

    # Check investing the entire amount in one pair
    for data in pairs:
        available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
        if available_liquidity >= total_investment:
            daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(data, v1_model, v2_params,
                                                                                      total_investment, bonus)
            if daily_profit > max_total_profit:
                max_total_profit = daily_profit
                best_allocation = [{
                    "data": data,
                    "investment": total_investment,
                    "daily_profit": daily_profit,
                    "new_lend_apr": new_lend_apr,
                    "new_utilization": new_utilization
                }]
                logger.info(
                    f"Updated best investment: ${total_investment:,.2f} in {data['Link']}, profit: ${daily_profit:,.2f}")

    # Iterative distribution across pairs
    remaining_investment = total_investment
    allocated_pairs = []
    used_urls = set()

    while remaining_investment >= MIN_INVESTMENT and pair_profits:
        # Sort by profit per dollar, excluding already used pairs
        valid_profits = [p for p in pair_profits if p["data"]["Link"] not in used_urls]
        if not valid_profits:
            break
        valid_profits.sort(key=lambda x: x["profit_per_dollar"], reverse=True)
        best_option = valid_profits[0]
        pair_url = best_option["data"]["Link"]
        available_liquidity = parse_dollar_amount(best_option["data"].get("Available Liquidity", "0"))

        # Check different investment amounts for this pair
        max_investment = min(remaining_investment, available_liquidity)
        best_investment = 0
        best_profit = 0
        best_lend_apr = 0
        best_utilization = 0
        for investment in range(MIN_INVESTMENT, int(max_investment) + 1, INVESTMENT_STEP):
            daily_profit, new_lend_apr, new_utilization, data = calculate_pair_profit(
                best_option["data"], v1_model, v2_params, investment, bonus
            )
            if daily_profit > best_profit:
                best_profit = daily_profit
                best_investment = investment
                best_lend_apr = new_lend_apr
                best_utilization = new_utilization

        if best_profit > 0:
            allocated_pairs.append({
                "data": best_option["data"],
                "investment": best_investment,
                "daily_profit": best_profit,
                "new_lend_apr": best_lend_apr,
                "new_utilization": best_utilization
            })
            remaining_investment -= best_investment
            used_urls.add(pair_url)
            logger.info(f"Invested ${best_investment:,.2f} in {pair_url}, profit: ${best_profit:,.2f}")
        else:
            break

    # Compare with best single-pair investment
    total_profit = sum(p["daily_profit"] for p in allocated_pairs)
    if total_profit > max_total_profit:
        best_allocation = allocated_pairs
        max_total_profit = total_profit
        logger.info(f"Updated best distribution: {len(best_allocation)} pairs, profit: ${total_profit:,.2f}")

    if not best_allocation:
        logger.info("No suitable combinations for investment found")
        return []

    remaining_investment = total_investment - sum(p["investment"] for p in best_allocation)
    if remaining_investment > 0:
        logger.info(f"Remaining unallocated funds: ${remaining_investment:,.2f}")
        send_to_telegram(f"Remaining unallocated funds: ${remaining_investment:,.2f} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")

    return best_allocation


def send_to_telegram(message: str):
    """Sends a message to Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            payload = {
                "chat_id": chat_id,
                "text": message[:4096]
            }
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Error sending to Telegram for chat_id {chat_id}: {response.text}")
            else:
                logger.info(f"Message successfully sent to Telegram for chat_id {chat_id}")
        except requests.RequestException as e:
            logger.error(f"Error sending to Telegram for chat_id {chat_id}: {type(e).__name__}: {e}")


def kill_chromedriver():
    """Forcefully terminates all chromedriver processes."""
    try:
        subprocess.run(['pkill', '-f', 'chromedriver'], check=True)
        logger.info("All chromedriver processes terminated")
    except subprocess.CalledProcessError:
        logger.info("No chromedriver processes found")
    except Exception as e:
        logger.error(f"Error terminating chromedriver: {type(e).__name__}: {e}")


def get_driver(chromedriver_path: str):
    """Initializes WebDriver with specified settings."""
    options = Options()
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
    try:
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
        logger.info(f"WebDriver successfully initialized using {chromedriver_path}")
        return driver
    except NoSuchDriverException as e:
        logger.error(f"WebDriver initialization error: {type(e).__name__}: {e}. Check chromedriver path or install a compatible version.")
        send_to_telegram(f"Error: Unable to initialize WebDriver. Check chromedriver path ({chromedriver_path}) or install a compatible version. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing WebDriver: {type(e).__name__}: {e}")
        send_to_telegram(f"Error: Failed to initialize WebDriver. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
        raise


def get_fraxlend_fxs_lower_bound(driver) -> float:
    """Gets the lower bound APR for Fraxlend V1 FRAX/FXS."""
    url = "https://app.frax.finance/staking/overview"
    logger.info(f"Loading page for bonus: {url}")
    try:
        driver.get(url)
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
                logger.info(f"Extracted lower bound APR: {match.group(1)}%")
                return float(match.group(1))
            else:
                logger.warning(f"Failed to extract lower bound from range: {fees_range}")
                return 0.0
        else:
            logger.warning(f"'Fraxlend V1 FRAX/FXS' or related APR range not found on {url}")
            return 0.0

    except Exception as e:
        logger.error(f"Error getting bonus for Fraxlend V1 FRAX/FXS: {type(e).__name__}: {e}")
        return 0.0


def process_pairs():
    """Processes Fraxlend pairs and distributes investments to maximize profit."""
    logger.info(
        f"Starting process_pairs, memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    send_to_telegram("Server started, beginning parsing (Время: 11:18 AM EEST, Friday, August 22, 2025)")
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    logger.info(f"Using chromedriver path: {chromedriver_path}")

    v1_params = TimeWeightedInterestRateParams()
    v2_params = InterestRateParams()
    v1_model = TimeWeightedVariableInterestRate(v1_params)

    iteration_count = 0

    while True:
        iteration_start_time = time.time()
        iteration_count += 1
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

        processed_urls = set()
        logger.info(f"Reset processed_urls for cycle #{iteration_count}")

        logger.info(f"Current blacklist: {BLACKLISTED_PAIRS}")
        send_to_telegram(f"Starting cycle #{iteration_count}. Blacklist: {BLACKLISTED_PAIRS} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")

        if iteration_count % 24 == 0:
            logger.info("Resetting blacklist")
            BLACKLISTED_PAIRS.clear()
            send_to_telegram(f"Blacklist reset (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")

        driver = None
        try:
            logger.info(f"Initializing WebDriver for cycle #{iteration_count}")
            kill_chromedriver()
            driver = get_driver(chromedriver_path)
            logger.info("WebDriver successfully initialized")

            pair_links = get_pair_links(driver)
            logger.info(f"Retrieved pairs: {pair_links}")

            if not pair_links:
                logger.warning("Pair list is empty. Skipping iteration.")
                send_to_telegram(f"Warning: Pair list is empty. Check website or selector. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
                time.sleep(CHECK_INTERVAL)
                continue

            frax_fxs_pair = "https://facts.frax.finance/fraxlend/pairs/0xdbe88dbac39263c47629ebba02b3ef4cf0752a72"
            bonus = get_fraxlend_fxs_lower_bound(driver) if frax_fxs_pair in pair_links else 0.0
            logger.info(f"Bonus for FRAX/FXS: {bonus}%")

            pairs_data = []
            skipped_pairs = []
            for url in tqdm(pair_links, desc="Processing pairs"):
                if time.time() - iteration_start_time > MAX_ITERATION_TIME:
                    logger.warning(
                        f"Exceeded maximum iteration time ({MAX_ITERATION_TIME} seconds). Skipping remaining pairs.")
                    send_to_telegram(
                        f"Exceeded iteration time ({MAX_ITERATION_TIME} seconds). Skipped {len(pair_links) - pair_links.index(url)} pairs. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
                    break

                if url in processed_urls or url in BLACKLISTED_PAIRS:
                    logger.info(f"Skipped pair (already processed or in blacklist): {url}")
                    skipped_pairs.append(url)
                    continue

                logger.info(f"Processing pair: {url}")
                data = fetch_metrics(driver, url)
                logger.info(f"Retrieved data for {url}: {data}")

                if all(data.get(label, "N/A") == "N/A" for label in
                       ["Available Liquidity", "Utilization Rate", "Lend APR", "Reserve Size"]):
                    logger.info(f"Skipped pair due to invalid data: {url}")
                    processed_urls.add(url)
                    send_to_telegram(
                        f"Skipped pair {url} (Collateral: {data.get('Collateral', 'N/A')}) due to invalid data: {data} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
                    continue

                pairs_data.append(data)
                processed_urls.add(url)

            if pairs_data:
                allocated_pairs = optimize_investment_distribution(pairs_data, v1_model, v2_params, TOTAL_INVESTMENT,
                                                                  bonus)
                if allocated_pairs:
                    total_daily_profit = sum(p["daily_profit"] for p in allocated_pairs)
                    message = f"Investment distribution results (${TOTAL_INVESTMENT:,.2f}):\n"
                    for pair in allocated_pairs:
                        data = pair["data"]
                        message += (
                            f"\nPair: {data.get('Collateral', 'N/A')} ({data.get('Rate Type', 'N/A')})\n"
                            f"Link: {data.get('Link')}\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST\n"
                            f"Current Lend APR: {data.get('Lend APR')} {'+' + str(bonus) + '%' if data.get('Link').split('/')[-1].lower() == '0xdbe88dbac39263c47629ebba02b3ef4cf0752a72' else ''}\n"
                            f"New Lend APR: {pair['new_lend_apr']:.2f}%\n"
                            f"Invested: ${pair['investment']:,.2f}\n"
                            f"Daily Profit: ${pair['daily_profit']:,.2f}\n"
                            f"New Utilization: {pair['new_utilization'] * 100:.2f}%\n"
                            f"Liquidity: {data.get('Available Liquidity')}\n"
                            f"Current Utilization: {data.get('Utilization Rate')}\n"
                            f"Borrow APR: {data.get('Borrow APR')}\n"
                            f"Reserve: {data.get('Reserve Size')}\n"
                        )
                    message += f"\nTotal daily profit: ${total_daily_profit:,.2f}"
                    send_to_telegram(message)
                else:
                    send_to_telegram("No suitable pairs found for investment. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")
            else:
                send_to_telegram("No pair data to process. (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")

            if skipped_pairs:
                send_to_telegram(f"Skipped pairs in cycle #{iteration_count}: {', '.join(skipped_pairs)} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")

        except Exception as e:
            logger.error(f"Error processing pairs: {type(e).__name__}: {e}")
            send_to_telegram(f"Error processing pairs: {type(e).__name__}: {e} (Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EEST)")

        finally:
            if driver:
                try:
                    driver.quit()
                except WebDriverException:
                    logger.info("Ignoring WebDriverException on WebDriver close")
                kill_chromedriver()
                logger.info(f"WebDriver closed, memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            elapsed_time = time.time() - iteration_start_time
            logger.info(f"Iteration completed in {elapsed_time:.2f} seconds")
            logger.info(f"Peak memory in iteration: {peak_memory:.2f} MB")
            logger.info(f"Waiting {CHECK_INTERVAL} seconds before next iteration")
            remaining_time = CHECK_INTERVAL
            while remaining_time > 0:
                sleep_time = min(30, remaining_time)
                time.sleep(sleep_time)
                remaining_time -= sleep_time
                peak_memory = max(peak_memory, psutil.Process().memory_info().rss / 1024 / 1024)
                logger.info(
                    f"Waiting, {remaining_time} seconds left, memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB, peak memory: {peak_memory:.2f} MB")


def main():
    """Main function to start pair processing."""
    logger.info("Starting Background Worker")
    send_to_telegram("Background Worker started (Время: 11:18 AM EEST, Friday, August 22, 2025)")
    process_pairs()


if __name__ == "__main__":
    main()