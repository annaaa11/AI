# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from dataclasses import dataclass
# from typing import Tuple
# from tqdm import tqdm
# import re
# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
#
# # Настройки Telegram
# BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"  # Замените на ваш токен бота
# CHAT_IDS = [6192278046, 306507209]  # Список ID пользователей
# CHECK_INTERVAL = 60*60  # Секунд между проверками
# CHROMEDRIVER_PATH = "C:/Users/User/.wdm/drivers/chromedriver/win64/138.0.7204.183/chromedriver-win32/chromedriver.exe"
#
# app = Flask(__name__)
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
#     protocol_fee: float = 0.1  # 10%
#     min_apr: float = 0.005  # 0.5%
#     max_apr: float = 100.0131  # 10,001.31%
#     min_utilization: float = 0.75  # 75%
#     max_utilization: float = 0.85  # 85%
#     rate_half_life_days: float = 0.5  # 0.5 days
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
#         seconds_per_year = 365.24 * 24 * 3600
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (seconds_per_year * 100)
#         term = (
#             old_borrow_rate_per_sec - self.params.zero_util_rate) * self.params.vertex_util / old_utilization if old_utilization != 0 else 0
#         vertex_interest = term + self.params.zero_util_rate
#         full_utilization_interest = ((
#             vertex_interest - self.params.zero_util_rate) / self.params.vertex_rate_percent) + self.params.zero_util_rate
#         return full_utilization_interest
#
#     def get_full_utilization_interest(self, delta_time: float, utilization: float,
#                                       full_utilization_interest: float) -> float:
#         if utilization < self.params.min_target_util:
#             delta_utilization = ((
#                 self.params.min_target_util - utilization) * self.params.rate_precision) / self.params.min_target_util
#             decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * (
#                 self.params.rate_half_life * 1e36)) / decay_growth
#         elif utilization > self.params.max_target_util:
#             delta_utilization = ((utilization - self.params.max_target_util) * self.params.rate_precision) / (
#                 self.params.util_precision - self.params.max_target_util)
#             decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * decay_growth) / (
#                 self.params.rate_half_life * 1e36)
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
#             new_full_utilization_interest - self.params.zero_util_rate) * self.params.vertex_rate_percent) + self.params.zero_util_rate)
#         if utilization < self.params.vertex_util:
#             new_rate_per_sec = (self.params.zero_util_rate + (
#                 utilization * (vertex_interest - self.params.zero_util_rate)) / self.params.vertex_util)
#         else:
#             new_rate_per_sec = (vertex_interest + (
#                 (utilization - self.params.vertex_util) * (new_full_utilization_interest - vertex_interest)) / (
#                     1.0 - self.params.vertex_util))
#         return new_rate_per_sec, new_full_utilization_interest
#
#
# class TimeWeightedVariableInterestRate:
#     def __init__(self, params: TimeWeightedInterestRateParams):
#         self.params = params
#         self.seconds_per_year = 365.24 * 24 * 3600
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100
#
#     def calculate_new_full_utilization_rate(self,
#                                             old_utilization: float,
#                                             old_lend_apr: float,
#                                             delta_time: float) -> float:
#         old_rate_per_sec = self.calculate_rate_per_sec(old_lend_apr)
#         old_full_util_rate = old_rate_per_sec / (
#             old_utilization * (1 - self.params.protocol_fee)) if old_utilization != 0 else 0
#
#         if old_utilization < self.params.min_utilization:
#             delta_utilization = ((self.params.min_utilization - old_utilization) *
#                                  self.params.rate_precision) / self.params.min_utilization
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * (self.rate_half_life_secs * 1e36)) / decay_growth
#         elif old_utilization > self.params.max_utilization:
#             delta_utilization = ((old_utilization - self.params.max_utilization) *
#                                  self.params.rate_precision) / (
#                 self.params.util_precision - self.params.max_utilization)
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * decay_growth) / (self.rate_half_life_secs * 1e36)
#         else:
#             new_full_util_rate = old_full_util_rate
#
#         min_rate_per_sec = self.calculate_rate_per_sec(self.params.min_apr)
#         max_rate_per_sec = self.calculate_rate_per_sec(self.params.max_apr)
#         new_full_util_rate = min(max(new_full_util_rate, min_rate_per_sec), max_rate_per_sec)
#
#         return new_full_util_rate
#
#     def get_new_lend_apr(self,
#                          delta_time: float,
#                          current_utilization: float,
#                          old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(
#             old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (
#             1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * 100
#         return lend_apr
#
#
# def get_pair_links(driver):
#     url = "https://facts.frax.finance/fraxlend/pairs"
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 30).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")))
#         elems = driver.find_elements(By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")
#         links = set()
#         for e in elems:
#             href = e.get_attribute("href")
#             if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
#                 links.add(href)
#         # print(f"[DEBUG] Найдено ссылок на пары: {len(links)}")
#         return list(links)
#     except Exception as e:
#         # print(f"[DEBUG] Ошибка при получении ссылок: {e}")
#         return []
#
#
# def fetch_metrics(driver, url):
#     # print(f"[DEBUG] Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#         labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type"]
#         data = {"Link": url}
#         for label in labels:
#             try:
#                 el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                 data[label] = el.text.strip()
#             except:
#                 data[label] = "N/A"
#         # print(f"[DEBUG] Данные для {url}: {data}")
#         return data
#     except Exception as e:
#         # print(f"[DEBUG] Ошибка при загрузке данных для {url}: {e}")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A"}
#
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if is_reserve_size and 'm' in amount_str.lower():
#             return value * 1_000_000  # Для Reserve Size с суффиксом 'M' умножаем на миллион
#         elif 'k' in amount_str.lower():
#             return value * 1_000  # Для других величин с суффиксом 'k' умножаем на тысячу
#         return value
#     except:
#         # print(f"[DEBUG] Ошибка парсинга суммы: {amount_str}")
#         return 0.0
#
#
# def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str)
#         utilization = float(utilization_str) / 100
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         # print(
#         #     f"[DEBUG] Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         # print(f"[DEBUG] Ошибка парсинга данных: {e}")
#         return None, None, None, None
#
#     # Фильтрация: Lend APR > 20%, Utilization Rate < 101%, Reserve Size != 0
#     if lend_apr <= 20 or utilization >= 1.01 or reserve_size == 0:
#         # print(
#         #     f"[DEBUG] Пара отфильтрована: Lend APR={lend_apr}, Utilization={utilization}, Reserve Size={reserve_size}, Rate Type={rate_type}")
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
#             # print(f"[DEBUG] Investment={investment}, new_utilization={new_utilization:.4f}")
#             if new_utilization < 0.76:  # Условие: new_utilization >= 76% для V2
#                 # print(f"[DEBUG] Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
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
#             # print(f"[DEBUG] Investment={investment}, new_utilization={new_utilization:.4f}")
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
#         # print(f"[DEBUG] Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return None, None, None, None
#
#     if max_profit == 0:
#         # print(f"[DEBUG] Не найдено допустимых вложений для пары, valid_investments={valid_investments}")
#         return None, None, None, None
#
#     return optimal_investment, max_profit, optimal_lend_apr, optimal_utilization
#
#
# def send_to_telegram(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             response = requests.post(url, data=payload)
#             if response.status_code != 200:
#                 print(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#         except Exception as e:
#             print(f"Ошибка отправки в Telegram для chat_id {chat_id}: {e}")
#
#
# def process_pairs():
#     options = Options()
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')
#     service = Service(CHROMEDRIVER_PATH)
#     driver = webdriver.Chrome(service=service, options=options)
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     processed_urls = set()  # Для отслеживания обработанных пар
#
#     while True:
#         try:
#             pair_links = get_pair_links(driver)
#             print(f"Найдено пар: {len(pair_links)}")
#
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if url in processed_urls:
#                     continue  # Пропускаем уже обработанные пары
#
#                 data = fetch_metrics(driver, url)
#                 optimal_investment, max_profit, optimal_lend_apr, optimal_utilization = calculate_optimal_investment(
#                     data, v1_model, v2_model)
#
#                 rate_type = data.get("Rate Type", "N/A")
#                 if (optimal_investment is not None) and data.get("Lend APR", "0") != "N/A" and data.get(
#                         "Utilization Rate", "0") != "N/A":
#                     timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#                     message = (
#                         f"📄 Пара: {url} ({rate_type})\n"
#                         f"Timestamp (UTC): {timestamp} +3 часа\n"
#                         f"Старая Lend APR: {data.get('Lend APR')}\n"
#                         f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
#                         f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
#                         f"Максимальный доход за 1 день: ${max_profit:,.2f}\n"
#                         f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
#                         f"Available Liquidity: {data.get('Available Liquidity')}\n"
#                         f"Utilization Rate: {data.get('Utilization Rate')}\n"
#                         f"Borrow APR: {data.get('Borrow APR')}\n"
#                         f"Reserve Size: {data.get('Reserve Size')}\n"
#                         f"Rate Type: {rate_type}"
#                     )
#                     send_to_telegram(message)
#                     processed_urls.add(url)
#
#         except Exception as e:
#             print(f"Ошибка обработки пар: {e}")
#
#         driver.quit()  # Закрываем драйвер после каждого цикла
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
#
# def run_background():
#     thread = threading.Thread(target=process_pairs)
#     thread.daemon = True
#     thread.start()
#
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dataclasses import dataclass
from typing import Tuple
from tqdm import tqdm
import re
import requests
import time
import threading
from datetime import datetime, timezone
# from flask import Flask
# import os
#
#
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from dataclasses import dataclass
# from typing import Tuple
# from tqdm import tqdm
# import re
# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# import os
# import shutil
#
# # Настройки Telegram
# BOT_TOKEN = "8263165108:AAFJIRRFpTt2_f6xtCsVIa5e2ksa23PBsCY"  # Получаем из переменной окружения
# CHAT_IDS = [6192278046, 306507209]  # Список ID пользователей
# CHECK_INTERVAL = 60  # Секунд между проверками
# CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"  # Путь для chromedriver в Render
#
# app = Flask(__name__)
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
#     protocol_fee: float = 0.1  # 10%
#     min_apr: float = 0.005  # 0.5%
#     max_apr: float = 100.0131  # 10,001.31%
#     min_utilization: float = 0.75  # 75%
#     max_utilization: float = 0.85  # 85%
#     rate_half_life_days: float = 0.5  # 0.5 days
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
#         seconds_per_year = 365.24 * 24 * 3600
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (seconds_per_year * 100)
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
#         self.seconds_per_year = 365.24 * 24 * 3600
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100
#
#     def calculate_new_full_utilization_rate(self,
#                                             old_utilization: float,
#                                             old_lend_apr: float,
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
#
#         return new_full_util_rate
#
#     def get_new_lend_apr(self,
#                          delta_time: float,
#                          current_utilization: float,
#                          old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(
#             old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (
#                 1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * 100
#         return lend_apr
#
#
# def get_pair_links(driver):
#     url = "https://facts.frax.finance/fraxlend/pairs"
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 30).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")))
#         elems = driver.find_elements(By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")
#         links = set()
#         for e in elems:
#             href = e.get_attribute("href")
#             if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
#                 links.add(href)
#         # print(f"[DEBUG] Найдено ссылок на пары: {len(links)}")
#         return list(links)
#     except Exception as e:
#         # print(f"[DEBUG] Ошибка при получении ссылок: {e}")
#         return []
#
#
# def fetch_metrics(driver, url):
#     # print(f"[DEBUG] Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#         labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type"]
#         data = {"Link": url}
#         for label in labels:
#             try:
#                 el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                 data[label] = el.text.strip()
#             except:
#                 data[label] = "N/A"
#         # print(f"[DEBUG] Данные для {url}: {data}")
#         return data
#     except Exception as e:
#         # print(f"[DEBUG] Ошибка при загрузке данных для {url}: {e}")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A"}
#
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if is_reserve_size and 'm' in amount_str.lower():
#             return value * 1_000_000  # Для Reserve Size с суффиксом 'M' умножаем на миллион
#         elif 'k' in amount_str.lower():
#             return value * 1_000  # Для других величин с суффиксом 'k' умножаем на тысячу
#         return value
#     except:
#         # print(f"[DEBUG] Ошибка парсинга суммы: {amount_str}")
#         return 0.0
#
#
# def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str)
#         utilization = float(utilization_str) / 100
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         # print(
#         #     f"[DEBUG] Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         # print(f"[DEBUG] Ошибка парсинга данных: {e}")
#         return None, None, None, None
#
#     # Фильтрация: Lend APR > 20%, Utilization Rate < 101%, Reserve Size != 0
#     if lend_apr <= 20 or utilization >= 1.01 or reserve_size == 0:
#         # print(
#         #     f"[DEBUG] Пара отфильтрована: Lend APR={lend_apr}, Utilization={utilization}, Reserve Size={reserve_size}, Rate Type={rate_type}")
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
#             # print(f"[DEBUG] Investment={investment}, new_utilization={new_utilization:.4f}")
#             if new_utilization < 0.76:  # Условие: new_utilization >= 76% для V2
#                 # print(f"[DEBUG] Пропущено (V2): new_utilization={new_utilization:.4f} < 0.76")
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
#             # print(f"[DEBUG] Investment={investment}, new_utilization={new_utilization:.4f}")
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
#         # print(f"[DEBUG] Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
#         return None, None, None, None
#
#     if max_profit == 0:
#         # print(f"[DEBUG] Не найдено допустимых вложений для пары, valid_investments={valid_investments}")
#         return None, None, None, None
#
#     return optimal_investment, max_profit, optimal_lend_apr, optimal_utilization
#
#
# def send_to_telegram(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             response = requests.post(url, data=payload)
#             if response.status_code != 200:
#                 print(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#         except Exception as e:
#             print(f"Ошибка отправки в Telegram для chat_id {chat_id}: {e}")
#
#
# def process_pairs():
#     options = Options()
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')
#     service = Service(CHROMEDRIVER_PATH)
#
#     try:
#         driver = webdriver.Chrome(service=service, options=options)
#     except Exception as e:
#         print(f"Ошибка инициализации WebDriver: {e}")
#         return
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     processed_urls = set()  # Для отслеживания обработанных пар
#
#     while True:
#         try:
#             pair_links = get_pair_links(driver)
#             print(f"Найдено пар: {len(pair_links)}")
#
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if url in processed_urls:
#                     continue  # Пропускаем уже обработанные пары
#
#                 data = fetch_metrics(driver, url)
#                 optimal_investment, max_profit, optimal_lend_apr, optimal_utilization = calculate_optimal_investment(
#                     data, v1_model, v2_model)
#
#                 rate_type = data.get("Rate Type", "N/A")
#                 if (optimal_investment is not None) and data.get("Lend APR", "0") != "N/A" and data.get(
#                         "Utilization Rate", "0") != "N/A":
#                     timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#                     message = (
#                         f"📄 Пара: {url} ({rate_type})\n"
#                         f"Timestamp (UTC): {timestamp} +3 часа\n"
#                         f"Старая Lend APR: {data.get('Lend APR')}\n"
#                         f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
#                         f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
#                         f"Максимальный доход за 1 день: ${max_profit:,.2f}\n"
#                         f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
#                         f"Available Liquidity: {data.get('Available Liquidity')}\n"
#                         f"Utilization Rate: {data.get('Utilization Rate')}\n"
#                         f"Borrow APR: {data.get('Borrow APR')}\n"
#                         f"Reserve Size: {data.get('Reserve Size')}\n"
#                         f"Rate Type: {rate_type}"
#                     )
#                     send_to_telegram(message)
#                     processed_urls.add(url)
#
#         except Exception as e:
#             print(f"Ошибка обработки пар: {e}")
#
#         driver.quit()  # Закрываем драйвер после каждого цикла
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
#
# def run_background():
#     thread = threading.Thread(target=process_pairs)
#     thread.daemon = True
#     thread.start()
#
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)

#############

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from dataclasses import dataclass
# from typing import Tuple
# from tqdm import tqdm
# import re
# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# import os
# import shutil
# import logging
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)
#
# # Настройки Telegram
# BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8"  # Получаем из переменной окружения
# CHAT_IDS = [6192278046, 306507209]  # Список ID пользователей
# CHECK_INTERVAL = 60  # Секунд между проверками
# CHROMEDRIVER_PATH = "/usr/bin/chromedriver"  # Динамический поиск chromedriver
#
# app = Flask(__name__)
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
#     protocol_fee: float = 0.1  # 10%
#     min_apr: float = 0.005  # 0.5%
#     max_apr: float = 100.0131  # 10,001.31%
#     min_utilization: float = 0.75  # 75%
#     max_utilization: float = 0.85  # 85%
#     rate_half_life_days: float = 0.5  # 0.5 days
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
#         seconds_per_year = 365.24 * 24 * 3600
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (seconds_per_year * 100)
#         term = (
#             old_borrow_rate_per_sec - self.params.zero_util_rate) * self.params.vertex_util / old_utilization if old_utilization != 0 else 0
#         vertex_interest = term + self.params.zero_util_rate
#         full_utilization_interest = ((
#             vertex_interest - self.params.zero_util_rate) / self.params.vertex_rate_percent) + self.params.zero_util_rate
#         return full_utilization_interest
#
#     def get_full_utilization_interest(self, delta_time: float, utilization: float,
#                                       full_utilization_interest: float) -> float:
#         if utilization < self.params.min_target_util:
#             delta_utilization = ((
#                 self.params.min_target_util - utilization) * self.params.rate_precision) / self.params.min_target_util
#             decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * (
#                 self.params.rate_half_life * 1e36)) / decay_growth
#         elif utilization > self.params.max_target_util:
#             delta_utilization = ((utilization - self.params.max_target_util) * self.params.rate_precision) / (
#                 self.params.util_precision - self.params.max_target_util)
#             decay_growth = (self.params.rate_half_life * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_utilization_interest = (full_utilization_interest * decay_growth) / (
#                 self.params.rate_half_life * 1e36)
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
#             new_full_utilization_interest - self.params.zero_util_rate) * self.params.vertex_rate_percent) + self.params.zero_util_rate)
#         if utilization < self.params.vertex_util:
#             new_rate_per_sec = (self.params.zero_util_rate + (
#                 utilization * (vertex_interest - self.params.zero_util_rate)) / self.params.vertex_util)
#         else:
#             new_rate_per_sec = (vertex_interest + (
#                 (utilization - self.params.vertex_util) * (new_full_utilization_interest - vertex_interest)) / (
#                     1.0 - self.params.vertex_util))
#         return new_rate_per_sec, new_full_utilization_interest
#
#
# class TimeWeightedVariableInterestRate:
#     def __init__(self, params: TimeWeightedInterestRateParams):
#         self.params = params
#         self.seconds_per_year = 365.24 * 24 * 3600
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100
#
#     def calculate_new_full_utilization_rate(self,
#                                             old_utilization: float,
#                                             old_lend_apr: float,
#                                             delta_time: float) -> float:
#         old_rate_per_sec = self.calculate_rate_per_sec(old_lend_apr)
#         old_full_util_rate = old_rate_per_sec / (
#             old_utilization * (1 - self.params.protocol_fee)) if old_utilization != 0 else 0
#
#         if old_utilization < self.params.min_utilization:
#             delta_utilization = ((self.params.min_utilization - old_utilization) *
#                                  self.params.rate_precision) / self.params.min_utilization
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * (self.rate_half_life_secs * 1e36)) / decay_growth
#         elif old_utilization > self.params.max_utilization:
#             delta_utilization = ((old_utilization - self.params.max_utilization) *
#                                  self.params.rate_precision) / (
#                 self.params.util_precision - self.params.max_utilization)
#             decay_growth = (self.rate_half_life_secs * 1e36) + (delta_utilization * delta_utilization * delta_time)
#             new_full_util_rate = (old_full_util_rate * decay_growth) / (self.rate_half_life_secs * 1e36)
#         else:
#             new_full_util_rate = old_full_util_rate
#
#         min_rate_per_sec = self.calculate_rate_per_sec(self.params.min_apr)
#         max_rate_per_sec = self.calculate_rate_per_sec(self.params.max_apr)
#         new_full_util_rate = min(max(new_full_util_rate, min_rate_per_sec), max_rate_per_sec)
#
#         return new_full_util_rate
#
#     def get_new_lend_apr(self,
#                          delta_time: float,
#                          current_utilization: float,
#                          old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(
#             old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (
#             1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * 100
#         return lend_apr
#
#
# def get_pair_links(driver):
#     url = "https://facts.frax.finance/fraxlend/pairs"
#     logger.info(f"Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 30).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")))
#         elems = driver.find_elements(By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")
#         links = set()
#         for e in elems:
#             href = e.get_attribute("href")
#             if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
#                 links.add(href)
#         logger.info(f"Найдено ссылок на пары: {len(links)}")
#         return list(links)
#     except Exception as e:
#         logger.error(f"Ошибка при получении ссылок: {e}")
#         return []
#
#
# def fetch_metrics(driver, url):
#     logger.info(f"Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#         labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type"]
#         data = {"Link": url}
#         for label in labels:
#             try:
#                 el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                 data[label] = el.text.strip()
#             except:
#                 data[label] = "N/A"
#         logger.info(f"Данные для {url}: {data}")
#         return data
#     except Exception as e:
#         logger.error(f"Ошибка при загрузке данных для {url}: {e}")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A"}
#
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if is_reserve_size and 'm' in amount_str.lower():
#             return value * 1_000_000  # Для Reserve Size с суффиксом 'M' умножаем на миллион
#         elif 'k' in amount_str.lower():
#             return value * 1_000  # Для других величин с суффиксом 'k' умножаем на тысячу
#         return value
#     except:
#         logger.error(f"Ошибка парсинга суммы: {amount_str}")
#         return 0.0
#
#
# def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str)
#         utilization = float(utilization_str) / 100
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных: {e}")
#         return None, None, None, None
#
#     # Фильтрация: Lend APR > 20%, Utilization Rate < 101%, Reserve Size != 0
#     if lend_apr <= 20 or utilization >= 1.01 or reserve_size == 0:
#         logger.info(
#             f"Пара отфильтрована: Lend APR={lend_apr}, Utilization={utilization}, Reserve Size={reserve_size}, Rate Type={rate_type}")
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
#             if new_utilization < 0.76:  # Условие: new_utilization >= 76% для V2
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
#
#
# def send_to_telegram(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             response = requests.post(url, data=payload)
#             if response.status_code != 200:
#                 logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#             else:
#                 logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
#         except Exception as e:
#             logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {e}")
#
#
# def process_pairs():
#     logger.info("Запуск функции process_pairs")
#     send_to_telegram("Тест: Сервер запущен, начинаем парсинг")
#     logger.info(f"Используемый путь chromedriver: {CHROMEDRIVER_PATH}")
#     options = Options()
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')
#     service = Service(CHROMEDRIVER_PATH)
#
#     for attempt in range(3):  # Попытки инициализации WebDriver
#         try:
#             driver = webdriver.Chrome(service=service, options=options)
#             logger.info(f"WebDriver успешно инициализирован с chromedriver по пути: {CHROMEDRIVER_PATH}")
#             break
#         except Exception as e:
#             logger.error(f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver: {e}")
#             if attempt == 2:
#                 logger.error("Не удалось инициализировать WebDriver после 3 попыток. Прекращаем выполнение.")
#                 send_to_telegram("Ошибка: Не удалось инициализировать WebDriver. Проверьте chromedriver.")
#                 return
#             time.sleep(2)  # Пауза перед следующей попыткой
#     else:
#         return  # Выход, если все попытки неудачны
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     processed_urls = set()  # Для отслеживания обработанных пар
#
#     while True:
#         try:
#             pair_links = get_pair_links(driver)
#             logger.info(f"Найдено пар: {len(pair_links)}")
#
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if url in processed_urls:
#                     continue  # Пропускаем уже обработанные пары
#
#                 data = fetch_metrics(driver, url)
#                 optimal_investment, max_profit, optimal_lend_apr, optimal_utilization = calculate_optimal_investment(
#                     data, v1_model, v2_model)
#
#                 rate_type = data.get("Rate Type", "N/A")
#                 if (optimal_investment is not None) and data.get("Lend APR", "0") != "N/A" and data.get(
#                         "Utilization Rate", "0") != "N/A":
#                     timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#                     message = (
#                         f"📄 Пара: {url} ({rate_type})\n"
#                         f"Timestamp (UTC): {timestamp} +3 часа\n"
#                         f"Старая Lend APR: {data.get('Lend APR')}\n"
#                         f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
#                         f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
#                         f"Максимальный доход за 1 день: ${max_profit:,.2f}\n"
#                         f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
#                         f"Available Liquidity: {data.get('Available Liquidity')}\n"
#                         f"Utilization Rate: {data.get('Utilization Rate')}\n"
#                         f"Borrow APR: {data.get('Borrow APR')}\n"
#                         f"Reserve Size: {data.get('Reserve Size')}\n"
#                         f"Rate Type: {rate_type}"
#                     )
#                     send_to_telegram(message)
#                     processed_urls.add(url)
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки пар: {e}")
#             send_to_telegram(f"Ошибка при обработке пар: {e}")
#
#         driver.quit()  # Закрываем драйвер после каждого цикла
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
#
# def run_background():
#     logger.info("Запуск фонового потока для парсинга")
#     thread = threading.Thread(target=process_pairs)
#     thread.daemon = True
#     thread.start()
#
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)  # Для локального тестирования

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# from dataclasses import dataclass
# from typing import Tuple
# from tqdm import tqdm
# import re
# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# import os
# import logging
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)
#
# # Настройки Telegram
# BOT_TOKEN = "8218685044:AAESCtKJJEi0guAAH4iOtt_haD7LL_Ukow8" # Замените на ваш токен или задайте в Render
# CHAT_IDS = [6192278046, 306507209]  # Список ID пользователей
# CHECK_INTERVAL = 60  # Секунд между проверками
#
# app = Flask(__name__)
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
#     protocol_fee: float = 0.1  # 10%
#     min_apr: float = 0.005  # 0.5%
#     max_apr: float = 100.0131  # 10,001.31%
#     min_utilization: float = 0.75  # 75%
#     max_utilization: float = 0.85  # 85%
#     rate_half_life_days: float = 0.5  # 0.5 days
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
#         seconds_per_year = 365.24 * 24 * 3600
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (seconds_per_year * 100)
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
#         self.seconds_per_year = 365.24 * 24 * 3600
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100
#
#     def calculate_new_full_utilization_rate(self,
#                                             old_utilization: float,
#                                             old_lend_apr: float,
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
#
#         return new_full_util_rate
#
#     def get_new_lend_apr(self,
#                          delta_time: float,
#                          current_utilization: float,
#                          old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(
#             old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (
#                 1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * 100
#         return lend_apr
#
#
# def get_pair_links(driver):
#     url = "https://facts.frax.finance/fraxlend/pairs"
#     logger.info(f"Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 30).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")))
#         elems = driver.find_elements(By.CSS_SELECTOR, "a[href^='/fraxlend/pairs/']")
#         links = set()
#         for e in elems:
#             href = e.get_attribute("href")
#             if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
#                 links.add(href)
#         logger.info(f"Найдено ссылок на пары: {len(links)}")
#         return list(links)
#     except Exception as e:
#         logger.error(f"Ошибка при получении ссылок: {e}")
#         return []
#
#
# def fetch_metrics(driver, url):
#     logger.info(f"Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#         labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type"]
#         data = {"Link": url}
#         for label in labels:
#             try:
#                 el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                 data[label] = el.text.strip()
#             except:
#                 data[label] = "N/A"
#         logger.info(f"Данные для {url}: {data}")
#         return data
#     except Exception as e:
#         logger.error(f"Ошибка при загрузке данных для {url}: {e}")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A"}
#
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if is_reserve_size and 'm' in amount_str.lower():
#             return value * 1_000_000  # Для Reserve Size с суффиксом 'M' умножаем на миллион
#         elif 'k' in amount_str.lower():
#             return value * 1_000  # Для других величин с суффиксом 'k' умножаем на тысячу
#         return value
#     except:
#         logger.error(f"Ошибка парсинга суммы: {amount_str}")
#         return 0.0
#
#
# def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str)
#         utilization = float(utilization_str) / 100
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных: {e}")
#         return None, None, None, None
#
#     # Фильтрация: Lend APR > 20%, Utilization Rate < 101%, Reserve Size != 0
#     if lend_apr <= 20 or utilization >= 1.01 or reserve_size == 0:
#         logger.info(
#             f"Пара отфильтрована: Lend APR={lend_apr}, Utilization={utilization}, Reserve Size={reserve_size}, Rate Type={rate_type}")
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
#             if new_utilization < 0.76:  # Условие: new_utilization >= 76% для V2
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
#
#
# def send_to_telegram(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             response = requests.post(url, data=payload)
#             if response.status_code != 200:
#                 logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#             else:
#                 logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
#         except Exception as e:
#             logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {e}")
#
#
# def process_pairs():
#     logger.info("Запуск функции process_pairs")
#     send_to_telegram("Тест: Сервер запущен, начинаем парсинг")
#
#     options = Options()
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')
#
#     for attempt in range(3):  # Попытки инициализации WebDriver
#         try:
#             driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#             logger.info("WebDriver успешно инициализирован")
#             break
#         except Exception as e:
#             logger.error(f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver: {e}")
#             if attempt == 2:
#                 logger.error("Не удалось инициализировать WebDriver после 3 попыток. Прекращаем выполнение.")
#                 send_to_telegram("Ошибка: Не удалось инициализировать WebDriver. Проверьте chromedriver.")
#                 return
#             time.sleep(2)  # Пауза перед следующей попыткой
#     else:
#         return  # Выход, если все попытки неудачны
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     processed_urls = set()  # Для отслеживания обработанных пар
#
#     while True:
#         try:
#             pair_links = get_pair_links(driver)
#             logger.info(f"Найдено пар: {len(pair_links)}")
#
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if url in processed_urls:
#                     continue  # Пропускаем уже обработанные пары
#
#                 data = fetch_metrics(driver, url)
#                 optimal_investment, max_profit, optimal_lend_apr, optimal_utilization = calculate_optimal_investment(
#                     data, v1_model, v2_model)
#
#                 rate_type = data.get("Rate Type", "N/A")
#                 if (optimal_investment is not None) and data.get("Lend APR", "0") != "N/A" and data.get(
#                         "Utilization Rate", "0") != "N/A":
#                     timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#                     message = (
#                         f"📄 Пара: {url} ({rate_type})\n"
#                         f"Timestamp (UTC): {timestamp} +3 часа\n"
#                         f"Старая Lend APR: {data.get('Lend APR')}\n"
#                         f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
#                         f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
#                         f"Максимальный доход за 1 день: ${max_profit:,.2f}\n"
#                         f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
#                         f"Available Liquidity: {data.get('Available Liquidity')}\n"
#                         f"Utilization Rate: {data.get('Utilization Rate')}\n"
#                         f"Borrow APR: {data.get('Borrow APR')}\n"
#                         f"Reserve Size: {data.get('Reserve Size')}\n"
#                         f"Rate Type: {rate_type}"
#                     )
#                     send_to_telegram(message)
#                     processed_urls.add(url)
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки пар: {e}")
#             send_to_telegram(f"Ошибка при обработке пар: {e}")
#
#         driver.quit()  # Закрываем драйвер после каждого цикла
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
#
# def run_background():
#     logger.info("Запуск фонового потока для парсинга")
#     thread = threading.Thread(target=process_pairs)
#     thread.daemon = True
#     thread.start()
#
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)  # Для локального тестирования

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from dataclasses import dataclass
# from typing import Tuple
# from tqdm import tqdm
# import re
# import requests
# import time
# from datetime import datetime, timezone
# import os
# import logging
# import psutil
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger(__name__)
#
# # Настройки Telegram
# BOT_TOKEN = os.getenv("BOT_TOKEN", "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk")
# CHAT_IDS = [6192278046, 306507209]
# CHECK_INTERVAL = 300  # Интервал проверки в секундах
# MAX_PAIRS = 1  # Ограничение количества пар для теста
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
#         seconds_per_year = 365.24 * 24 * 3600
#         old_borrow_apr = old_lend_apr / old_utilization if old_utilization != 0 else 0
#         old_borrow_rate_per_sec = old_borrow_apr / (seconds_per_year * 100)
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
#         self.seconds_per_year = 365.24 * 24 * 3600
#         self.rate_half_life_secs = params.rate_half_life_days * 24 * 3600
#
#     def calculate_rate_per_sec(self, apr: float) -> float:
#         return apr / 100
#
#     def calculate_new_full_utilization_rate(self,
#                                             old_utilization: float,
#                                             old_lend_apr: float,
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
#
#         return new_full_util_rate
#
#     def get_new_lend_apr(self,
#                          delta_time: float,
#                          current_utilization: float,
#                          old_lend_apr: float,
#                          old_utilization: float) -> float:
#         full_utilization_rate = self.calculate_new_full_utilization_rate(
#             old_utilization, old_lend_apr, delta_time)
#         lend_rate_per_sec = full_utilization_rate * current_utilization * (
#                 1 - self.params.protocol_fee) if current_utilization != 0 else 0
#         lend_apr = lend_rate_per_sec * 100
#         return lend_apr
#
#
# def get_pair_links(driver, max_retries=3):
#     url = "https://facts.frax.finance/fraxlend/pairs"
#     logger.info(f"Попытка загрузки страницы: {url}")
#
#     for attempt in range(max_retries):
#         try:
#             # Проверка сетевой доступности
#             response = requests.get(url, timeout=30)
#             logger.info(f"Статус HTTP-запроса к {url}: {response.status_code}")
#
#             # Загрузка страницы в WebDriver
#             driver.get(url)
#
#             # Ожидание появления ссылок
#             WebDriverWait(driver, 20).until(
#                 EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']"))
#             )
#             elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']")
#             links = set()
#             for e in elems:
#                 href = e.get_attribute("href")
#                 if href and '/fraxlend/pairs/' in href:
#                     # Приведение относительных ссылок к абсолютным
#                     if not href.startswith('http'):
#                         href = f"https://facts.frax.finance{href}"
#                     if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
#                         links.add(href)
#             logger.info(f"Найдено ссылок на пары: {len(links)}")
#             return list(links)[:MAX_PAIRS]  # Ограничение количества пар
#
#         except EC.TimeoutException as e:
#             logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при ожидании элементов на {url}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error("Не удалось загрузить ссылки после всех попыток.")
#                 send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
#                 return []
#             time.sleep(5)
#
#         except Exception as e:
#             logger.error(f"Попытка {attempt + 1}/{max_retries}: Ошибка при получении ссылок: {type(e).__name__}: {e}")
#             if attempt == max_retries - 1:
#                 logger.error("Не удалось загрузить ссылки после всех попыток.")
#                 send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
#                 return []
#             time.sleep(5)
#
#     return []
#
#
# def fetch_metrics(driver, url):
#     logger.info(f"Загрузка страницы: {url}")
#     driver.get(url)
#     try:
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
#         labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type"]
#         data = {"Link": url}
#         for label in labels:
#             try:
#                 el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
#                 data[label] = el.text.strip()
#             except:
#                 data[label] = "N/A"
#         logger.info(f"Данные для {url}: {data}")
#         return data
#     except Exception as e:
#         logger.error(f"Ошибка при загрузке данных для {url}: {e}")
#         return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
#                 "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A"}
#
#
# def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
#     try:
#         cleaned = re.sub(r'[^\d.]', '', amount_str)
#         value = float(cleaned)
#         if is_reserve_size and 'm' in amount_str.lower():
#             return value * 1_000_000
#         elif 'k' in amount_str.lower():
#             return value * 1_000
#         return value
#     except:
#         logger.error(f"Ошибка парсинга суммы: {amount_str}")
#         return 0.0
#
#
# def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
#     try:
#         lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
#         utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
#         lend_apr = float(lend_apr_str)
#         utilization = float(utilization_str) / 100
#         available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
#         reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
#         rate_type = data.get("Rate Type", "N/A")
#         logger.info(
#             f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных: {e}")
#         return None, None, None, None
#
#     if lend_apr <= 20 or utilization >= 1.01 or reserve_size == 0:
#         logger.info(
#             f"Пара отфильтрована: Lend APR={lend_apr}, Utilization={utilization}, Reserve Size={reserve_size}, Rate Type={rate_type}")
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
#
#
# def send_to_telegram(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             response = requests.post(url, data=payload)
#             if response.status_code != 200:
#                 logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
#             else:
#                 logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
#         except Exception as e:
#             logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {type(e).__name__}: {e}")
#
#
# def process_pairs():
#     logger.info(
#         f"Запуск функции process_pairs, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#     send_to_telegram("Тест: Сервер запущен, начинаем парсинг")
#
#     options = Options()
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--window-size=1920,1080')
#     options.binary_location = '/usr/bin/chromium'
#
#     chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
#     logger.info(f"Используемый путь к chromedriver: {chromedriver_path}")
#
#     v1_params = TimeWeightedInterestRateParams()
#     v2_params = InterestRateParams()
#     v1_model = TimeWeightedVariableInterestRate(v1_params)
#     v2_model = VariableInterestRate(v2_params)
#
#     processed_urls = set()
#
#     while True:
#         driver = None
#         try:
#             logger.info(
#                 f"Начало парсинга пар, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#
#             # Инициализация WebDriver
#             for attempt in range(3):
#                 try:
#                     logger.info(f"Попытка {attempt + 1}/3: Инициализация WebDriver")
#                     driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
#                     logger.info("WebDriver успешно инициализирован")
#                     break
#                 except Exception as e:
#                     logger.error(f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver: {type(e).__name__}: {e}")
#                     if attempt == 2:
#                         logger.error("Не удалось инициализировать WebDriver после 3 попыток.")
#                         send_to_telegram("Ошибка: Не удалось инициализировать WebDriver. Проверьте конфигурацию.")
#                         return
#                     time.sleep(2)
#
#             pair_links = get_pair_links(driver)
#             logger.info(f"Найдено пар: {len(pair_links)}")
#
#             if not pair_links:
#                 logger.warning("Список пар пуст. Пропускаем итерацию.")
#                 send_to_telegram("Предупреждение: Список пар пуст. Проверьте сайт или селектор.")
#                 time.sleep(CHECK_INTERVAL)
#                 continue
#
#             for url in tqdm(pair_links, desc="Обработка пар"):
#                 if url in processed_urls:
#                     logger.debug(f"Пропущена пара (уже обработана): {url}")
#                     continue
#
#                 data = fetch_metrics(driver, url)
#                 optimal_investment, max_profit, optimal_lend_apr, optimal_utilization = calculate_optimal_investment(
#                     data, v1_model, v2_model)
#
#                 rate_type = data.get("Rate Type", "N/A")
#                 if (optimal_investment is not None) and data.get("Lend APR", "0") != "N/A" and data.get(
#                         "Utilization Rate", "0") != "N/A":
#                     timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#                     message = (
#                         f"📄 Пара: {url} ({rate_type})\n"
#                         f"Timestamp (UTC): {timestamp} +3 часа\n"
#                         f"Старая Lend APR: {data.get('Lend APR')}\n"
#                         f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
#                         f"Оптимальная сумма для вложения: ${optimal_investment:,.2f}\n"
#                         f"Максимальный доход за 1 день: ${max_profit:,.2f}\n"
#                         f"Новая ставка утилизации: {optimal_utilization * 100:.2f}%\n"
#                         f"Available Liquidity: {data.get('Available Liquidity')}\n"
#                         f"Utilization Rate: {data.get('Utilization Rate')}\n"
#                         f"Borrow APR: {data.get('Borrow APR')}\n"
#                         f"Reserve Size: {data.get('Reserve Size')}\n"
#                         f"Rate Type: {rate_type}"
#                     )
#                     send_to_telegram(message)
#                     processed_urls.add(url)
#
#                 # Освобождение памяти после обработки каждой пары
#                 logger.info(f"Память после обработки {url}: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#                 driver.execute_script("window.localStorage.clear();")
#                 driver.execute_script("window.sessionStorage.clear();")
#                 time.sleep(1)  # Короткая пауза для снижения нагрузки
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки пар: {type(e).__name__}: {e}")
#             send_to_telegram(f"Ошибка при обработке пар: {type(e).__name__}: {e}")
#
#         finally:
#             # Закрытие WebDriver
#             if driver:
#                 driver.quit()
#                 logger.info(f"WebDriver закрыт, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
#             time.sleep(CHECK_INTERVAL)
#
#
# def main():
#     logger.info("Запуск Background Worker")
#     send_to_telegram("Тест: Background Worker запущен")
#     process_pairs()
#
#
# if __name__ == "__main__":
#     main()

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dataclasses import dataclass
from typing import Tuple
from tqdm import tqdm
import re
import requests
import time
from datetime import datetime, timezone
import os
import logging
import psutil

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Настройки Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN", "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk")
CHAT_IDS = [6192278046, 306507209]
CHECK_INTERVAL = 60*20  # Интервал проверки в секундах
MAX_PAIRS = 100  # Ограничение количества пар для теста


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
            # Проверка сетевой доступности
            response = requests.get(url, timeout=30)
            logger.info(f"Статус HTTP-запроса к {url}: {response.status_code}")

            # Загрузка страницы в WebDriver
            driver.get(url)

            # Ожидание появления ссылок
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']"))
            )
            elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/fraxlend/pairs/']")
            links = set()
            for e in elems:
                href = e.get_attribute("href")
                if href and '/fraxlend/pairs/' in href:
                    # Приведение относительных ссылок к абсолютным
                    if not href.startswith('http'):
                        href = f"https://facts.frax.finance{href}"
                    if href.startswith("https://facts.frax.finance/fraxlend/pairs/"):
                        links.add(href)
            logger.info(f"Найдено ссылок на пары: {len(links)}")
            return list(links)[:MAX_PAIRS]  # Ограничение количества пар

        except EC.TimeoutException as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Тайм-аут при ожидании элементов на {url}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ссылки после всех попыток.")
                send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
                return []
            time.sleep(5)

        except Exception as e:
            logger.error(f"Попытка {attempt + 1}/{max_retries}: Ошибка при получении ссылок: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                logger.error("Не удалось загрузить ссылки после всех попыток.")
                send_to_telegram("Ошибка: Не удалось загрузить ссылки на пары.")
                return []
            time.sleep(5)

    return []


def fetch_metrics(driver, url):
    logger.info(f"Загрузка страницы: {url}")
    driver.get(url)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Available Liquidity')]")))
        labels = ["Available Liquidity", "Utilization Rate", "Lend APR", "Borrow APR", "Reserve Size", "Rate Type"]
        data = {"Link": url}
        for label in labels:
            try:
                el = driver.find_element(By.XPATH, f"//div[contains(text(), '{label}')]/following-sibling::div")
                data[label] = el.text.strip()
            except:
                data[label] = "N/A"
        logger.info(f"Данные для {url}: {data}")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных для {url}: {e}")
        return {"Link": url, "Available Liquidity": "N/A", "Utilization Rate": "N/A", "Lend APR": "N/A",
                "Borrow APR": "N/A", "Reserve Size": "N/A", "Rate Type": "N/A"}


def parse_dollar_amount(amount_str: str, is_reserve_size: bool = False) -> float:
    try:
        cleaned = re.sub(r'[^\d.]', '', amount_str)
        value = float(cleaned)
        if is_reserve_size and 'm' in amount_str.lower():
            return value * 1_000_000
        elif 'k' in amount_str.lower():
            return value * 1_000
        return value
    except:
        logger.error(f"Ошибка парсинга суммы: {amount_str}")
        return 0.0


def calculate_optimal_investment(data, v1_model, v2_model, delta_time=86400.0):
    try:
        lend_apr_str = data.get("Lend APR", "0").replace("%", "").strip()
        utilization_str = data.get("Utilization Rate", "0").replace("%", "").strip()
        lend_apr = float(lend_apr_str)
        utilization = float(utilization_str) / 100
        available_liquidity = parse_dollar_amount(data.get("Available Liquidity", "0"))
        reserve_size = parse_dollar_amount(data.get("Reserve Size", "0"), is_reserve_size=True)
        rate_type = data.get("Rate Type", "N/A")
        logger.info(
            f"Распарсенные данные: Lend APR={lend_apr}, Utilization={utilization}, Available Liquidity={available_liquidity}, Reserve Size={reserve_size}, Rate Type={rate_type}")
    except Exception as e:
        logger.error(f"Ошибка парсинга данных: {e}")
        return None, None, None, None

    if lend_apr <= 20 or utilization >= 1.01 or reserve_size == 0:
        logger.info(
            f"Пара отфильтрована: Lend APR={lend_apr}, Utilization={utilization}, Reserve Size={reserve_size}, Rate Type={rate_type}")
        return None, None, None, None

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
            daily_profit = (investment * new_lend_apr / 100) / 365.24

            if daily_profit > max_profit:
                max_profit = daily_profit
                optimal_investment = investment
                optimal_lend_apr = new_lend_apr
                optimal_utilization = new_utilization

    else:
        logger.info(f"Пара отфильтрована: неподдерживаемый Rate Type={rate_type}")
        return None, None, None, None

    if max_profit == 0:
        logger.info(f"Не найдено допустимых вложений для пары, valid_investments={valid_investments}")
        return None, None, None, None

    return optimal_investment, max_profit, optimal_lend_apr, optimal_utilization


def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": message
        }
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {response.text}")
            else:
                logger.info(f"Сообщение успешно отправлено в Telegram для chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {type(e).__name__}: {e}")


def process_pairs():
    logger.info(
        f"Запуск функции process_pairs, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    send_to_telegram("Тест: Сервер запущен, начинаем парсинг")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.binary_location = '/usr/bin/chromium'

    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    logger.info(f"Используемый путь к chromedriver: {chromedriver_path}")

    v1_params = TimeWeightedInterestRateParams()
    v2_params = InterestRateParams()
    v1_model = TimeWeightedVariableInterestRate(v1_params)
    v2_model = VariableInterestRate(v2_params)

    processed_urls = set()

    while True:
        driver = None
        try:
            logger.info(
                f"Начало парсинга пар, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            # Инициализация WebDriver
            for attempt in range(3):
                try:
                    logger.info(f"Попытка {attempt + 1}/3: Инициализация WebDriver")
                    driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
                    logger.info("WebDriver успешно инициализирован")
                    break
                except Exception as e:
                    logger.error(f"Попытка {attempt + 1}/3: Ошибка инициализации WebDriver: {type(e).__name__}: {e}")
                    if attempt == 2:
                        logger.error("Не удалось инициализировать WebDriver после 3 попыток.")
                        send_to_telegram("Ошибка: Не удалось инициализировать WebDriver. Проверьте конфигурацию.")
                        return
                    time.sleep(2)

            pair_links = get_pair_links(driver)
            logger.info(f"Найдено пар: {len(pair_links)}")

            if not pair_links:
                logger.warning("Список пар пуст. Пропускаем итерацию.")
                send_to_telegram("Предупреждение: Список пар пуст. Проверьте сайт или селектор.")
                time.sleep(CHECK_INTERVAL)
                continue

            for url in tqdm(pair_links, desc="Обработка пар"):
                if url in processed_urls:
                    logger.debug(f"Пропущена пара (уже обработана): {url}")
                    continue

                data = fetch_metrics(driver, url)
                optimal_investment, max_profit, optimal_lend_apr, optimal_utilization = calculate_optimal_investment(
                    data, v1_model, v2_model)

                rate_type = data.get("Rate Type", "N/A")
                if (optimal_investment is not None) and data.get("Lend APR", "0") != "N/A" and data.get(
                        "Utilization Rate", "0") != "N/A":
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    message = (
                        f"📄 Пара: {url} ({rate_type})\n"
                        f"Timestamp (UTC): {timestamp} +3 часа\n"
                        f"Старая Lend APR: {data.get('Lend APR')}\n"
                        f"Новая оптимальная Lend APR: {optimal_lend_apr:.2f}%\n"
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

                # Освобождение памяти после обработки каждой пары
                logger.info(f"Память после обработки {url}: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
                driver.execute_script("window.localStorage.clear();")
                driver.execute_script("window.sessionStorage.clear();")
                time.sleep(1)  # Короткая пауза для снижения нагрузки

        except Exception as e:
            logger.error(f"Ошибка обработки пар: {type(e).__name__}: {e}")
            send_to_telegram(f"Ошибка при обработке пар: {type(e).__name__}: {e}")

        finally:
            # Закрытие WebDriver
            if driver:
                driver.quit()
                logger.info(f"WebDriver закрыт, память: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

            # Периодический вывод в логи во время ожидания
            logger.info(f"Ожидание {CHECK_INTERVAL} секунд перед следующей итерацией")
            for i in range(CHECK_INTERVAL // 60):
                time.sleep(60)
                logger.info(
                    f"Ожидание, осталось {CHECK_INTERVAL - (i + 1) * 60} секунд, использование памяти: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")


def main():
    logger.info("Запуск Background Worker")
    send_to_telegram("Тест: Background Worker запущен")
    process_pairs()


if __name__ == "__main__":
    main()