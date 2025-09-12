#
#
# import requests
# import time
# from datetime import datetime, timedelta
#
# # Настройки
# API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
# ADDRESS = "0x4579a27af00a62c0eb156349f31b345c08386419".lower()
# BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
# CHAT_ID = "6192278046"  # ID пользователя или канала
# CHECK_INTERVAL = 15  # секунд между проверками
# WINDOW_MINUTES = 5  # интервал поиска в минутах
#
# # Хранилище уже отправленных транзакций
# sent_tx_hashes = set()
#
# def get_token_transactions(address, count=100):
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={address}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY}"
#     )
#     try:
#         resp = requests.get(url, timeout=10).json()
#         if resp["status"] == "1":
#             return resp["result"][:count]
#     except Exception as e:
#         print(f"Ошибка при получении транзакций: {e}")
#     return []
#
#
# def send_telegram_message(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
#     try:
#         requests.post(url, data=payload, timeout=10)
#     except Exception as e:
#         print(f"Ошибка отправки в Telegram: {e}")
#
# def format_tx_message(tx):
#     timestamp_utc = datetime.utcfromtimestamp(int(tx['timeStamp']))
#     timestamp_msk = timestamp_utc + timedelta(hours=3)
#     value = int(tx['value']) / (10 ** int(tx['tokenDecimal']))
#     return (
#         f"<b>Новая транзакция:</b>\n"
#         f"<b>Tx Hash:</b> <code>{tx['hash']}</code>\n"
#         f"<b>Время:</b> {timestamp_msk.strftime('%Y-%m-%d %H:%M:%S')} (UTC+3)\n"
#         f"<b>Token:</b> {tx['tokenName']} ({tx['tokenSymbol']})\n"
#         f"<b>Value:</b> {value:.8f}"
#     )
#
# def main_loop():
#     print("⏳ Запуск мониторинга...")
#     while True:
#         now = datetime.utcnow()
#         recent_cutoff = now - timedelta(minutes=WINDOW_MINUTES)
#
#         transactions = get_token_transactions(ADDRESS, count=100)
#         for tx in transactions:
#             if tx['from'].lower() != ADDRESS:
#                 continue
#             tx_hash = tx['hash']
#             if tx_hash in sent_tx_hashes:
#                 continue
#
#             tx_time = datetime.utcfromtimestamp(int(tx['timeStamp']))
#             if tx_time < recent_cutoff:
#                 continue  # старее 60 минут
#
#             # Отправляем сообщение
#             message = format_tx_message(tx)
#             send_telegram_message(message)
#             sent_tx_hashes.add(tx_hash)
#             print(f"✅ Отправлено: {tx_hash}")
#
#         time.sleep(CHECK_INTERVAL)
#
# if __name__ == "__main__":
#     main_loop()


# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# from web3 import Web3
# import json
#
# API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
# ADDRESS = "0x4579a27af00a62c0eb156349f31b345c08386419".lower()
# BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
# CHAT_ID = "6192278046"  # ID пользователя или канала
# CHECK_INTERVAL = 15  # секунд между проверками
#
# app = Flask(__name__)
#
# def get_transactions():
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={ADDRESS}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY}"
#     )
#     resp = requests.get(url).json()
#     if resp["status"] == "1":
#         return resp["result"]
#     return []
#
#
# def send_to_telegram(message):
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     payload = {
#         "chat_id": CHAT_ID,
#         "text": message
#     }
#     try:
#         requests.post(url, data=payload)
#     except Exception as e:
#         print(f"Ошибка отправки в Telegram: {e}")
#
# def process_transactions():
#     seen_hashes = set()
#     while True:
#         try:
#             txs = get_transactions()
#             now = datetime.now(timezone.utc)
#             for tx in txs[:20]:
#                 # Фильтр по полю From
#                 if tx["from"].lower() != ADDRESS.lower():
#                     continue
#
#                 tx_hash = tx["hash"]
#                 if tx_hash in seen_hashes:
#                     continue
#
#                 tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
#                 # Фильтр по времени - только за последний час
#                 if (now - tx_time).total_seconds() > 300:
#                     continue
#
#                 # Формируем сообщение
#                 token_name = tx.get("tokenName", "Unknown token")
#                 token_decimal = int(tx.get("tokenDecimal", 18))
#                 value = int(tx["value"]) / (10 ** token_decimal)
#
#                 message = (
#                     f"Tx Hash:         {tx_hash}\n\n"
#                     f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
#                     f"Token Name:      {token_name}\n"
#                     f"Value:           {value}"
#                 )
#                 print(f"Отправляю в Telegram: {tx_hash}")
#                 send_to_telegram(message)
#                 seen_hashes.add(tx_hash)
#
#         except Exception as e:
#             print(f"Ошибка обработки транзакций: {e}")
#
#         time.sleep(CHECK_INTERVAL)
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
# def run_background():
#     thread = threading.Thread(target=process_transactions)
#     thread.daemon = True
#     thread.start()
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)


# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# from web3 import Web3
# import json
#
# API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
# ADDRESS = "0x4579a27af00a62c0eb156349f31b345c08386419".lower()
# BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
# CHAT_ID = "6192278046"  # ID пользователя или канала
# CHECK_INTERVAL = 15  # секунд между проверками
#
# app = Flask(__name__)
#
# def get_transactions():
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={ADDRESS}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY}"
#     )
#     resp = requests.get(url).json()
#     if resp["status"] == "1":
#         return resp["result"]
#     return []
#
# CHAT_IDS = [6192278046, 306507209]  # список id пользователей
#
# def send_to_telegram(message):
#     url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             requests.post(url_base, data=payload)
#         except Exception as e:
#             print(f"Ошибка при отправке в чат {chat_id}: {e}")
#
# # def send_to_telegram(message):
# #     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
# #     payload = {
# #         "chat_id": CHAT_ID,
# #         "text": message
# #     }
# #     try:
# #         requests.post(url, data=payload)
# #     except Exception as e:
# #         print(f"Ошибка отправки в Telegram: {e}")
#
# def process_transactions():
#     seen_hashes = set()
#     while True:
#         try:
#             txs = get_transactions()
#             now = datetime.now(timezone.utc)
#             for tx in txs[:20]:
#                 # Фильтр по полю From
#                 if tx["from"].lower() != ADDRESS.lower():
#                     continue
#
#                 tx_hash = tx["hash"]
#                 if tx_hash in seen_hashes:
#                     continue
#
#                 tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
#                 # Фильтр по времени - только за последний час
#                 if (now - tx_time).total_seconds() > 2000:
#                     continue
#
#                 # Формируем сообщение
#                 token_name = tx.get("tokenName", "Unknown token")
#                 token_decimal = int(tx.get("tokenDecimal", 18))
#                 value = int(tx["value"]) / (10 ** token_decimal)
#
#                 message = (
#                     f"Tx Hash:         {tx_hash}\n\n"
#                     f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
#                     f"Token Name:      {token_name}\n"
#                     f"Value:           {value}"
#                 )
#                 print(f"Отправляю в Telegram: {tx_hash}")
#                 send_to_telegram(message)
#                 seen_hashes.add(tx_hash)
#
#         except Exception as e:
#             print(f"Ошибка обработки транзакций: {e}")
#
#         time.sleep(CHECK_INTERVAL)
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
# def run_background():
#     thread = threading.Thread(target=process_transactions)
#     thread.daemon = True
#     thread.start()
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)

# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# from web3 import Web3
# import json
#
# API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
# ADDRESSES = [
#     "0x38c503a438185cde29b5cf4dc1442fd6f074f1cc",
#     "0x285866acb0d60105b4ed350a463361c2d9afa0e2",
#     "0x38a5357ce55c81add62abc84fb32981e2626adef",
# "0x4579a27af00a62c0eb156349f31b345c08386419",
# ]
# BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
# CHAT_ID = "6192278046"  # ID пользователя или канала
# CHECK_INTERVAL = 15  # секунд между проверками
#
# app = Flask(__name__)
#
# def get_transactions(address):
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={address}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY}"
#     )
#     resp = requests.get(url).json()
#     if resp["status"] == "1":
#         return resp["result"]
#     return []
#
# CHAT_IDS = [6192278046, 306507209]  # список id пользователей
#
# def send_to_telegram(message):
#     url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {
#             "chat_id": chat_id,
#             "text": message
#         }
#         try:
#             requests.post(url_base, data=payload)
#         except Exception as e:
#             print(f"Ошибка при отправке в чат {chat_id}: {e}")
#
# # def send_to_telegram(message):
# #     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
# #     payload = {
# #         "chat_id": CHAT_ID,
# #         "text": message
# #     }
# #     try:
# #         requests.post(url, data=payload)
# #     except Exception as e:
# #         print(f"Ошибка отправки в Telegram: {e}")
#
# def process_transactions():
#     seen_hashes = set()
#     while True:
#         try:
#             now = datetime.now(timezone.utc)
#             for address in ADDRESSES:
#                 txs = get_transactions(address)
#                 for tx in txs[:20]:
#                     # Фильтр по полю From
#                     if tx["from"].lower() != address:
#                         continue
#
#                     tx_hash = tx["hash"]
#                     if tx_hash in seen_hashes:
#                         continue
#
#                     tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
#                     # Фильтр по времени - только за последний час
#                     if (now - tx_time).total_seconds() > 2000:
#                         continue
#
#                     # Формируем сообщение
#                     token_name = tx.get("tokenName", "Unknown token")
#                     token_decimal = int(tx.get("tokenDecimal", 18))
#                     value = int(tx["value"]) / (10 ** token_decimal)
#
#                     message = (
#                         f"Address: {address}\n"
#                         f"Tx Hash:         {tx_hash}\n\n"
#                         f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
#                         f"Token Name:      {token_name}\n"
#                         f"Value:           {value}"
#                     )
#                     print(f"Отправляю в Telegram: {tx_hash}")
#                     send_to_telegram(message)
#                     seen_hashes.add(tx_hash)
#
#         except Exception as e:
#             print(f"Ошибка обработки транзакций: {e}")
#
#         time.sleep(CHECK_INTERVAL)
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
# @app.route("/status")
# def status():
#     return {"status": "ok"}
#
# def run_background():
#     thread = threading.Thread(target=process_transactions)
#     thread.daemon = True
#     thread.start()
#
# if __name__ == "__main__":
#     run_background()
#     app.run(host="0.0.0.0", port=10000)

# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# from web3 import Web3
# import json
# from decimal import Decimal  # Добавлен импорт Decimal в глобальный scope
#
# API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
# ADDRESSES = [
#     "0x38c503a438185cde29b5cf4dc1442fd6f074f1cc",
#     "0x285866acb0d60105b4ed350a463361c2d9afa0e2",
#     "0x38a5357ce55c81add62abc84fb32981e2626adef",
#     "0x4579a27af00a62c0eb156349f31b345c08386419",
# ]
# BOT_TOKEN = "7652720412:AAHEpoBovaezzfQqrmoli_3uY-EfzYFweZ0"
#
# CHAT_ID = "6192278046"  # ID пользователя или канала
# CHECK_INTERVAL = 15  # секунд между проверками
#
# app = Flask(__name__)
#
#
# def get_transactions(address):
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={address}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY}"
#     )
#     resp = requests.get(url).json()
#     if resp["status"] == "1":
#         return resp["result"]
#     return []
#
#
# CHAT_IDS = [6192278046, 306507209]  # список id пользователей
#
#
# def send_to_telegram(message):
#     url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {"chat_id": chat_id, "text": message}
#         try:
#             resp = requests.post(url_base, data=payload, timeout=10)
#             if resp.status_code != 200:
#                 print(f"Ошибка Telegram ({chat_id}): {resp.text}")
#         except Exception as e:
#             print(f"Ошибка при отправке в чат {chat_id}: {e}")
#
#
# def get_paraswap_rate():
#     """Получение текущего рыночного курса USDT/USDC для небольшого объема (1000 USDT)."""
#     USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # USDT контракт
#     USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC контракт
#     NETWORK = 1  # Ethereum Mainnet
#     PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
#     params = {
#         "srcToken": USDT_ADDRESS,
#         "destToken": USDC_ADDRESS,
#         "amount": "100000000000",  # 1000 USDT в wei
#         "side": "SELL",
#         "network": str(NETWORK),
#         "version": "6.2"
#     }
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
#         "Accept": "application/json"
#     }
#     try:
#         response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=20)
#         response.raise_for_status()
#         data = response.json()
#         if "error" in data:
#             print(f"Ошибка при получении курса: {data['error']}")
#             return None
#         rate = Decimal(data["priceRoute"]["destAmount"]) / Decimal(data["priceRoute"]["srcAmount"])
#         print(f"Текущий курс USDT/USDC: {rate:.6f}")
#         return rate
#     except Exception as e:
#         print(f"Ошибка получения курса: {e}")
#         return None
#
#
# def get_paraswap_quote(amount_usdt):
#     """Получение котировки ParaSwap для указанной суммы USDT."""
#     USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
#     USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
#     SLIPPAGE = Decimal("0")  # 0.01%
#     NETWORK = 1
#     INCLUDE_DEXES = "FluidDex"
#     PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
#     amount_wei = int(amount_usdt * 10 ** 6)
#     params = {
#         "srcToken": USDT_ADDRESS,
#         "destToken": USDC_ADDRESS,
#         "amount": str(amount_wei),
#         "side": "SELL",
#         "network": str(NETWORK),
#         "version": "6.2",
#         "includeDEXes": INCLUDE_DEXES,
#         "excludeContractMethodsWithoutFeeModel": "true"
#     }
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
#         "Accept": "application/json"
#     }
#     try:
#         response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=20)
#         response.raise_for_status()
#         data = response.json()
#         if "error" in data:
#             print(f"ParaSwap API ошибка: {data['error']}")
#             return None, None
#         amount_out = Decimal(data["priceRoute"]["destAmount"]) / Decimal(10 ** 6)
#         amount_out_with_slippage = amount_out * (1 - SLIPPAGE)
#         return amount_out, amount_out_with_slippage
#     except Exception as e:
#         print(f"Ошибка получения котировки: {e}")
#         return None, None
#
#
# def process_paraswap_alert():
#     """Проверка курсов и отправка алерта каждые 30 минут."""
#     AMOUNT_USDT = Decimal("1779963")
#     THRESHOLD = Decimal("1.000210") # !!!parametr
#     while True:
#         try:
#             print(f"[{datetime.now(timezone.utc)}] Проверка ParaSwap курсов...")
#             market_rate = get_paraswap_rate()
#             if market_rate is None:
#                 print("Не удалось получить рыночный курс, пропуск...")
#                 time.sleep(1800)  # 30 мин
#                 continue
#
#             quote_out, quote_out_slippage = get_paraswap_quote(AMOUNT_USDT)
#             if quote_out is None or quote_out_slippage is None:
#                 print("Не удалось получить котировку, пропуск...")
#                 time.sleep(1800) #30
#                 continue
#
#             para_ratio = quote_out_slippage / AMOUNT_USDT
#             print(f"ParaSwap отношение (с slippage): {para_ratio:.6f}")
#
#             if market_rate > THRESHOLD or para_ratio > THRESHOLD:
#                 message = (
#                     f"🚨 АЛЕРТ: Высокий курс!\n"
#                     f"Текущий курс USDT/USDC = {market_rate:.6f}\n"
#                     f"ParaSwap on USDT/USDC = {para_ratio:.6f}\n"
#                     f"(Порог: {THRESHOLD})"
#                 )
#                 print(f"Отправляю алерт в Telegram: {message}")
#                 send_to_telegram(message)
#
#         except Exception as e:
#             print(f"Ошибка в process_paraswap_alert: {e}")
#
#         time.sleep(180)  # 30 минут!!!
#
#
# def process_transactions():
#     seen_hashes = set()
#     while True:
#         try:
#             now = datetime.now(timezone.utc)
#             for address in ADDRESSES:
#                 txs = get_transactions(address)
#                 for tx in txs[:20]:
#                     # Фильтр по полю From
#                     if tx["from"].lower() != address:
#                         continue
#
#                     tx_hash = tx["hash"]
#                     if tx_hash in seen_hashes:
#                         continue
#
#                     tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
#                     # Фильтр по времени - только за последний час
#                     if (now - tx_time).total_seconds() > 2000:
#                         continue
#
#                     # Формируем сообщение
#                     token_name = tx.get("tokenName", "Unknown token")
#                     token_decimal = int(tx.get("tokenDecimal", 18))
#                     value = int(tx["value"]) / (10 ** token_decimal)
#
#                     message = (
#                         f"Address: {address}\n"
#                         f"Tx Hash:         {tx_hash}\n\n"
#                         f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
#                         f"Token Name:      {token_name}\n"
#                         f"Value:           {value}"
#                     )
#                     print(f"Отправляю в Telegram: {tx_hash}")
#                     send_to_telegram(message)
#                     seen_hashes.add(tx_hash)
#
#         except Exception as e:
#             print(f"Ошибка обработки транзакций: {e}")
#
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
# @app.route("/test")
# def test():
#     send_to_telegram("✅ Тестовое сообщение от Render")
#     return {"status": "sent"}
#
#
# @app.route("/status")
# def status():
#     return {
#         "status": "ok",
#         "threads": [t.name for t in threading.enumerate()]
#     }
#
# def run_background():
#     # Thread для транзакций
#     # tx_thread = threading.Thread(target=process_transactions)
#     # tx_thread.daemon = True
#     # tx_thread.start()
#
#     # Thread для ParaSwap алертов (каждые 30 мин)
#     alert_thread = threading.Thread(target=process_paraswap_alert)
#     alert_thread.daemon = True
#     alert_thread.start()
#
# run_background()
#
# if __name__ == "__main__":
#
#     import os
#
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
#     #app.run(host="0.0.0.0", port=10000)
#
#
# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# import json
# from decimal import Decimal
# import logging  # Добавлено для логов
#
# # Настройка логирования (видно в Render logs)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# API_KEY = "1F4W48VXNQ1CY1YCU9BAD7V27TIY4YDEX4"
# ADDRESSES = [
#     "0x38c503a438185cde29b5cf4dc1442fd6f074f1cc",
#     "0x285866acb0d60105b4ed350a463361c2d9afa0e2",
#     "0x38a5357ce55c81add62abc84fb32981e2626adef",
#     "0x4579a27af00a62c0eb156349f31b345c08386419",
# ]
# BOT_TOKEN = "7652720412:AAHEpoBovaezzfQqrmoli_3uY-EfzYFweZ0"
# CHAT_IDS = [6192278046, 306507209]  # Список ID чатов (int или str)
# CHECK_INTERVAL = 15  # секунд между проверками
#
# app = Flask(__name__)
#
#
# def get_transactions(address):
#     url = (
#         f"https://api.etherscan.io/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={address}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY}"
#     )
#     try:
#         resp = requests.get(url, timeout=30).json()  # Увеличен timeout
#         if resp["status"] == "1":
#             return resp["result"]
#         else:
#             logger.warning(f"Etherscan API error for {address}: {resp.get('message', 'Unknown')}")
#             return []
#     except Exception as e:
#         logger.error(f"Error fetching transactions for {address}: {e}")
#         return []
#
#
# def send_to_telegram(message):
#     url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {"chat_id": str(chat_id), "text": message}  # chat_id как str
#         try:
#             resp = requests.post(url_base, data=payload, timeout=30)
#             logger.info(f"Telegram response for {chat_id}: Status {resp.status_code}, Body: {resp.text[:200]}...")  # Логируем ответ
#             if resp.status_code != 200:
#                 logger.error(f"Ошибка Telegram ({chat_id}): {resp.text}")
#         except Exception as e:
#             logger.error(f"Ошибка при отправке в чат {chat_id}: {e}")
#
#
# def get_paraswap_rate():
#     """Получение текущего рыночного курса USDT/USDC для небольшого объема (1000 USDT)."""
#     USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # USDT контракт
#     USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC контракт
#     NETWORK = 1  # Ethereum Mainnet
#     PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
#     params = {
#         "srcToken": USDT_ADDRESS,
#         "destToken": USDC_ADDRESS,
#         "amount": "100000000000",  # 1000 USDT (с decimals=6)
#         "side": "SELL",
#         "network": str(NETWORK),
#         "version": "6.2"
#     }
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
#         "Accept": "application/json"
#     }
#     try:
#         response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         if "error" in data:
#             logger.error(f"Ошибка при получении курса: {data['error']}")
#             return None
#         rate = Decimal(data["priceRoute"]["destAmount"]) / Decimal(data["priceRoute"]["srcAmount"])
#         logger.info(f"Текущий курс USDT/USDC: {rate:.6f}")
#         return rate
#     except Exception as e:
#         logger.error(f"Ошибка получения курса: {e}")
#         return None
#
#
# def get_paraswap_quote(amount_usdt):
#     """Получение котировки ParaSwap для указанной суммы USDT."""
#     USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
#     USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
#     SLIPPAGE = Decimal("0.0001")  # 0.01% (исправлено с 0)
#     NETWORK = 1
#     INCLUDE_DEXES = "FluidDex"
#     PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
#     amount_wei = int(amount_usdt * 10 ** 6)  # USDT decimals=6
#     params = {
#         "srcToken": USDT_ADDRESS,
#         "destToken": USDC_ADDRESS,
#         "amount": str(amount_wei),
#         "side": "SELL",
#         "network": str(NETWORK),
#         "version": "6.2",
#         "includeDEXes": INCLUDE_DEXES,
#         "excludeContractMethodsWithoutFeeModel": "true"
#     }
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
#         "Accept": "application/json"
#     }
#     try:
#         response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         if "error" in data:
#             logger.error(f"ParaSwap API ошибка: {data['error']}")
#             return None, None
#         amount_out = Decimal(data["priceRoute"]["destAmount"]) / Decimal(10 ** 6)  # USDC decimals=6
#         amount_out_with_slippage = amount_out * (1 - SLIPPAGE)
#         logger.info(f"ParaSwap quote для {amount_usdt} USDT: {amount_out} USDC, с slippage: {amount_out_with_slippage}")
#         return amount_out, amount_out_with_slippage
#     except Exception as e:
#         logger.error(f"Ошибка получения котировки: {e}")
#         return None, None
#
#
# def process_paraswap_alert():
#     """Проверка курсов и отправка алерта каждые 30 минут."""
#     AMOUNT_USDT = Decimal("1779963")
#     THRESHOLD = Decimal("1.000210")  # !!!parametr
#     logger.info("ParaSwap alert thread started!")  # Для проверки запуска
#     while True:
#         try:
#             logger.info(f"[{datetime.now(timezone.utc)}] Проверка ParaSwap курсов...")
#             market_rate = get_paraswap_rate()
#             if market_rate is None:
#                 logger.warning("Не удалось получить рыночный курс, пропуск...")
#                 time.sleep(1800)  # 30 мин
#                 continue
#
#             quote_out, quote_out_slippage = get_paraswap_quote(AMOUNT_USDT)
#             if quote_out is None or quote_out_slippage is None:
#                 logger.warning("Не удалось получить котировку, пропуск...")
#                 time.sleep(1800)  # 30 мин
#                 continue
#
#             para_ratio = quote_out_slippage / AMOUNT_USDT
#             logger.info(f"ParaSwap отношение (с slippage): {para_ratio:.6f}")
#
#             if market_rate > THRESHOLD or para_ratio > THRESHOLD:
#                 message = (
#                     f"🚨 АЛЕРТ: Высокий курс!\n"
#                     f"Текущий курс USDT/USDC = {market_rate:.6f}\n"
#                     f"ParaSwap on USDT/USDC = {para_ratio:.6f}\n"
#                     f"(Порог: {THRESHOLD})"
#                 )
#                 logger.info(f"Отправляю алерт в Telegram: {message[:100]}...")
#                 send_to_telegram(message)
#             else:
#                 logger.info("Курс в норме, алерт не отправлен.")
#
#         except Exception as e:
#             logger.error(f"Ошибка в process_paraswap_alert: {e}")
#
#         time.sleep(1800)  # 30 минут (исправлено с 180)
#
#
# def process_transactions():
#     seen_hashes = set()
#     logger.info("Transactions thread started!")  # Для проверки запуска
#     while True:
#         try:
#             now = datetime.now(timezone.utc)
#             for address in ADDRESSES:
#                 txs = get_transactions(address)
#                 for tx in txs[:20]:
#                     # Фильтр по полю From
#                     if tx["from"].lower() != address.lower():  # .lower() для безопасности
#                         continue
#
#                     tx_hash = tx["hash"]
#                     if tx_hash in seen_hashes:
#                         continue
#
#                     tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
#                     # Фильтр по времени - только за последние 2000 сек (~33 мин, вместо часа)
#                     if (now - tx_time).total_seconds() > 2000:
#                         continue
#
#                     # Формируем сообщение
#                     token_name = tx.get("tokenName", "Unknown token")
#                     token_decimal = int(tx.get("tokenDecimal", 18))
#                     value = int(tx["value"]) / (10 ** token_decimal)
#
#                     message = (
#                         f"Address: {address}\n"
#                         f"Tx Hash: {tx_hash}\n\n"
#                         f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
#                         f"Token Name: {token_name}\n"
#                         f"Value: {value}"
#                     )
#                     logger.info(f"Отправляю в Telegram: {tx_hash}")
#                     send_to_telegram(message)
#                     seen_hashes.add(tx_hash)
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки транзакций: {e}")
#
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
#
# @app.route("/test")
# def test():
#     send_to_telegram("✅ Тестовое сообщение от Render")
#     return {"status": "sent"}
#
#
# @app.route("/status")
# def status():
#     return {
#         "status": "ok",
#         "threads": [t.name for t in threading.enumerate()],
#         "active_threads_count": threading.active_count()  # Для диагностики
#     }
#
#
# def run_background():
#     """Запуск фоновых потоков с обработкой ошибок."""
#     try:
#         # Thread для транзакций (раскомментировано)
#         tx_thread = threading.Thread(target=process_transactions, name="TxMonitor")
#         tx_thread.daemon = True
#         tx_thread.start()
#         logger.info("Tx thread started")
#
#         # Thread для ParaSwap алертов
#         alert_thread = threading.Thread(target=process_paraswap_alert, name="ParaSwapAlert")
#         alert_thread.daemon = True
#         alert_thread.start()
#         logger.info("Alert thread started")
#     except Exception as e:
#         logger.error(f"Ошибка запуска фоновых потоков: {e}")
#
#
# # Запуск фоновых задач
# run_background()
#
# if __name__ == "__main__":
#     import os
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
#     # app.run(host="0.0.0.0", port=10000)  # Локальный порт закомментирован

# import requests
# import time
# import threading
# from datetime import datetime, timezone
# from flask import Flask
# import json
# from decimal import Decimal
# import logging  # Для логов
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# # !!! ОБНОВИТЕ ЗДЕСЬ !!!
# API_KEY_ETHERSCAN = "YOUR_NEW_ETHERSCAN_API_KEY_HERE"  # Создайте на https://etherscan.io/myapikey
# API_KEY_PARASWAP = "YOUR_NEW_PARASWAP_API_KEY_HERE"   # Создайте на https://developers.paraswap.network/
#
# ADDRESSES = [
#     "0x38c503a438185cde29b5cf4dc1442fd6f074f1cc",
#     "0x285866acb0d60105b4ed350a463361c2d9afa0e2",
#     "0x38a5357ce55c81add62abc84fb32981e2626adef",
#     "0x4579a27af00a62c0eb156349f31b345c08386419",
# ]
# BOT_TOKEN = "7652720412:AAHEpoBovaezzfQqrmoli_3uY-EfzYFweZ0"
# CHAT_IDS = [6192278046, 306507209]
# CHECK_INTERVAL = 30  # Увеличено до 30 сек, чтобы не превысить rate limit
#
# app = Flask(__name__)
#
#
# def retry_request(func, *args, max_retries=3, **kwargs):
#     """Retry wrapper для API с backoff."""
#     for attempt in range(max_retries):
#         try:
#             return func(*args, **kwargs)
#         except requests.exceptions.RequestException as e:
#             if attempt == max_retries - 1:
#                 raise e
#             time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
#
#
# def get_transactions(address):
#     url = (
#         f"https://api.etherscan.io/v2/api"
#         f"?module=account"
#         f"&action=tokentx"
#         f"&address={address}"
#         f"&startblock=0"
#         f"&endblock=99999999"
#         f"&sort=desc"
#         f"&apikey={API_KEY_ETHERSCAN}"
#     )
#     try:
#         resp = requests.get(url, timeout=30).json()
#         logger.info(f"Etherscan response for {address}: {resp.get('status', 'Unknown')}")
#         if resp["status"] == "1":
#             return resp["result"]
#         else:
#             logger.warning(f"Etherscan API error for {address}: {resp.get('message', 'Unknown')} - {resp.get('result', 'No details')}")
#             return []
#     except Exception as e:
#         logger.error(f"Error fetching transactions for {address}: {e}")
#         return []
#
#
# def send_to_telegram(message):
#     url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         payload = {"chat_id": str(chat_id), "text": message}
#         try:
#             resp = requests.post(url_base, data=payload, timeout=30)
#             logger.info(f"Telegram response for {chat_id}: Status {resp.status_code}, Body: {resp.text[:200]}...")
#             if resp.status_code != 200:
#                 logger.error(f"Ошибка Telegram ({chat_id}): {resp.text}")
#         except Exception as e:
#             logger.error(f"Ошибка при отправке в чат {chat_id}: {e}")
#
#
# def get_paraswap_rate():
#     USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
#     USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
#     NETWORK = 1
#     PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
#     params = {
#         "srcToken": USDT_ADDRESS,
#         "destToken": USDC_ADDRESS,
#         "amount": "100000000000",  # 1000 USDT (6 decimals)
#         "side": "SELL",
#         "network": str(NETWORK),
#         "version": "6.2",
#         "srcDecimals": "6",
#         "destDecimals": "6"  # Добавлено по docs
#     }
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
#         "Accept": "application/json",
#     }
#     if API_KEY_PARASWAP != "YOUR_NEW_PARASWAP_API_KEY_HERE":
#         headers["X-API-KEY"] = API_KEY_PARASWAP  # Добавлен API key
#
#     def make_request():
#         response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         if "error" in data:
#             raise ValueError(f"ParaSwap error: {data['error']}")
#         rate = Decimal(data["priceRoute"]["destAmount"]) / Decimal(data["priceRoute"]["srcAmount"])
#         logger.info(f"Текущий курс USDT/USDC: {rate:.6f}")
#         return rate
#
#     return retry_request(make_request)
#
#
# def get_paraswap_quote(amount_usdt):
#     USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
#     USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
#     SLIPPAGE = Decimal("0.0001")  # 0.01%
#     NETWORK = 1
#     INCLUDE_DEXES = "FluidDex"
#     PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
#     amount_wei = int(amount_usdt * 10 ** 6)
#     params = {
#         "srcToken": USDT_ADDRESS,
#         "destToken": USDC_ADDRESS,
#         "amount": str(amount_wei),
#         "side": "SELL",
#         "network": str(NETWORK),
#         "version": "6.2",
#         "includeDEXes": INCLUDE_DEXES,
#         "excludeContractMethodsWithoutFeeModel": "true",
#         "srcDecimals": "6",
#         "destDecimals": "6"
#     }
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
#         "Accept": "application/json",
#     }
#     if API_KEY_PARASWAP != "YOUR_NEW_PARASWAP_API_KEY_HERE":
#         headers["X-API-KEY"] = API_KEY_PARASWAP
#
#     def make_request():
#         response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         if "error" in data:
#             raise ValueError(f"ParaSwap API ошибка: {data['error']}")
#         amount_out = Decimal(data["priceRoute"]["destAmount"]) / Decimal(10 ** 6)
#         amount_out_with_slippage = amount_out * (1 - SLIPPAGE)
#         logger.info(f"ParaSwap quote для {amount_usdt} USDT: {amount_out} USDC, с slippage: {amount_out_with_slippage}")
#         return amount_out, amount_out_with_slippage
#
#     return retry_request(make_request)
#
#
# def process_paraswap_alert():
#     AMOUNT_USDT = Decimal("1779963")
#     THRESHOLD = Decimal("1.000210")
#     logger.info("ParaSwap alert thread started!")
#     while True:
#         try:
#             logger.info(f"[{datetime.now(timezone.utc)}] Проверка ParaSwap курсов...")
#             market_rate = get_paraswap_rate()
#             if market_rate is None:
#                 logger.warning("Не удалось получить рыночный курс, пропуск...")
#                 time.sleep(1800)
#                 continue
#
#             quote_out, quote_out_slippage = get_paraswap_quote(AMOUNT_USDT)
#             if quote_out is None or quote_out_slippage is None:
#                 logger.warning("Не удалось получить котировку, пропуск...")
#                 time.sleep(1800)
#                 continue
#
#             para_ratio = quote_out_slippage / AMOUNT_USDT
#             logger.info(f"ParaSwap отношение (с slippage): {para_ratio:.6f}")
#
#             if market_rate > THRESHOLD or para_ratio > THRESHOLD:
#                 message = (
#                     f"🚨 АЛЕРТ: Высокий курс!\n"
#                     f"Текущий курс USDT/USDC = {market_rate:.6f}\n"
#                     f"ParaSwap on USDT/USDC = {para_ratio:.6f}\n"
#                     f"(Порог: {THRESHOLD})"
#                 )
#                 logger.info(f"Отправляю алерт в Telegram: {message[:100]}...")
#                 send_to_telegram(message)
#             else:
#                 logger.info("Курс в норме, алерт не отправлен.")
#
#         except Exception as e:
#             logger.error(f"Ошибка в process_paraswap_alert: {e}")
#
#         time.sleep(1800)  # 30 мин
#
#
# def process_transactions():
#     seen_hashes = set()
#     logger.info("Transactions thread started!")
#     while True:
#         try:
#             now = datetime.now(timezone.utc)
#             for address in ADDRESSES:
#                 txs = get_transactions(address)
#                 for tx in txs[:20]:
#                     if tx["from"].lower() != address.lower():
#                         continue
#
#                     tx_hash = tx["hash"]
#                     if tx_hash in seen_hashes:
#                         continue
#
#                     tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
#                     if (now - tx_time).total_seconds() > 2000:
#                         continue
#
#                     token_name = tx.get("tokenName", "Unknown token")
#                     token_decimal = int(tx.get("tokenDecimal", 18))
#                     value = int(tx["value"]) / (10 ** token_decimal)
#
#                     message = (
#                         f"Address: {address}\n"
#                         f"Tx Hash: {tx_hash}\n\n"
#                         f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
#                         f"Token Name: {token_name}\n"
#                         f"Value: {value}"
#                     )
#                     logger.info(f"Отправляю в Telegram: {tx_hash}")
#                     send_to_telegram(message)
#                     seen_hashes.add(tx_hash)
#
#         except Exception as e:
#             logger.error(f"Ошибка обработки транзакций: {e}")
#
#         time.sleep(CHECK_INTERVAL)
#
#
# @app.route("/")
# def home():
#     return "Parser is running!"
#
#
# @app.route("/test")
# def test():
#     send_to_telegram("✅ Тестовое сообщение от Render")
#     return {"status": "sent"}
#
#
# @app.route("/status")
# def status():
#     return {
#         "status": "ok",
#         "threads": [t.name for t in threading.enumerate()],
#         "active_threads_count": threading.active_count()
#     }
#
#
# def run_background():
#     try:
#         tx_thread = threading.Thread(target=process_transactions, name="TxMonitor")
#         tx_thread.daemon = True
#         tx_thread.start()
#         logger.info("Tx thread started")
#
#         alert_thread = threading.Thread(target=process_paraswap_alert, name="ParaSwapAlert")
#         alert_thread.daemon = True
#         alert_thread.start()
#         logger.info("Alert thread started")
#     except Exception as e:
#         logger.error(f"Ошибка запуска фоновых потоков: {e}")
#
#
# run_background()
#
# if __name__ == "__main__":
#     import os
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
#


import requests
import time
import threading
from datetime import datetime, timezone
from flask import Flask
import json
from decimal import Decimal
import logging  # Для логов
import random  # Для ротации User-Agent

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# !!! ОБНОВИТЕ ЗДЕСЬ !!!
API_KEY_ETHERSCAN = "1F4W48VXNQ1CY1YCU9BAD7V27TIY4YDEX4"  # Создайте на https://etherscan.io/myapikey

ADDRESSES = [
    "0x38c503a438185cde29b5cf4dc1442fd6f074f1cc",
    "0x285866acb0d60105b4ed350a463361c2d9afa0e2",
 #   "0x38a5357ce55c81add62abc84fb32981e2626adef",
]
BOT_TOKEN = "7652720412:AAHEpoBovaezzfQqrmoli_3uY-EfzYFweZ0"
CHAT_IDS = [6192278046, 306507209]
CHECK_INTERVAL = 30

app = Flask(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
    }

def retry_request(func, *args, max_retries=3, **kwargs):
    """Retry wrapper для API с backoff."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Retry {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt + random.uniform(0, 1))  # Backoff + jitter

def get_transactions(address):
    url = (
        f"https://api.etherscan.io/api"
        f"?module=account"
        f"&action=tokentx"
        f"&address={address}"
        f"&startblock=0"
        f"&endblock=99999999"
        f"&sort=desc"
        f"&apikey={API_KEY_ETHERSCAN}"
    )
    try:
        resp = requests.get(url, timeout=30).json()
        logger.info(f"Etherscan response for {address}: {resp.get('status', 'Unknown')}")
        if resp["status"] == "1":
            return resp["result"]
        else:
            logger.warning(f"Etherscan API error for {address}: {resp.get('message', 'Unknown')}")
            return []
    except Exception as e:
        logger.error(f"Error fetching transactions for {address}: {e}")
        return []

def send_to_telegram(message):
    url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        payload = {"chat_id": str(chat_id), "text": message}
        try:
            resp = requests.post(url_base, data=payload, timeout=30)
            logger.info(f"Telegram response for {chat_id}: Status {resp.status_code}")
            if resp.status_code != 200:
                logger.error(f"Ошибка Telegram ({chat_id}): {resp.text}")
        except Exception as e:
            logger.error(f"Ошибка при отправке в чат {chat_id}: {e}")

def get_paraswap_rate():
    USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    NETWORK = 1
    PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
    params = {
        "srcToken": USDT_ADDRESS,
        "destToken": USDC_ADDRESS,
        "amount": "100000000000",  # 1000 USDT
        "side": "SELL",
        "network": str(NETWORK),
        "version": "6.2",
        "srcDecimals": "6",
        "destDecimals": "6"
    }
    headers = get_random_headers()  # Ротация UA

    def make_request():
        response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"ParaSwap error: {data['error']}")
        rate = Decimal(data["priceRoute"]["destAmount"]) / Decimal(data["priceRoute"]["srcAmount"])
        logger.info(f"ParaSwap: Текущий курс USDT/USDC: {rate:.6f}")
        return rate

    try:
        return retry_request(make_request)
    except Exception as e:
        logger.error(f"ParaSwap failed after retries: {e}. Falling back to 1inch...")
        return get_1inch_rate()  # Fallback

def get_1inch_rate():
    """Fallback: 1inch API (публичный, без ключа)."""
    USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    AMOUNT = "100000000000"  # 1000 USDT
    URL = f"https://api.1inch.io/v5.0/1/quote?fromTokenAddress={USDT_ADDRESS}&toTokenAddress={USDC_ADDRESS}&amount={AMOUNT}"
    headers = get_random_headers()

    def make_request():
        response = requests.get(URL, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        rate = Decimal(data["toTokenAmount"]) / Decimal(AMOUNT)
        logger.info(f"1inch fallback: Текущий курс USDT/USDC: {rate:.6f}")
        return rate

    return retry_request(make_request)

# Аналогично обновите get_paraswap_quote с fallback на 1inch_quote (если нужно, добавьте функцию)

def get_paraswap_quote(amount_usdt):
    USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    SLIPPAGE = Decimal("0")  # 0.01%
    NETWORK = 1
    INCLUDE_DEXES = "FluidDex"
    PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
    amount_wei = int(amount_usdt * 10 ** 6)
    params = {
        "srcToken": USDT_ADDRESS,
        "destToken": USDC_ADDRESS,
        "amount": str(amount_wei),
        "side": "SELL",
        "network": str(NETWORK),
        "version": "6.2",
        "includeDEXes": INCLUDE_DEXES,
        "excludeContractMethodsWithoutFeeModel": "true",
        "srcDecimals": "6",
        "destDecimals": "6"
    }
    headers = get_random_headers()

    def make_request():
        response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"ParaSwap API ошибка: {data['error']}")
        amount_out = Decimal(data["priceRoute"]["destAmount"]) / Decimal(10 ** 6)
        amount_out_with_slippage = amount_out * (1 - SLIPPAGE)
        logger.info(f"ParaSwap quote для {amount_usdt} USDT: {amount_out} USDC, с slippage: {amount_out_with_slippage}")
        return amount_out, amount_out_with_slippage

    try:
        return retry_request(make_request)
    except Exception as e:
        logger.error(f"ParaSwap quote failed: {e}. Falling back to 1inch...")
        return get_1inch_quote(amount_usdt)

def get_1inch_quote(amount_usdt):
    """Fallback для quote."""
    USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    AMOUNT = str(int(amount_usdt * 10 ** 6))
    URL = f"https://api.1inch.io/v5.0/1/quote?fromTokenAddress={USDT_ADDRESS}&toTokenAddress={USDC_ADDRESS}&amount={AMOUNT}"
    headers = get_random_headers()

    def make_request():
        response = requests.get(URL, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        amount_out = Decimal(data["toTokenAmount"]) / Decimal(10 ** 6)
        amount_out_with_slippage = amount_out * (1 - Decimal("0"))
        logger.info(f"1inch quote для {amount_usdt} USDT: {amount_out} USDC")
        return amount_out, amount_out_with_slippage

    return retry_request(make_request)

def process_paraswap_alert():
    AMOUNT_USDT = Decimal("1779963")
    THRESHOLD = Decimal("1.000210")
    logger.info("ParaSwap alert thread started!")
    while True:
        try:
            logger.info(f"[{datetime.now(timezone.utc)}] Проверка ParaSwap курсов...")
            market_rate = get_paraswap_rate()
            if market_rate is None:
                logger.warning("Не удалось получить рыночный курс, пропуск...")
                time.sleep(3600)  # 1 час
                continue

            quote_out, quote_out_slippage = get_paraswap_quote(AMOUNT_USDT)
            if quote_out is None or quote_out_slippage is None:
                logger.warning("Не удалось получить котировку, пропуск...")
                time.sleep(3600)
                continue

            para_ratio = quote_out_slippage / AMOUNT_USDT
            logger.info(f"ParaSwap отношение (с slippage): {para_ratio:.6f}")

            if market_rate > THRESHOLD or para_ratio > THRESHOLD:
                message = (
                    f"🚨 АЛЕРТ: Высокий курс!\n"
                    f"Текущий курс USDT/USDC = {market_rate:.6f}\n"
                    f"ParaSwap on USDT/USDC = {para_ratio:.6f}\n"
                    f"(Порог: {THRESHOLD})"
                )
                logger.info(f"Отправляю алерт в Telegram: {message[:100]}...")
                send_to_telegram(message)
            else:
                logger.info("Курс в норме, алерт не отправлен.")

        except Exception as e:
            logger.error(f"Ошибка в process_paraswap_alert: {e}")

        time.sleep(36)  # 1 час

# Остальные функции (process_transactions, routes, run_background) без изменений...

def process_transactions():
    seen_hashes = set()
    logger.info("Transactions thread started!")
    while True:
        try:
            now = datetime.now(timezone.utc)
            for address in ADDRESSES:
                txs = get_transactions(address)
                for tx in txs[:20]:
                    if tx["from"].lower() != address.lower():
                        continue

                    tx_hash = tx["hash"]
                    if tx_hash in seen_hashes:
                        continue

                    tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
                    if (now - tx_time).total_seconds() > 2000:
                        continue

                    token_name = tx.get("tokenName", "Unknown token")
                    token_decimal = int(tx.get("tokenDecimal", 18))
                    value = int(tx["value"]) / (10 ** token_decimal)

                    message = (
                        f"Address: {address}\n"
                        f"Tx Hash: {tx_hash}\n\n"
                        f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
                        f"Token Name: {token_name}\n"
                        f"Value: {value}"
                    )
                    logger.info(f"Отправляю в Telegram: {tx_hash}")
                    send_to_telegram(message)
                    seen_hashes.add(tx_hash)

        except Exception as e:
            logger.error(f"Ошибка обработки транзакций: {e}")

        time.sleep(CHECK_INTERVAL)

@app.route("/")
def home():
    return "Parser is running!"

@app.route("/test")
def test():
    send_to_telegram("✅ Тестовое сообщение от Render")
    return {"status": "sent"}

@app.route("/status")
def status():
    return {
        "status": "ok",
        "threads": [t.name for t in threading.enumerate()],
        "active_threads_count": threading.active_count()
    }

def run_background():
    try:
        tx_thread = threading.Thread(target=process_transactions, name="TxMonitor")
        tx_thread.daemon = True
        tx_thread.start()
        logger.info("Tx thread started")

        # alert_thread = threading.Thread(target=process_paraswap_alert, name="ParaSwapAlert")
        # alert_thread.daemon = True
        # alert_thread.start()
        # logger.info("Alert thread started")
    except Exception as e:
        logger.error(f"Ошибка запуска фоновых потоков: {e}")

run_background()

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))