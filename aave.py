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

import requests
import time
import threading
from datetime import datetime, timezone
from flask import Flask
from web3 import Web3
import json
from decimal import Decimal  # Добавлен импорт Decimal в глобальный scope

API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
ADDRESSES = [
    "0x38c503a438185cde29b5cf4dc1442fd6f074f1cc",
    "0x285866acb0d60105b4ed350a463361c2d9afa0e2",
    "0x38a5357ce55c81add62abc84fb32981e2626adef",
]
BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
CHAT_ID = "6192278046"  # ID пользователя или канала
CHECK_INTERVAL = 15  # секунд между проверками

app = Flask(__name__)


def get_transactions(address):
    url = (
        f"https://api.etherscan.io/api"
        f"?module=account"
        f"&action=tokentx"
        f"&address={address}"
        f"&startblock=0"
        f"&endblock=99999999"
        f"&sort=desc"
        f"&apikey={API_KEY}"
    )
    resp = requests.get(url).json()
    if resp["status"] == "1":
        return resp["result"]
    return []


CHAT_IDS = [6192278046, 306507209]  # список id пользователей


def send_to_telegram(message):
    url_base = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": message
        }
        try:
            requests.post(url_base, data=payload)
        except Exception as e:
            print(f"Ошибка при отправке в чат {chat_id}: {e}")


def get_paraswap_rate():
    """Получение текущего рыночного курса USDT/USDC для небольшого объема (1000 USDT)."""
    USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # USDT контракт
    USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC контракт
    NETWORK = 1  # Ethereum Mainnet
    PARASWAP_QUOTE_URL = "https://api.paraswap.io/prices"
    params = {
        "srcToken": USDT_ADDRESS,
        "destToken": USDC_ADDRESS,
        "amount": "100000000000",  # 1000 USDT в wei
        "side": "SELL",
        "network": str(NETWORK),
        "version": "6.2"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            print(f"Ошибка при получении курса: {data['error']}")
            return None
        rate = Decimal(data["priceRoute"]["destAmount"]) / Decimal(data["priceRoute"]["srcAmount"])
        print(f"Текущий курс USDT/USDC: {rate:.6f}")
        return rate
    except Exception as e:
        print(f"Ошибка получения курса: {e}")
        return None


def get_paraswap_quote(amount_usdt):
    """Получение котировки ParaSwap для указанной суммы USDT."""
    USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    SLIPPAGE = Decimal("0.001")  # 0.1%
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
        "excludeContractMethodsWithoutFeeModel": "true"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(PARASWAP_QUOTE_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            print(f"ParaSwap API ошибка: {data['error']}")
            return None, None
        amount_out = Decimal(data["priceRoute"]["destAmount"]) / Decimal(10 ** 6)
        amount_out_with_slippage = amount_out * (1 - SLIPPAGE)
        return amount_out, amount_out_with_slippage
    except Exception as e:
        print(f"Ошибка получения котировки: {e}")
        return None, None


def process_paraswap_alert():
    """Проверка курсов и отправка алерта каждые 30 минут."""
    AMOUNT_USDT = Decimal("1779963")
    THRESHOLD = Decimal("1.000410")
    while True:
        try:
            print(f"[{datetime.now(timezone.utc)}] Проверка ParaSwap курсов...")
            market_rate = get_paraswap_rate()
            if market_rate is None:
                print("Не удалось получить рыночный курс, пропуск...")
                time.sleep(1800)  # 30 мин
                continue

            quote_out, quote_out_slippage = get_paraswap_quote(AMOUNT_USDT)
            if quote_out is None or quote_out_slippage is None:
                print("Не удалось получить котировку, пропуск...")
                time.sleep(1800)
                continue

            para_ratio = quote_out_slippage / AMOUNT_USDT
            print(f"ParaSwap отношение (с slippage): {para_ratio:.6f}")

            if market_rate > THRESHOLD or para_ratio > THRESHOLD:
                message = (
                    f"🚨 АЛЕРТ: Высокий курс!\n"
                    f"Текущий курс USDT/USDC = {market_rate:.6f}\n"
                    f"ParaSwap on USDT/USDC = {para_ratio:.6f}\n"
                    f"(Порог: {THRESHOLD})"
                )
                print(f"Отправляю алерт в Telegram: {message}")
                send_to_telegram(message)

        except Exception as e:
            print(f"Ошибка в process_paraswap_alert: {e}")

        time.sleep(1800)  # 30 минут


def process_transactions():
    seen_hashes = set()
    while True:
        try:
            now = datetime.now(timezone.utc)
            for address in ADDRESSES:
                txs = get_transactions(address)
                for tx in txs[:20]:
                    # Фильтр по полю From
                    if tx["from"].lower() != address:
                        continue

                    tx_hash = tx["hash"]
                    if tx_hash in seen_hashes:
                        continue

                    tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
                    # Фильтр по времени - только за последний час
                    if (now - tx_time).total_seconds() > 2000:
                        continue

                    # Формируем сообщение
                    token_name = tx.get("tokenName", "Unknown token")
                    token_decimal = int(tx.get("tokenDecimal", 18))
                    value = int(tx["value"]) / (10 ** token_decimal)

                    message = (
                        f"Address: {address}\n"
                        f"Tx Hash:         {tx_hash}\n\n"
                        f"Timestamp (UTC): {tx_time.strftime('%Y-%m-%d %H:%M:%S')} + 3 часа\n\n"
                        f"Token Name:      {token_name}\n"
                        f"Value:           {value}"
                    )
                    print(f"Отправляю в Telegram: {tx_hash}")
                    send_to_telegram(message)
                    seen_hashes.add(tx_hash)

        except Exception as e:
            print(f"Ошибка обработки транзакций: {e}")

        time.sleep(CHECK_INTERVAL)


@app.route("/")
def home():
    return "Parser is running!"


@app.route("/status")
def status():
    return {"status": "ok"}


def run_background():
    # Thread для транзакций
    tx_thread = threading.Thread(target=process_transactions)
    tx_thread.daemon = True
    tx_thread.start()

    # Thread для ParaSwap алертов (каждые 30 мин)
    alert_thread = threading.Thread(target=process_paraswap_alert)
    alert_thread.daemon = True
    alert_thread.start()


if __name__ == "__main__":
    run_background()
    app.run(host="0.0.0.0", port=10000)