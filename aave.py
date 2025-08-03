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


import requests
import time
import threading
from datetime import datetime, timezone
from flask import Flask
from web3 import Web3
import json

API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
ADDRESS = "0x4579a27af00a62c0eb156349f31b345c08386419".lower()
BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
CHAT_ID = "6192278046"  # ID пользователя или канала
CHECK_INTERVAL = 15  # секунд между проверками

app = Flask(__name__)

def get_transactions():
    url = (
        f"https://api.etherscan.io/api"
        f"?module=account"
        f"&action=tokentx"
        f"&address={ADDRESS}"
        f"&startblock=0"
        f"&endblock=99999999"
        f"&sort=desc"
        f"&apikey={API_KEY}"
    )
    resp = requests.get(url).json()
    if resp["status"] == "1":
        return resp["result"]
    return []

def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

def process_transactions():
    seen_hashes = set()
    while True:
        try:
            txs = get_transactions()
            now = datetime.now(timezone.utc)
            for tx in txs[:20]:
                # Фильтр по полю From
                if tx["from"].lower() != ADDRESS.lower():
                    continue

                tx_hash = tx["hash"]
                if tx_hash in seen_hashes:
                    continue

                tx_time = datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc)
                # Фильтр по времени - только за последний час
                if (now - tx_time).total_seconds() > 300:
                    continue

                # Формируем сообщение
                token_name = tx.get("tokenName", "Unknown token")
                token_decimal = int(tx.get("tokenDecimal", 18))
                value = int(tx["value"]) / (10 ** token_decimal)

                message = (
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
    thread = threading.Thread(target=process_transactions)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    run_background()
    app.run(host="0.0.0.0", port=10000)
