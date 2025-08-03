

import requests
import time
from datetime import datetime, timedelta

# Настройки
API_KEY = "4ZNE8XMUQMTKWCWB6N112S2KI3A4TF2DBI"
ADDRESS = "0x4579a27af00a62c0eb156349f31b345c08386419".lower()
BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
CHAT_ID = "6192278046"  # ID пользователя или канала
CHECK_INTERVAL = 15  # секунд между проверками
WINDOW_MINUTES = 5  # интервал поиска в минутах

# Хранилище уже отправленных транзакций
sent_tx_hashes = set()

def get_token_transactions(address, count=100):
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
    try:
        resp = requests.get(url, timeout=10).json()
        if resp["status"] == "1":
            return resp["result"][:count]
    except Exception as e:
        print(f"Ошибка при получении транзакций: {e}")
    return []

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

def format_tx_message(tx):
    timestamp_utc = datetime.utcfromtimestamp(int(tx['timeStamp']))
    timestamp_msk = timestamp_utc + timedelta(hours=3)
    value = int(tx['value']) / (10 ** int(tx['tokenDecimal']))
    return (
        f"<b>Новая транзакция:</b>\n"
        f"<b>Tx Hash:</b> <code>{tx['hash']}</code>\n"
        f"<b>Время:</b> {timestamp_msk.strftime('%Y-%m-%d %H:%M:%S')} (UTC+3)\n"
        f"<b>Token:</b> {tx['tokenName']} ({tx['tokenSymbol']})\n"
        f"<b>Value:</b> {value:.8f}"
    )

def main_loop():
    print("⏳ Запуск мониторинга...")
    while True:
        now = datetime.utcnow()
        recent_cutoff = now - timedelta(minutes=WINDOW_MINUTES)

        transactions = get_token_transactions(ADDRESS, count=100)
        for tx in transactions:
            if tx['from'].lower() != ADDRESS:
                continue
            tx_hash = tx['hash']
            if tx_hash in sent_tx_hashes:
                continue

            tx_time = datetime.utcfromtimestamp(int(tx['timeStamp']))
            if tx_time < recent_cutoff:
                continue  # старее 60 минут

            # Отправляем сообщение
            message = format_tx_message(tx)
            send_telegram_message(message)
            sent_tx_hashes.add(tx_hash)
            print(f"✅ Отправлено: {tx_hash}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()
