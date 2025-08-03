import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from datetime import datetime

# Конфигурация
BOT_TOKEN = "7652720412:AAFkPwpqFa3iRr23xw8rE9MYXtj_ptvq6kk"
CHAT_ID = "6192278046"  # ID пользователя или канала
CHECK_INTERVAL = 60  # Интервал проверки в секундах
PAGE_URLS = [
    "https://fluid.io/vaults/1/44/strategies/multiply",
    "https://fluid.io/vaults/1/127/strategies/multiply",
    "https://fluid.io/vaults/1/61/strategies/multiply"
]

# Словарь для хранения предыдущих значений
previous_values = {}


def send_to_telegram(message):
    """Отправка сообщения в Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print(f"Сообщение отправлено в Telegram: {message}")
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")


def parse_immediate_borrowable(page_url):
    """Парсинг страницы через Selenium и проверка Immediate Borrowable."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')  # Необходимо для Render.com
    options.add_argument('--disable-dev-shm-usage')  # Необходимо для Render.com
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(page_url)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )

        borrowable_label = None
        try:
            borrowable_label = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//*[contains(text(), "Immediate Borrowable")]'))
            )
            print(f"Элемент 'Immediate Borrowable' найден на {page_url}.")
        except:
            print(f"Элемент с текстом 'Immediate Borrowable' не найден на {page_url}.")
            with open(f'page_source_{page_url.split("/")[-3]}.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print(f"HTML страницы сохранён в page_source_{page_url.split('/')[-3]}.html")
            return

        parent_div = borrowable_label.find_element(By.XPATH, './parent::div')
        try:
            borrowable_value_element = parent_div.find_element(By.XPATH, './/span[@class="leading-5"]')
            borrowable_text = borrowable_value_element.text.strip()
            print(f"Найденный текст числа на {page_url}: '{borrowable_text}'")
        except:
            print(
                f"Элемент <span class='leading-5'> не найден на {page_url}. Проверяем весь текст родительского <div>.")
            parent_text = parent_div.text.strip()
            print(f"Текст родительского <div> на {page_url}: '{parent_text}'")
            borrowable_text = parent_text

        match = re.search(r'[\d,\s]+\.?\d*', borrowable_text)
        if match:
            number_str = re.sub(r'[^\d.]', '', match.group())
            try:
                borrowable_value = float(number_str)
                # Проверяем предыдущее значение
                prev_value = previous_values.get(page_url, None)
                should_notify = False

                if borrowable_value > 0:
                    if prev_value is None or prev_value == 0:
                        should_notify = True  # Первая проверка или предыдущее значение 0
                    else:
                        # Проверяем, отличается ли новое значение более чем на 20%
                        percent_change = abs(borrowable_value - prev_value) / prev_value
                        if percent_change > 0.2:
                            should_notify = True

                if should_notify:
                    message = f"по {page_url} Immediate Borrowable {borrowable_value}"
                    print(f"Immediate Borrowable на {page_url}: {borrowable_value} (уведомление отправлено)")
                    send_to_telegram(message)
                else:
                    print(
                        f"Immediate Borrowable на {page_url}: {borrowable_value} (не отправлено, изменение <= 20% или <= 1000)")

                # Обновляем предыдущее значение
                previous_values[page_url] = borrowable_value
            except ValueError:
                print(f"Не удалось преобразовать в число на {page_url}: '{number_str}'")
        else:
            print(f"Не удалось извлечь число из текста на {page_url}: '{borrowable_text}'")
            with open(f'page_source_{page_url.split("/")[-3]}.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print(f"HTML страницы сохранён в page_source_{page_url.split('/')[-3]}.html")

    except Exception as e:
        print(f"Ошибка на {page_url}: {e}")
        with open(f'page_source_{page_url.split("/")[-3]}.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source if 'driver' in locals() else '')



def check_all_pages():
    """Проверка всех страниц."""
    for page_url in PAGE_URLS:
        parse_immediate_borrowable(page_url)


def run_background():
    """Запуск фоновой задачи для периодической проверки."""
    last_check = datetime.now()
    while True:
        check_all_pages()
        last_check = datetime.now()
        time.sleep(CHECK_INTERVAL)
        if (datetime.now() - last_check).total_seconds() > CHECK_INTERVAL * 2:
            print("Обнаружен пропуск проверки, выполняем немедленно.")
            check_all_pages()


if __name__ == "__main__":
    print("Background Worker запущен. Проверка страниц каждые 60 секунд.")
    run_background()