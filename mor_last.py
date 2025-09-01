# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from webdriver_manager.chrome import ChromeDriverManager
# import pandas as pd
# import time
# import re
# import requests
# import json
# import math
#
#
# # --- Функции из первого кода ---
# def convert_to_number(value):
#     """Преобразование строки в число с поддержкой форматов вроде 1.23M USDC, 16.29k WETH."""
#     try:
#         value = value.replace('$', '').replace(',', '').strip()
#         value = value.split('\n')[0]
#         value = re.sub(r'\s*(USDC|WETH|DAI|[a-zA-Z]+)$', '', value).strip()
#         if 'M' in value:
#             return float(value.replace('M', '')) * 1_000_000
#         elif 'k' in value or 'K' in value:
#             return float(value.replace('k', '').replace('K', '')) * 1_000
#         elif '%' in value:
#             return float(value.replace('%', ''))
#         return float(value)
#     except (ValueError, AttributeError) as e:
#         print(f"Ошибка преобразования значения '{value}': {e}")
#         return None
#
#
# def clean_name(name):
#     """Очистка названия организации: убираем .Svg и приводим к единому формату."""
#     name = name.replace('.Svg', '').replace('.svg', '').replace('-', ' ').title()
#     if name.lower() == 'mevcapital':
#         return 'MEV Capital'
#     return name
#
#
# def extract_trusted_by_names(col, driver):
#     """Извлечение названий организаций из ячейки Trusted By."""
#     try:
#         tooltip = (col.get_attribute('title') or
#                    col.get_attribute('data-tooltip') or
#                    col.get_attribute('aria-label') or
#                    col.get_attribute('data-tip') or
#                    col.get_attribute('data-tooltip-content'))
#         if tooltip:
#             names = [clean_name(name.strip()) for name in tooltip.split(',')]
#             print(f"Tooltip найден (атрибут): {names}")
#             return names
#
#         images = col.find_elements(By.TAG_NAME, 'img')
#         names = []
#         for img in images:
#             src = img.get_attribute('src') or ''
#             alt = img.get_attribute('alt') or ''
#             if src and 'morpho.org' in src:
#                 name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
#                 names.append(clean_name(name))
#             elif alt and alt != 'avatar':
#                 names.append(clean_name(alt))
#         if names:
#             print(f"Изображения найдены в ячейке: {names}")
#             return names
#
#         ActionChains(driver).move_to_element(col).perform()
#         time.sleep(1)
#
#         try:
#             tooltip_element = driver.find_element(
#                 By.CSS_SELECTOR,
#                 '[class*="tooltip"], [class*="popover"], [role="tooltip"], [data-tip], [data-tooltip-content], .css-11nlj6z, .eedv4i31'
#             )
#             tooltip_images = tooltip_element.find_elements(By.TAG_NAME, 'img')
#             for img in tooltip_images:
#                 src = img.get_attribute('src') or ''
#                 alt = img.get_attribute('alt') or ''
#                 if src and 'morpho.org' in src:
#                     name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
#                     names.append(clean_name(name))
#                 elif alt and alt != 'avatar':
#                     names.append(clean_name(alt))
#             if names:
#                 print(f"Изображения найдены в tooltip: {names}")
#                 return names
#             text = tooltip_element.text.strip()
#             if text:
#                 names = [clean_name(name.strip()) for name in text.split(',')]
#                 print(f"Tooltip найден (текст): {names}")
#                 return names
#         except:
#             print(f"Tooltip не найден для Trusted By в строке: {col.text}")
#
#         return [col.text.strip()]
#     except Exception as e:
#         print(f"Ошибка при извлечении Trusted By: {e}")
#         return [col.text.strip()]
#
#
# def extract_collateral_links(col):
#     """Извлечение ссылок из столбца Collateral."""
#     try:
#         link_elements = col.find_elements(By.TAG_NAME, 'a')
#         for link in link_elements:
#             href = link.get_attribute('href')
#             if href:
#                 return href
#         href = (col.get_attribute('href') or
#                 col.get_attribute('data-href') or
#                 col.get_attribute('data-url'))
#         if href:
#             return href
#         images = col.find_elements(By.TAG_NAME, 'img')
#         for img in images:
#             parent = img.find_element(By.XPATH, './..')
#             href = parent.get_attribute('href') or parent.get_attribute('data-href')
#             if href:
#                 return href
#         return None
#     except Exception as e:
#         print(f"Ошибка при извлечении ссылки из Collateral: {e}")
#         return None
#
#
# def parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages=38):
#     try:
#         chrome_options = Options()
#         chrome_options.add_argument("--headless")
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#
#         print("Открываем страницу...")
#         driver.get(url)
#
#         print("Ожидаем загрузки таблицы...")
#         table_selector = 'table'
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#         )
#
#         table = driver.find_element(By.CSS_SELECTOR, table_selector)
#         headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
#         print("Заголовки:", headers)
#
#         all_data = []
#
#         for page in range(1, max_pages + 1):
#             print(f"Обработка страницы {page}...")
#
#             try:
#                 WebDriverWait(driver, 20).until(
#                     EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                 )
#                 table = driver.find_element(By.CSS_SELECTOR, table_selector)
#
#                 rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # Пропускаем заголовок
#                 if not rows:
#                     print(f"На странице {page} нет строк данных.")
#                     break
#
#                 for row in rows:
#                     cols = row.find_elements(By.TAG_NAME, 'td')
#                     row_data = []
#                     for i, col in enumerate(cols):
#                         if headers[i] == 'Trusted By':
#                             row_data.append(extract_trusted_by_names(col, driver))
#                         elif headers[i] == 'Collateral':
#                             row_data.append({
#                                 'text': col.text.strip(),
#                                 'link': extract_collateral_links(col)
#                             })
#                         else:
#                             row_data.append(col.text.strip())
#                     all_data.append(row_data)
#                 print(f"Собрано {len(rows)} строк на странице {page}")
#
#                 if page < max_pages:
#                     try:
#                         next_button = WebDriverWait(driver, 10).until(
#                             EC.element_to_be_clickable(
#                                 (By.CSS_SELECTOR,
#                                  'button.css-ktorsf:not([disabled]):not([style*="rotate(180deg)"]), '
#                                  'button:has(svg[icon="ArrowPlain20"]):not([disabled]):not([style*="rotate(180deg)"]), '
#                                  '[class*="pagination"] button:not([style*="rotate(180deg)"])')
#                             )
#                         )
#                         next_button.click()
#                         print("Кликаем по кнопке 'Next'...")
#                         time.sleep(1)
#                         WebDriverWait(driver, 20).until(EC.staleness_of(table))
#                     except Exception as e:
#                         print(f"Кнопка 'Next' не найдена: {e}")
#                         next_url = f"{url}?page={page + 1}"
#                         print(f"Пробуем перейти на страницу {page + 1} через URL: {next_url}")
#                         driver.get(next_url)
#                         try:
#                             WebDriverWait(driver, 20).until(
#                                 EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                             )
#                         except Exception as url_e:
#                             print(f"Ошибка перехода через URL: {url_e}")
#                             break
#             except Exception as page_e:
#                 print(f"Ошибка обработки страницы {page}: {page_e}")
#                 break
#
#         driver.quit()
#
#         if not all_data:
#             print("Нет собранных данных.")
#             return None
#
#         df = pd.DataFrame(all_data, columns=headers)
#         df['text Collateral'] = df['Collateral'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
#         df['link Collateral'] = df['Collateral'].apply(lambda x: x['link'] if isinstance(x, dict) else None)
#         df['Rate'] = df['Rate'].apply(convert_to_number)
#         df['Total Liquidity'] = df['Total Liquidity'].apply(convert_to_number)
#
#         df = df.dropna(subset=['Rate', 'Total Liquidity'])
#
#         filtered_df = df[
#             (df['Rate'] > rate_threshold) &
#             (df['Total Liquidity'] > liquidity_threshold) &
#             (df['Trusted By'].apply(lambda x: any(name in trusted_by_allowed for name in x)))
#         ]
#
#         result_df = filtered_df[['text Collateral', 'link Collateral', 'Rate', 'Trusted By', 'Total Liquidity']]
#         sorted_df = result_df.sort_values(by='Rate', ascending=False)
#
#         if sorted_df.empty:
#             print("Нет строк, удовлетворяющих условиям фильтрации.")
#             return None
#
#         return sorted_df
#
#     except Exception as e:
#         print(f"Ошибка при обработке: {e}")
#         if 'driver' in locals():
#             driver.quit()
#         return None
#
#
# # --- Функции из второго кода ---
# def format_percentage(value):
#     """Конвертирует значение из долей в проценты."""
#     if value is not None:
#         return float(value) * 100
#     return None
#
# def get_morpho_data():
#     """Получает данные через API Morpho."""
#     url = "https://api.morpho.org/graphql"
#     unique_key = "0x64d65c9a2d91c36d56fbc42d69e979335320169b3df63bf92789e2c8883fcc64"
#     chain_id = 1  # Ethereum mainnet
#
#     # GraphQL запрос
#     query = """
#     query {
#         marketByUniqueKey(uniqueKey: "%s", chainId: %d) {
#             state {
#                 borrowAssetsUsd
#                 supplyAssetsUsd
#                 borrowApy
#                 utilization
#             }
#         }
#     }
#     """ % (unique_key, chain_id)
#
#     try:
#         # Формируем тело запроса
#         payload = {"query": query}
#         headers = {
#             'Content-Type': 'application/json',
#             'Accept': 'application/graphql-response+json, application/json',
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         }
#
#         # Отправляем запрос
#         response = requests.post(url, json=payload, headers=headers)
#         response.raise_for_status()  # Проверяем статус ответа
#
#         # Парсим ответ
#         data = response.json()
#
#         # Проверяем наличие ошибок в ответе
#         if 'errors' in data:
#             print("Ошибки в ответе API:")
#             for error in data['errors']:
#                 print(f"- {error.get('message')}")
#                 if 'extensions' in error:
#                     print(f"  Дополнительно: {error['extensions']}")
#             return None, None, None, None
#
#         # Извлекаем данные
#         market_data = data.get('data', {}).get('marketByUniqueKey', {}).get('state', {})
#         total_borrow = market_data.get('borrowAssetsUsd')
#         total_supply = market_data.get('supplyAssetsUsd')
#         borrow_apy = market_data.get('borrowApy')
#         utilization = market_data.get('utilization')
#
#         if any(v is None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#             print("Не удалось найти некоторые данные. Проверьте uniqueKey, chainId или схему API.")
#             return None, None, None, None
#
#         # Конвертируем значения
#         total_borrow = convert_to_number(total_borrow)
#         total_supply = convert_to_number(total_supply)
#         borrow_apy = format_percentage(borrow_apy)  # Преобразуем в проценты
#         utilization = format_percentage(utilization)  # Преобразуем в проценты
#
#         return total_borrow, total_supply, borrow_apy, utilization
#
#     except requests.HTTPError as e:
#         print(f"HTTP ошибка: {e}")
#         if e.response.status_code == 400:
#             try:
#                 error_data = e.response.json()
#                 print("Подробности ошибки:")
#                 for error in error_data.get('errors', []):
#                     print(f"- {error.get('message')}")
#                     if 'extensions' in error:
#                         print(f"  Дополнительно: {error['extensions']}")
#             except ValueError:
#                 print("Не удалось разобрать ответ сервера.")
#         return None, None, None, None
#     except Exception as e:
#         print(f"Произошла ошибка: {e}")
#         return None, None, None, None
#
# # def get_morpho_market_data(market_url, driver):
# #     """Извлекает данные рынка с указанной страницы Morpho."""
# #     try:
# #         driver.get(market_url)
# #         print(f"Открываем рынок: {market_url}")
# #
# #         # Ожидаем загрузки данных на странице
# #         WebDriverWait(driver, 20).until(
# #             EC.presence_of_element_located((By.CSS_SELECTOR, '[class*="market-details"], [class*="stats"]'))
# #         )
# #
# #         # Извлекаем данные (пример селекторов, нужно адаптировать под реальную структуру страницы)
# #         total_borrow = None
# #         total_supply = None
# #         borrow_apy = None
# #         utilization = None
# #
# #         # Попробуем найти элементы с данными
# #         try:
# #             total_borrow_elem = driver.find_element(By.CSS_SELECTOR, '[data-testid="total-borrow"], [class*="total-borrow"]')
# #             total_borrow = convert_to_number(total_borrow_elem.text)
# #         except Exception as e:
# #             print(f"Ошибка при извлечении Total Borrow: {e}")
# #
# #         try:
# #             total_supply_elem = driver.find_element(By.CSS_SELECTOR, '[data-testid="total-supply"], [class*="total-supply"]')
# #             total_supply = convert_to_number(total_supply_elem.text)
# #         except Exception as e:
# #             print(f"Ошибка при извлечении Total Supply: {e}")
# #
# #         try:
# #             borrow_apy_elem = driver.find_element(By.CSS_SELECTOR, '[data-testid="borrow-apy"], [class*="borrow-apy"]')
# #             borrow_apy = convert_to_number(borrow_apy_elem.text)
# #         except Exception as e:
# #             print(f"Ошибка при извлечении Borrow APY: {e}")
# #
# #         try:
# #             utilization_elem = driver.find_element(By.CSS_SELECTOR, '[data-testid="utilization"], [class*="utilization"]')
# #             utilization = convert_to_number(utilization_elem.text)
# #         except Exception as e:
# #             print(f"Ошибка при извлечении Utilization: {e}")
# #
# #         return total_borrow, total_supply, borrow_apy, utilization
# #     except Exception as e:
# #         print(f"Ошибка при обработке страницы рынка {market_url}: {e}")
# #         return None, None, None, None
#
#
# def calculate_new_rates(total_borrow, total_supply, borrow_apy, utilization, deposit_amount):
#     """Рассчитывает новую Borrow Rate, Supply Rate, Instantaneous Supply Rate и новую утилизацию."""
#     # Константы
#     u_target = 0.9
#     k_d = 4
#     k_p = 50 / 31_536_000  # ADJUSTMENT_SPEED
#     seconds_per_year = 31_536_000
#     fee = 0  # Комиссия протокола
#     time_since_last_interaction = 1000  # Время с последней транзакции (секунды)
#
#     # 1. Перевод Borrow APY в мгновенную Borrow Rate (в секунду)
#     borrow_apy_decimal = borrow_apy / 100
#     r_prev = math.log(1 + borrow_apy_decimal) / seconds_per_year
#
#     # 2. Проверка текущей утилизации
#     u_current = utilization / 100  # Утилизация в долях
#
#     # 3. Расчет новой утилизации после вложения
#     new_total_supply = total_supply + deposit_amount
#     u_new = total_borrow / new_total_supply
#
#     # 4. Расчет ошибки (e(u)) для текущей и новой утилизации
#     if u_current > u_target:
#         e_u_current = (u_current - u_target) / (1 - u_target)
#     else:
#         e_u_current = (u_current - u_target) / u_target
#
#     if u_new > u_target:
#         e_u_new = (u_new - u_target) / (1 - u_target)
#     else:
#         e_u_new = (u_new - u_target) / u_target
#
#     # 5. Расчет curve(u) для текущей и новой утилизации
#     if u_current > u_target:
#         curve_u_current = (k_d - 1) * e_u_current + 1
#     else:
#         curve_u_current = (1 - 1/k_d) * e_u_current + 1
#
#     if u_new > u_target:
#         curve_u_new = (k_d - 1) * e_u_new + 1
#     else:
#         curve_u_new = (1 - 1/k_d) * e_u_new + 1
#
#     # 6. Расчет r_T(t)
#     r_T_last = r_prev / curve_u_current
#     speed_t = math.exp(k_p * e_u_current * time_since_last_interaction)
#     r_T_t = r_T_last * speed_t
#
#     # 7. Новая Borrow Rate
#     r_new = r_T_t * curve_u_new
#     borrow_apy_new = (math.exp(r_new * seconds_per_year) - 1) * 100  # В процентах
#
#     # 8. Новая Supply Rate
#     supply_apy_new = borrow_apy_new / 100 * u_new * (1 - fee) * 100  # В процентах
#
#     # 9. Instantaneous Supply Rate
#     r_supply_t = r_new * u_new * (1 - fee)
#     instantaneous_supply_rate_annual = r_supply_t * seconds_per_year * 100  # В процентах
#
#     # 10. Новая утилизация в процентах
#     u_new_percent = u_new * 100
#
#     return borrow_apy_new, supply_apy_new, instantaneous_supply_rate_annual, u_new_percent
#
#
# # --- Основная функция для объединения ---
# def main():
#     # Параметры для фильтрации
#     url = "https://app.morpho.org/ethereum/borrow"
#     rate_threshold = 11.0  # Порог для Rate (в процентах)
#     liquidity_threshold = 1_0_000  # Порог для Total Liquidity
#     trusted_by_allowed = ['Steakhouse', 'Gauntlet']  # Фильтрация по Trusted By
#     max_pages = 3
#     deposit_amount = 50_000  # Сумма вложения в USD
#
#     # Выполняем первый код для получения отфильтрованных данных
#     filtered_data = parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages)
#
#     if filtered_data is None or filtered_data.empty:
#         print("Не удалось получить данные или данные не соответствуют фильтрам.")
#         return
#
#     # Инициализация драйвера для обработки страниц рынков
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#
#     try:
#         # Создаем список для хранения результатов
#         results = []
#
#         # Проходим по каждому рынку из отфильтрованных данных
#         for index, row in filtered_data.iterrows():
#             market_name = row['text Collateral']
#             market_url = row['link Collateral']
#             print(f"\nОбработка рынка: {market_name} ({market_url})")
#
#             # Извлекаем данные рынка
#             total_borrow, total_supply, borrow_apy, utilization = get_morpho_data(market_url, driver)
#
#             if all(v is not None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#                 print(f"Данные для {market_name}:")
#                 print(f"Total Borrow (USD): {total_borrow:,.2f}")
#                 print(f"Total Supply (USD): {total_supply:,.2f}")
#                 print(f"Borrow APY: {borrow_apy:.2f}%")
#                 print(f"Utilization: {utilization:.2f}%")
#
#                 # Рассчитываем новые параметры
#                 borrow_apy_new, supply_apy_new, instantaneous_supply_rate_annual, u_new_percent = calculate_new_rates(
#                     total_borrow, total_supply, borrow_apy, utilization, deposit_amount
#                 )
#
#                 # Добавляем результаты в список
#                 results.append({
#                     'Market': market_name,
#                     'URL': market_url,
#                     'Original Supply APY': row['Rate'],
#                     'New Supply APY': supply_apy_new,
#                     'New Utilization': u_new_percent,
#                     'Total Borrow (USD)': total_borrow,
#                     'Total Supply (USD)': total_supply,
#                     'Original Borrow APY': borrow_apy,
#                     'Original Utilization': utilization
#                 })
#
#                 print(f"\nПосле вложения {deposit_amount:,.2f} USD в {market_name}:")
#                 print(f"Новая утилизация: {u_new_percent:.2f}%")
#                 print(f"Новая Supply APY: {supply_apy_new:.2f}%")
#                 print(f"Instantaneous Supply Rate (годовая): {instantaneous_supply_rate_annual:.2f}%")
#             else:
#                 print(f"Не удалось получить данные для рынка: {market_name}")
#
#         # Создаем DataFrame с результатами
#         if results:
#             results_df = pd.DataFrame(results)
#             print("\nИтоговые результаты:")
#             print(results_df[['Market', 'New Supply APY', 'New Utilization']].to_string(index=False))
#
#             # Сохраняем результаты в CSV
#             results_df.to_csv('morpho_market_analysis.csv', index=False)
#             print("\nРезультаты сохранены в 'morpho_market_analysis.csv'.")
#
#     finally:
#         driver.quit()
#
#
# if __name__ == "__main__":
#     main()

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from webdriver_manager.chrome import ChromeDriverManager
# import pandas as pd
# import time
# import re
# import requests
# import json
# import math
#
#
# # --- Функции из первого кода ---
# def convert_to_number(value):
#     """Преобразование строки в число с поддержкой форматов вроде 1.23M USDC, 16.29k WETH."""
#     try:
#         value = str(value).replace('$', '').replace(',', '').strip()
#         value = value.split('\n')[0]
#         value = re.sub(r'\s*(USDC|WETH|DAI|[a-zA-Z]+)$', '', value).strip()
#         if 'M' in value:
#             return float(value.replace('M', '')) * 1_000_000
#         elif 'k' in value or 'K' in value:
#             return float(value.replace('k', '').replace('K', '')) * 1_000
#         elif '%' in value:
#             return float(value.replace('%', ''))
#         return float(value)
#     except (ValueError, AttributeError) as e:
#         print(f"Ошибка преобразования значения '{value}': {e}")
#         return None
#
#
# def clean_name(name):
#     """Очистка названия организации: убираем .Svg и приводим к единому формату."""
#     name = name.replace('.Svg', '').replace('.svg', '').replace('-', ' ').title()
#     if name.lower() == 'mevcapital':
#         return 'MEV Capital'
#     return name
#
#
# def extract_trusted_by_names(col, driver):
#     """Извлечение названий организаций из ячейки Trusted By."""
#     try:
#         tooltip = (col.get_attribute('title') or
#                    col.get_attribute('data-tooltip') or
#                    col.get_attribute('aria-label') or
#                    col.get_attribute('data-tip') or
#                    col.get_attribute('data-tooltip-content'))
#         if tooltip:
#             names = [clean_name(name.strip()) for name in tooltip.split(',')]
#             print(f"Tooltip найден (атрибут): {names}")
#             return names
#
#         images = col.find_elements(By.TAG_NAME, 'img')
#         names = []
#         for img in images:
#             src = img.get_attribute('src') or ''
#             alt = img.get_attribute('alt') or ''
#             if src and 'morpho.org' in src:
#                 name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
#                 names.append(clean_name(name))
#             elif alt and alt != 'avatar':
#                 names.append(clean_name(alt))
#         if names:
#             print(f"Изображения найдены в ячейке: {names}")
#             return names
#
#         ActionChains(driver).move_to_element(col).perform()
#         time.sleep(1)
#
#         try:
#             tooltip_element = driver.find_element(
#                 By.CSS_SELECTOR,
#                 '[class*="tooltip"], [class*="popover"], [role="tooltip"], [data-tip], [data-tooltip-content], .css-11nlj6z, .eedv4i31'
#             )
#             tooltip_images = tooltip_element.find_elements(By.TAG_NAME, 'img')
#             for img in tooltip_images:
#                 src = img.get_attribute('src') or ''
#                 alt = img.get_attribute('alt') or ''
#                 if src and 'morpho.org' in src:
#                     name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
#                     names.append(clean_name(name))
#                 elif alt and alt != 'avatar':
#                     names.append(clean_name(alt))
#             if names:
#                 print(f"Изображения найдены в tooltip: {names}")
#                 return names
#             text = tooltip_element.text.strip()
#             if text:
#                 names = [clean_name(name.strip()) for name in text.split(',')]
#                 print(f"Tooltip найден (текст): {names}")
#                 return names
#         except:
#             print(f"Tooltip не найден для Trusted By в строке: {col.text}")
#
#         return [col.text.strip()]
#     except Exception as e:
#         print(f"Ошибка при извлечении Trusted By: {e}")
#         return [col.text.strip()]
#
#
# def extract_collateral_links(col):
#     """Извлечение ссылок из столбца Collateral."""
#     try:
#         link_elements = col.find_elements(By.TAG_NAME, 'a')
#         for link in link_elements:
#             href = link.get_attribute('href')
#             if href:
#                 return href
#         href = (col.get_attribute('href') or
#                 col.get_attribute('data-href') or
#                 col.get_attribute('data-url'))
#         if href:
#             return href
#         images = col.find_elements(By.TAG_NAME, 'img')
#         for img in images:
#             parent = img.find_element(By.XPATH, './..')
#             href = parent.get_attribute('href') or parent.get_attribute('data-href')
#             if href:
#                 return href
#         return None
#     except Exception as e:
#         print(f"Ошибка при извлечении ссылки из Collateral: {e}")
#         return None
#
#
# def parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages=38):
#     try:
#         chrome_options = Options()
#         chrome_options.add_argument("--headless")
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#
#         print("Открываем страницу...")
#         driver.get(url)
#
#         print("Ожидаем загрузки таблицы...")
#         table_selector = 'table'
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#         )
#
#         table = driver.find_element(By.CSS_SELECTOR, table_selector)
#         headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
#         print("Заголовки:", headers)
#
#         all_data = []
#
#         for page in range(1, max_pages + 1):
#             print(f"Обработка страницы {page}...")
#
#             try:
#                 WebDriverWait(driver, 20).until(
#                     EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                 )
#                 table = driver.find_element(By.CSS_SELECTOR, table_selector)
#
#                 rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # Пропускаем заголовок
#                 if not rows:
#                     print(f"На странице {page} нет строк данных.")
#                     break
#
#                 for row in rows:
#                     cols = row.find_elements(By.TAG_NAME, 'td')
#                     row_data = []
#                     for i, col in enumerate(cols):
#                         if headers[i] == 'Trusted By':
#                             row_data.append(extract_trusted_by_names(col, driver))
#                         elif headers[i] == 'Collateral':
#                             row_data.append({
#                                 'text': col.text.strip(),
#                                 'link': extract_collateral_links(col)
#                             })
#                         else:
#                             row_data.append(col.text.strip())
#                     all_data.append(row_data)
#                 print(f"Собрано {len(rows)} строк на странице {page}")
#
#                 if page < max_pages:
#                     try:
#                         next_button = WebDriverWait(driver, 10).until(
#                             EC.element_to_be_clickable(
#                                 (By.CSS_SELECTOR,
#                                  'button.css-ktorsf:not([disabled]):not([style*="rotate(180deg)"]), '
#                                  'button:has(svg[icon="ArrowPlain20"]):not([disabled]):not([style*="rotate(180deg)"]), '
#                                  '[class*="pagination"] button:not([style*="rotate(180deg)"])')
#                             )
#                         )
#                         next_button.click()
#                         print("Кликаем по кнопке 'Next'...")
#                         time.sleep(1)
#                         WebDriverWait(driver, 20).until(EC.staleness_of(table))
#                     except Exception as e:
#                         print(f"Кнопка 'Next' не найдена: {e}")
#                         next_url = f"{url}?page={page + 1}"
#                         print(f"Пробуем перейти на страницу {page + 1} через URL: {next_url}")
#                         driver.get(next_url)
#                         try:
#                             WebDriverWait(driver, 20).until(
#                                 EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                             )
#                         except Exception as url_e:
#                             print(f"Ошибка перехода через URL: {url_e}")
#                             break
#             except Exception as page_e:
#                 print(f"Ошибка обработки страницы {page}: {page_e}")
#                 break
#
#         driver.quit()
#
#         if not all_data:
#             print("Нет собранных данных.")
#             return None
#
#         df = pd.DataFrame(all_data, columns=headers)
#         df['text Collateral'] = df['Collateral'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
#         df['link Collateral'] = df['Collateral'].apply(lambda x: x['link'] if isinstance(x, dict) else None)
#         df['Rate'] = df['Rate'].apply(convert_to_number)
#         df['Total Liquidity'] = df['Total Liquidity'].apply(convert_to_number)
#
#         df = df.dropna(subset=['Rate', 'Total Liquidity'])
#
#         filtered_df = df[
#             (df['Rate'] > rate_threshold) &
#             (df['Total Liquidity'] > liquidity_threshold) &
#             (df['Trusted By'].apply(lambda x: any(name in trusted_by_allowed for name in x)))
#         ]
#
#         result_df = filtered_df[['text Collateral', 'link Collateral', 'Rate', 'Trusted By', 'Total Liquidity']]
#         sorted_df = result_df.sort_values(by='Rate', ascending=False)
#
#         if sorted_df.empty:
#             print("Нет строк, удовлетворяющих условиям фильтрации.")
#             return None
#
#         return sorted_df
#
#     except Exception as e:
#         print(f"Ошибка при обработке: {e}")
#         if 'driver' in locals():
#             driver.quit()
#         return None
#
#
# # --- Функции для работы с API ---
# def format_percentage(value):
#     """Конвертирует значение из долей в проценты."""
#     if value is not None:
#         return float(value) * 100
#     return None
#
# def extract_unique_key(market_url, driver):
#     """Извлекает unique_key из URL или страницы рынка."""
#     try:
#         # Проверяем URL на наличие unique_key
#         match = re.search(r'market/([^/]+)/?', market_url)
#         if match:
#             unique_key = match.group(1)
#             print(f"Извлечён unique_key из URL: {unique_key}")
#             return unique_key
#
#         # Если не удалось извлечь из URL, открываем страницу
#         driver.get(market_url)
#         print(f"Открываем страницу рынка для извлечения unique_key: {market_url}")
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.TAG_NAME, 'body'))
#         )
#
#         # Проверяем скрипты на странице
#         scripts = driver.find_elements(By.TAG_NAME, 'script')
#         for script in scripts:
#             script_content = script.get_attribute('innerHTML')
#             if script_content and 'uniqueKey' in script_content:
#                 match = re.search(r'"uniqueKey"\s*:\s*"([^"]+)"', script_content)
#                 if match:
#                     unique_key = match.group(1)
#                     print(f"Извлечён unique_key из скрипта: {unique_key}")
#                     return unique_key
#         print(f"Не удалось извлечь unique_key для {market_url}")
#         return None
#     except Exception as e:
#         print(f"Ошибка при извлечении unique_key для {market_url}: {e}")
#         return None
#
#
#
# def get_morpho_data(unique_key, chain_id=1):
#     """Получает данные через API Morpho для указанного unique_key."""
#     if not unique_key:
#         print("unique_key не предоставлен.")
#         return None, None, None, None
#
#     url = "https://api.morpho.org/graphql"
#
#     # GraphQL запрос
#     query = """
#     query {
#         marketByUniqueKey(uniqueKey: "%s", chainId: %d) {
#             state {
#                 borrowAssetsUsd
#                 supplyAssetsUsd
#                 borrowApy
#                 utilization
#             }
#         }
#     }
#     """ % (unique_key, chain_id)
#
#     try:
#         # Формируем тело запроса
#         payload = {"query": query}
#         headers = {
#             'Content-Type': 'application/json',
#             'Accept': 'application/graphql-response+json, application/json',
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         }
#
#         # Отправляем запрос
#         response = requests.post(url, json=payload, headers=headers)
#         response.raise_for_status()  # Проверяем статус ответа
#
#         # Парсим ответ
#         data = response.json()
#
#         # Проверяем наличие ошибок в ответе
#         if 'errors' in data:
#             print("Ошибки в ответе API:")
#             for error in data['errors']:
#                 print(f"- {error.get('message')}")
#                 if 'extensions' in error:
#                     print(f"  Дополнительно: {error['extensions']}")
#             return None, None, None, None
#
#         # Извлекаем данные
#         market_data = data.get('data', {}).get('marketByUniqueKey', {})
#         if not market_data:
#             print(f"Рынок с unique_key {unique_key} не найден.")
#             return None, None, None, None
#
#         state = market_data.get('state', {})
#         total_borrow = state.get('borrowAssetsUsd')
#         total_supply = state.get('supplyAssetsUsd')
#         borrow_apy = state.get('borrowApy')
#         utilization = state.get('utilization')
#
#         if any(v is None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#             print(f"Не удалось найти некоторые данные для unique_key {unique_key}.")
#             return None, None, None, None
#
#         # Конвертируем значения
#         total_borrow = convert_to_number(total_borrow)
#         total_supply = convert_to_number(total_supply)
#         borrow_apy = format_percentage(borrow_apy)  # Преобразуем в проценты
#         utilization = format_percentage(utilization)  # Преобразуем в проценты
#
#         return total_borrow, total_supply, borrow_apy, utilization
#
#     except requests.HTTPError as e:
#         print(f"HTTP ошибка: {e}")
#         if e.response.status_code == 400:
#             try:
#                 error_data = e.response.json()
#                 print("Подробности ошибки:")
#                 for error in error_data.get('errors', []):
#                     print(f"- {error.get('message')}")
#                     if 'extensions' in error:
#                         print(f"  Дополнительно: {error['extensions']}")
#             except ValueError:
#                 print("Не удалось разобрать ответ сервера.")
#         return None, None, None, None
#     except Exception as e:
#         print(f"Произошла ошибка: {e}")
#         return None, None, None, None
#
#
# def calculate_new_rates(total_borrow, total_supply, borrow_apy, utilization, deposit_amount):
#     """Рассчитывает новую Borrow Rate, Supply Rate, Instantaneous Supply Rate и новую утилизацию."""
#     # Константы
#     u_target = 0.9
#     k_d = 4
#     k_p = 50 / 31_536_000  # ADJUSTMENT_SPEED
#     seconds_per_year = 31_536_000
#     fee = 0  # Комиссия протокола
#     time_since_last_interaction = 1000  # Время с последней транзакции (секунды)
#
#     # 1. Перевод Borrow APY в мгновенную Borrow Rate (в секунду)
#     borrow_apy_decimal = borrow_apy / 100
#     r_prev = math.log(1 + borrow_apy_decimal) / seconds_per_year
#
#     # 2. Проверка текущей утилизации
#     u_current = utilization / 100  # Утилизация в долях
#
#     # 3. Расчет новой утилизации после вложения
#     new_total_supply = total_supply + deposit_amount
#     u_new = total_borrow / new_total_supply
#
#     # 4. Расчет ошибки (e(u)) для текущей и новой утилизации
#     if u_current > u_target:
#         e_u_current = (u_current - u_target) / (1 - u_target)
#     else:
#         e_u_current = (u_current - u_target) / u_target
#
#     if u_new > u_target:
#         e_u_new = (u_new - u_target) / (1 - u_target)
#     else:
#         e_u_new = (u_new - u_target) / u_target
#
#     # 5. Расчет curve(u) для текущей и новой утилизации
#     if u_current > u_target:
#         curve_u_current = (k_d - 1) * e_u_current + 1
#     else:
#         curve_u_current = (1 - 1/k_d) * e_u_current + 1
#
#     if u_new > u_target:
#         curve_u_new = (k_d - 1) * e_u_new + 1
#     else:
#         curve_u_new = (1 - 1/k_d) * e_u_new + 1
#
#     # 6. Расчет r_T(t)
#     r_T_last = r_prev / curve_u_current
#     speed_t = math.exp(k_p * e_u_current * time_since_last_interaction)
#     r_T_t = r_T_last * speed_t
#
#     # 7. Новая Borrow Rate
#     r_new = r_T_t * curve_u_new
#     borrow_apy_new = (math.exp(r_new * seconds_per_year) - 1) * 100  # В процентах
#
#     # 8. Новая Supply Rate
#     supply_apy_new = borrow_apy_new / 100 * u_new * (1 - fee) * 100  # В процентах
#
#     # 9. Instantaneous Supply Rate
#     r_supply_t = r_new * u_new * (1 - fee)
#     instantaneous_supply_rate_annual = r_supply_t * seconds_per_year * 100  # В процентах
#
#     # 10. Новая утилизация в процентах
#     u_new_percent = u_new * 100
#
#     return borrow_apy_new, supply_apy_new, instantaneous_supply_rate_annual, u_new_percent
#
# def main():
#     # Параметры для фильтрации
#     url = "https://app.morpho.org/ethereum/borrow"
#     rate_threshold = 11.0  # Порог для Rate (в процентах)
#     liquidity_threshold = 1_000_000  # Порог для Total Liquidity
#     trusted_by_allowed = ['Steakhouse', 'Gauntlet']  # Фильтрация по Trusted By
#     max_pages = 3
#     deposit_amount = 50_000  # Сумма вложения в USD
#     chain_id = 1  # Ethereum mainnet
#
#     # Выполняем парсинг и фильтрацию данных
#     filtered_data = parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages)
#
#     if filtered_data is None or filtered_data.empty:
#         print("Не удалось получить данные или данные не соответствуют фильтрам.")
#         return
#
#     # Инициализация драйвера
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#
#     try:
#         # Создаем список для хранения результатов
#         results = []
#
#         # Проходим по каждому рынку из отфильтрованных данных
#         for index, row in filtered_data.iterrows():
#             market_name = row['text Collateral']
#             market_url = row['link Collateral']
#             trusted_by = row['Trusted By']  # Извлекаем Trusted By
#             print(f"\nОбработка рынка: {market_name} ({market_url})")
#
#             # Извлекаем unique_key
#             unique_key = extract_unique_key(market_url, driver)
#             if not unique_key:
#                 print(f"Пропуск рынка {market_name}: не удалось извлечь unique_key.")
#                 continue
#
#             # Извлекаем данные рынка через API
#             total_borrow, total_supply, borrow_apy, utilization = get_morpho_data(unique_key, chain_id)
#
#             if all(v is not None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#                 print(f"Данные для {market_name}:")
#                 print(f"Total Borrow (USD): {total_borrow:,.2f}")
#                 print(f"Total Supply (USD): {total_supply:,.2f}")
#                 print(f"Borrow APY: {borrow_apy:.2f}%")
#                 print(f"Utilization: {utilization:.2f}%")
#
#                 # Рассчитываем новые параметры
#                 borrow_apy_new, supply_apy_new, instantaneous_supply_rate_annual, u_new_percent = calculate_new_rates(
#                     total_borrow, total_supply, borrow_apy, utilization, deposit_amount
#                 )
#
#                 # Добавляем результаты в список
#                 results.append({
#                     'Market': market_name,
#                     'URL': market_url,
#                     'Trusted By': ', '.join(trusted_by),  # Преобразуем список в строку
#                     'Original Supply APY': row['Rate'],
#                     'New Supply APY': supply_apy_new,
#                     'New Utilization': u_new_percent,
#                     'Total Borrow (USD)': total_borrow,
#                     'Total Supply (USD)': total_supply,
#                     'Original Borrow APY': borrow_apy,
#                     'Original Utilization': utilization
#                 })
#
#                 print(f"\nПосле вложения {deposit_amount:,.2f} USD в {market_name}:")
#                 print(f"Новая утилизация: {u_new_percent:.2f}%")
#                 print(f"Новая Supply APY: {supply_apy_new:.2f}%")
#                 print(f"Instantaneous Supply Rate (годовая): {instantaneous_supply_rate_annual:.2f}%")
#             else:
#                 print(f"Не удалось получить данные для рынка: {market_name}")
#
#         # Создаем DataFrame с результатами
#         if results:
#             results_df = pd.DataFrame(results)
#             print("\nИтоговые результаты:")
#             print(results_df[['Market', 'New Supply APY', 'New Utilization', 'Trusted By', 'URL']].to_string(index=False))
#
#             # Сохраняем результаты в CSV
#             results_df.to_csv('morpho_market_analysis.csv', index=False)
#             print("\nРезультаты сохранены в 'morpho_market_analysis.csv'.")
#
#     finally:
#         driver.quit()
#
#
#
# if __name__ == "__main__":
#     main()

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from webdriver_manager.chrome import ChromeDriverManager
# import pandas as pd
# import time
# import re
# import requests
# import json
# import math
# import logging
# from typing import Dict, List, Optional, Tuple
# from itertools import combinations
#
# # Настройка логирования
# logging.basicConfig(
#     filename='morpho_scraper.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
#
# # Функции из предоставленного кода
# def convert_to_number(value):
#     """Преобразование строки или числа в число с поддержкой форматов вроде 1.23M USDC, 16.29k WETH."""
#     try:
#         # Если значение уже число (int или float), возвращаем его
#         if isinstance(value, (int, float)):
#             return float(value)
#
#         # Если значение строка, обрабатываем формат
#         value = value.replace('$', '').replace(',', '').strip()
#         value = value.split('\n')[0]
#         value = re.sub(r'\s*(USDC|WETH|DAI|rUSD|USDT|wUSDL|USDS|[a-zA-Z]+)$', '', value).strip()
#         if 'M' in value:
#             return float(value.replace('M', '')) * 1_000_000
#         elif 'k' in value or 'K' in value:
#             return float(value.replace('k', '').replace('K', '')) * 1_000
#         elif '%' in value:
#             return float(value.replace('%', ''))
#         return float(value)
#     except (ValueError, AttributeError) as e:
#         print(f"Ошибка преобразования значения '{value}': {e}")
#         logger.error(f"Ошибка преобразования значения '{value}': {e}")
#         return None
#
#
# def clean_name(name):
#     """Очистка названия организации: убираем .Svg и приводим к единому формату."""
#     name = name.replace('.Svg', '').replace('.svg', '').replace('-', ' ').title()
#     if name.lower() == 'mevcapital':
#         return 'MEV Capital'
#     return name
#
#
# def extract_trusted_by_names(col, driver):
#     """Извлечение названий организаций из ячейки Trusted By."""
#     try:
#         tooltip = (col.get_attribute('title') or
#                    col.get_attribute('data-tooltip') or
#                    col.get_attribute('aria-label') or
#                    col.get_attribute('data-tip') or
#                    col.get_attribute('data-tooltip-content'))
#         if tooltip:
#             names = [clean_name(name.strip()) for name in tooltip.split(',')]
#             print(f"Tooltip найден (атрибут): {names}")
#             return names
#
#         images = col.find_elements(By.TAG_NAME, 'img')
#         names = []
#         for img in images:
#             src = img.get_attribute('src') or ''
#             alt = img.get_attribute('alt') or ''
#             if src and 'morpho.org' in src:
#                 name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
#                 names.append(clean_name(name))
#             elif alt and alt != 'avatar':
#                 names.append(clean_name(alt))
#         if names:
#             print(f"Изображения найдены в ячейке: {names}")
#             return names
#
#         ActionChains(driver).move_to_element(col).perform()
#         time.sleep(1)
#
#         try:
#             tooltip_element = driver.find_element(
#                 By.CSS_SELECTOR,
#                 '[class*="tooltip"], [class*="popover"], [role="tooltip"], [data-tip], [data-tooltip-content], .css-11nlj6z, .eedv4i31'
#             )
#             tooltip_images = tooltip_element.find_elements(By.TAG_NAME, 'img')
#             for img in tooltip_images:
#                 src = img.get_attribute('src') or ''
#                 alt = img.get_attribute('alt') or ''
#                 if src and 'morpho.org' in src:
#                     name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
#                     names.append(clean_name(name))
#                 elif alt and alt != 'avatar':
#                     names.append(clean_name(alt))
#             if names:
#                 print(f"Изображения найдены в tooltip: {names}")
#                 return names
#             text = tooltip_element.text.strip()
#             if text:
#                 names = [clean_name(name.strip()) for name in text.split(',')]
#                 print(f"Tooltip найден (текст): {names}")
#                 return names
#         except:
#             print(f"Tooltip не найден для Trusted By в строке: {col.text}")
#             logger.debug(f"Tooltip не найден для Trusted By в строке: {col.text}")
#         return [col.text.strip()]
#     except Exception as e:
#         print(f"Ошибка при извлечении Trusted By: {e}")
#         logger.error(f"Ошибка при извлечении Trusted By: {e}")
#         return [col.text.strip()]
#
#
# def extract_collateral_links(col):
#     """Извлечение ссылок из столбца Collateral."""
#     try:
#         link_elements = col.find_elements(By.TAG_NAME, 'a')
#         for link in link_elements:
#             href = link.get_attribute('href')
#             if href:
#                 return href
#         href = (col.get_attribute('href') or
#                 col.get_attribute('data-href') or
#                 col.get_attribute('data-url'))
#         if href:
#             return href
#         images = col.find_elements(By.TAG_NAME, 'img')
#         for img in images:
#             parent = img.find_element(By.XPATH, './..')
#             href = parent.get_attribute('href') or parent.get_attribute('data-href')
#             if href:
#                 return href
#         return None
#     except Exception as e:
#         print(f"Ошибка при извлечении ссылки из Collateral: {e}")
#         logger.error(f"Ошибка при извлечении ссылки из Collateral: {e}")
#         return None
#
#
# def parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages=38):
#     try:
#         chrome_options = Options()
#         chrome_options.add_argument("--headless")
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         chrome_options.add_argument("--disable-gpu")
#         chrome_options.add_argument("--window-size=1920,1080")
#         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#         print("WebDriver успешно инициализирован.")
#
#         print(f"Открываем страницу: {url}")
#         driver.get(url)
#
#         print("Ожидаем загрузки таблицы...")
#         table_selector = 'table'
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#         )
#         print("Таблица найдена.")
#
#         table = driver.find_element(By.CSS_SELECTOR, table_selector)
#         headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
#         print("Заголовки:", headers)
#
#         all_data = []
#
#         for page in range(1, max_pages + 1):
#             print(f"Обработка страницы {page}...")
#             try:
#                 WebDriverWait(driver, 20).until(
#                     EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                 )
#                 table = driver.find_element(By.CSS_SELECTOR, table_selector)
#
#                 rows = table.find_elements(By.TAG_NAME, 'tr')[1:]
#                 if not rows:
#                     print(f"На странице {page} нет строк данных.")
#                     logger.info(f"На странице {page} нет строк данных.")
#                     break
#
#                 for row in rows:
#                     cols = row.find_elements(By.TAG_NAME, 'td')
#                     row_data = []
#                     for i, col in enumerate(cols):
#                         if headers[i] == 'Trusted By':
#                             row_data.append(extract_trusted_by_names(col, driver))
#                         elif headers[i] == 'Collateral':
#                             row_data.append({
#                                 'text': col.text.strip(),
#                                 'link': extract_collateral_links(col)
#                             })
#                         else:
#                             row_data.append(col.text.strip())
#                     all_data.append(row_data)
#                 print(f"Собрано {len(rows)} строк на странице {page}")
#                 logger.info(f"Собрано {len(rows)} строк на странице {page}")
#
#                 if page < max_pages:
#                     try:
#                         next_button = WebDriverWait(driver, 10).until(
#                             EC.element_to_be_clickable(
#                                 (By.CSS_SELECTOR,
#                                  'button.css-ktorsf:not([disabled]):not([style*="rotate(180deg)"]), '
#                                  'button:has(svg[icon="ArrowPlain20"]):not([disabled]):not([style*="rotate(180deg)"]), '
#                                  '[class*="pagination"] button:not([style*="rotate(180deg)"])')
#                             )
#                         )
#                         print("Кнопка 'Next' найдена.")
#                         next_button.click()
#                         print("Кликаем по кнопке 'Next'...")
#                         time.sleep(1)
#                         WebDriverWait(driver, 20).until(EC.staleness_of(table))
#                     except Exception as e:
#                         print(f"Кнопка 'Next' не найдена: {e}")
#                         logger.info(f"Кнопка 'Next' не найдена: {e}")
#                         next_url = f"{url}?page={page + 1}"
#                         print(f"Пробуем перейти на страницу {page + 1} через URL: {next_url}")
#                         driver.get(next_url)
#                         try:
#                             WebDriverWait(driver, 20).until(
#                                 EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                             )
#                         except Exception as url_e:
#                             print(f"Ошибка перехода через URL: {url_e}")
#                             logger.info(f"Ошибка перехода через URL: {url_e}")
#                             break
#             except Exception as page_e:
#                 print(f"Ошибка обработки страницы {page}: {page_e}")
#                 logger.error(f"Ошибка обработки страницы {page}: {page_e}")
#                 break
#
#         driver.quit()
#         print("WebDriver закрыт.")
#
#         if not all_data:
#             print("Нет собранных данных.")
#             logger.warning("Нет собранных данных.")
#             return None
#
#         df = pd.DataFrame(all_data, columns=headers)
#         print("Исходный DataFrame:\n", df.to_string())
#
#         df['text Collateral'] = df['Collateral'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
#         df['link Collateral'] = df['Collateral'].apply(lambda x: x['link'] if isinstance(x, dict) else None)
#         df['Rate'] = df['Rate'].apply(convert_to_number)
#         df['Total Liquidity'] = df['Total Liquidity'].apply(convert_to_number)
#
#         print("DataFrame после преобразования:\n", df.to_string())
#
#         df = df.dropna(subset=['Rate', 'Total Liquidity'])
#         print("DataFrame после удаления пустых значений:\n", df.to_string())
#
#         filtered_df = df[
#             (df['Rate'] > rate_threshold) &
#             (df['Total Liquidity'] > liquidity_threshold) &
#             (df['Trusted By'].apply(lambda x: not trusted_by_allowed or any(name in trusted_by_allowed for name in x)))
#             ]
#
#         print("Отфильтрованный DataFrame:\n", filtered_df.to_string())
#         if filtered_df.empty:
#             print("Нет строк, удовлетворяющих условиям фильтрации.")
#             logger.warning("Нет строк, удовлетворяющих условиям фильтрации.")
#             return None
#
#         result_df = filtered_df[['text Collateral', 'link Collateral', 'Rate', 'Trusted By', 'Total Liquidity']]
#         sorted_df = result_df.sort_values(by='Rate', ascending=False)
#
#         return sorted_df
#
#     except Exception as e:
#         print(f"Критическая ошибка в parse_and_filter_data: {e}")
#         logger.error(f"Критическая ошибка в parse_and_filter_data: {e}")
#         if 'driver' in locals():
#             driver.quit()
#         return None
#
#
# def format_percentage(value):
#     """Конвертирует значение из долей в проценты."""
#     if value is not None:
#         return float(value) * 100
#     return None
#
#
# def extract_unique_key(market_url, driver):
#     """Извлекает unique_key из URL или страницы рынка."""
#     try:
#         match = re.search(r'market/([^/]+)/?', market_url)
#         if match:
#             unique_key = match.group(1)
#             logger.debug(f"Извлечён unique_key из URL: {unique_key}")
#             return unique_key
#
#         driver.get(market_url)
#         logger.debug(f"Открываем страницу рынка для извлечения unique_key: {market_url}")
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.TAG_NAME, 'body'))
#         )
#
#         scripts = driver.find_elements(By.TAG_NAME, 'script')
#         for script in scripts:
#             script_content = script.get_attribute('innerHTML')
#             if script_content and 'uniqueKey' in script_content:
#                 match = re.search(r'"uniqueKey"\s*:\s*"([^"]+)"', script_content)
#                 if match:
#                     unique_key = match.group(1)
#                     logger.debug(f"Извлечён unique_key из скрипта: {unique_key}")
#                     return unique_key
#         logger.warning(f"Не удалось извлечь unique_key для {market_url}")
#         return None
#     except Exception as e:
#         logger.error(f"Ошибка при извлечении unique_key для {market_url}: {e}")
#         return None
#
#
# def get_morpho_data(unique_key, chain_id=1):
#     """Получает данные через API Morpho для указанного unique_key."""
#     if not unique_key:
#         logger.error("unique_key не предоставлен.")
#         return None, None, None, None
#
#     url = "https://api.morpho.org/graphql"
#     query = """
#     query {
#         marketByUniqueKey(uniqueKey: "%s", chainId: %d) {
#             state {
#                 borrowAssetsUsd
#                 supplyAssetsUsd
#                 borrowApy
#                 utilization
#             }
#         }
#     }
#     """ % (unique_key, chain_id)
#
#     try:
#         payload = {"query": query}
#         headers = {
#             'Content-Type': 'application/json',
#             'Accept': 'application/graphql-response+json, application/json',
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         }
#         response = requests.post(url, json=payload, headers=headers)
#         print(f"API response for unique_key {unique_key}: {response.text}")
#         response.raise_for_status()
#
#         data = response.json()
#         if 'errors' in data:
#             logger.error(f"Ошибки в ответе API для unique_key {unique_key}:")
#             for error in data['errors']:
#                 logger.error(f"- {error.get('message')}")
#                 if 'extensions' in error:
#                     logger.error(f"  Дополнительно: {error['extensions']}")
#             return None, None, None, None
#
#         market_data = data.get('data', {}).get('marketByUniqueKey', {})
#         if not market_data:
#             logger.warning(f"Рынок с unique_key {unique_key} не найден.")
#             return None, None, None, None
#
#         state = market_data.get('state', {})
#         total_borrow = state.get('borrowAssetsUsd')
#         total_supply = state.get('supplyAssetsUsd')
#         borrow_apy = state.get('borrowApy')
#         utilization = state.get('utilization')
#
#         print(f"Raw API values for {unique_key}: Total Borrow={total_borrow}, Total Supply={total_supply}, "
#               f"Borrow APY={borrow_apy}, Utilization={utilization}")
#
#         if any(v is None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#             logger.warning(f"Не удалось найти некоторые данные для unique_key {unique_key}: "
#                            f"Total Borrow={total_borrow}, Total Supply={total_supply}, "
#                            f"Borrow APY={borrow_apy}, Utilization={utilization}")
#             return None, None, None, None
#
#         total_borrow = convert_to_number(total_borrow)
#         total_supply = convert_to_number(total_supply)
#         borrow_apy = format_percentage(borrow_apy)
#         utilization = format_percentage(utilization)
#
#         print(f"Converted values for {unique_key}: Total Borrow={total_borrow}, Total Supply={total_supply}, "
#               f"Borrow APY={borrow_apy}, Utilization={utilization}")
#
#         if any(v is None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#             logger.warning(f"Ошибка преобразования данных для unique_key {unique_key}: "
#                            f"Total Borrow={total_borrow}, Total Supply={total_supply}, "
#                            f"Borrow APY={borrow_apy}, Utilization={utilization}")
#             return None, None, None, None
#
#         return total_borrow, total_supply, borrow_apy, utilization
#
#     except requests.HTTPError as e:
#         logger.error(f"HTTP ошибка для unique_key {unique_key}: {e}")
#         return None, None, None, None
#     except Exception as e:
#         logger.error(f"Произошла ошибка для unique_key {unique_key}: {e}")
#         return None, None, None, None
#
#
# def calculate_new_rates(total_borrow, total_supply, borrow_apy, utilization, deposit_amount):
#     """Рассчитывает новую Borrow Rate, Supply Rate, Instantaneous Supply Rate и новую утилизацию."""
#     u_target = 0.9
#     k_d = 4
#     k_p = 50 / 31_536_000
#     seconds_per_year = 31_536_000
#     fee = 0
#     time_since_last_interaction = 1000
#
#     borrow_apy_decimal = borrow_apy / 100
#     r_prev = math.log(1 + borrow_apy_decimal) / seconds_per_year
#
#     u_current = utilization / 100
#     new_total_supply = total_supply + deposit_amount
#     u_new = total_borrow / new_total_supply
#
#     if u_current > u_target:
#         e_u_current = (u_current - u_target) / (1 - u_target)
#     else:
#         e_u_current = (u_current - u_target) / u_target
#
#     if u_new > u_target:
#         e_u_new = (u_new - u_target) / (1 - u_target)
#     else:
#         e_u_new = (u_new - u_target) / u_target
#
#     if u_current > u_target:
#         curve_u_current = (k_d - 1) * e_u_current + 1
#     else:
#         curve_u_current = (1 - 1 / k_d) * e_u_current + 1
#
#     if u_new > u_target:
#         curve_u_new = (k_d - 1) * e_u_new + 1
#     else:
#         curve_u_new = (1 - 1 / k_d) * e_u_new + 1
#
#     r_T_last = r_prev / curve_u_current
#     speed_t = math.exp(k_p * e_u_current * time_since_last_interaction)
#     r_T_t = r_T_last * speed_t
#
#     r_new = r_T_t * curve_u_new
#     borrow_apy_new = (math.exp(r_new * seconds_per_year) - 1) * 100
#     supply_apy_new = borrow_apy_new / 100 * u_new * (1 - fee) * 100
#     r_supply_t = r_new * u_new * (1 - fee)
#     instantaneous_supply_rate_annual = r_supply_t * seconds_per_year * 100
#     u_new_percent = u_new * 100
#
#     return borrow_apy_new, supply_apy_new, instantaneous_supply_rate_annual, u_new_percent
#
#
# def get_max_investment_for_v2_utilization(total_borrow: float, total_supply: float) -> float:
#     """Рассчитывает максимальное вложение, чтобы новая утилизация была > 0.90."""
#     available_liquidity = total_supply - total_borrow
#     reserve_size = total_supply
#     max_investment = (0.10 * reserve_size - available_liquidity) / 0.90
#     return max_investment if max_investment > 0 else 0
#
#
# def calculate_profit_for_project(data: Dict, investment: float, total_borrow: float, total_supply: float,
#                                  borrow_apy: float, utilization: float, enforce_v2_utilization: bool = False) -> Tuple[
#     Optional[float], Optional[float], Optional[float]]:
#     """Рассчитывает дневную прибыль, новую Lend APR и утилизацию для заданного вложения."""
#     try:
#         utilization = utilization / 100
#         available_liquidity = total_supply - total_borrow
#         reserve_size = total_supply
#         lend_apr = borrow_apy * utilization
#         logger.debug(
#             f"Данные: Market={data['Market']}, Lend APR={lend_apr:.2f}%, Utilization={utilization * 100:.2f}%, "
#             f"Available Liquidity={available_liquidity:.2f}, Reserve Size={reserve_size:.2f}, "
#             f"Investment={investment:.2f}, Enforce V2={enforce_v2_utilization}"
#         )
#     # """Рассчитывает дневную прибыль, новую ставку и утилизацию для заданного вложения."""
#     # try:
#     #     lend_apr = data.get("Rate", 0.0)
#     #     utilization = utilization / 100
#     #     available_liquidity = total_supply - total_borrow
#     #     reserve_size = total_supply
#     #     logger.debug(
#     #         f"Данные: Market={data['Market']}, Lend APR={lend_apr}, Utilization={utilization}, "
#     #         f"Available Liquidity={available_liquidity}, Reserve Size={reserve_size}"
#     #     )
#     except Exception as e:
#         logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
#         return None, None, None
#
#     MIN_APR = 0
#     if lend_apr <= MIN_APR:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_APR}%)")
#         return None, None, None
#     if utilization >= 1.01:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization * 100}% >= 101%)")
#         return None, None, None
#     if reserve_size == 0:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
#         return None, None, None
#
#     if enforce_v2_utilization and utilization < 0.91:
#         logger.info(f"Пара отфильтрована: {data.get('Link')} (Initial Utilization={utilization * 100}% < 91%)")
#         return None, None, None
#
#     new_utilization = total_borrow / (total_supply + investment)
#
#     if enforce_v2_utilization and new_utilization <= 0.90:
#         logger.debug(f"Пропущено: new_utilization={new_utilization:.4f} <= 0.90")
#         return None, None, None
#
#     _, new_lend_apr, _, new_utilization_percent = calculate_new_rates(
#         total_borrow, total_supply, borrow_apy, utilization * 100, investment
#     )
#     daily_profit = (investment * new_lend_apr / 100) / 365.24
#
#     return daily_profit, new_lend_apr, new_utilization_percent / 100
#
# def calculate_optimal_investment(projects_data: List[Dict], max_total_investment: float = 200000) -> List[Dict]:
#     """Распределяет капитал между 1–4 проектами для максимизации общей дневной прибыли."""
#     # Рассчитываем начальную Lend APR для каждого проекта
#     for data in projects_data:
#         data['initial_lend_apr'] = data['Borrow APY'] * (data['Utilization'] / 100)
#
#     # Сортируем проекты по начальной Lend APR (в убывающем порядке)
#     sorted_projects = sorted(projects_data, key=lambda x: x['initial_lend_apr'], reverse=True)
#     valid_projects = sorted_projects[:4]  # Ограничиваем до 4 проектов
#
#     if not valid_projects:
#         logger.info("Ни один проект не прошел фильтрацию")
#         return []
#
#     step = 5000
#     results = []
#
#     def distribute_capital(projects: List[Dict], enforce_v2_utilization: bool) -> Tuple[float, Dict]:
#         best_total_profit = 0
#         best_investments = {}
#
#         # Проверяем комбинации из 1, 2, 3 или 4 проектов
#         for num_projects in range(1, min(5, len(projects) + 1)):
#             for combo in combinations(projects, num_projects):
#                 current_investments = {data['Link']: 0 for data in combo}
#                 remaining_capital = max_total_investment
#                 total_profit = 0
#
#                 # Распределяем капитал по шагам
#                 while remaining_capital >= step:
#                     best_additional_profit = 0
#                     best_project = None
#                     best_new_investment = None
#
#                     for data in combo:
#                         current_investment = current_investments[data['Link']]
#                         new_investment = current_investment + step
#                         max_investment = get_max_investment_for_v2_utilization(
#                             data['Total Borrow'], data['Total Supply']
#                         ) if enforce_v2_utilization else max_total_investment
#                         if new_investment > max_investment:
#                             continue
#                         profit, lend_apr, utilization = calculate_profit_for_project(
#                             data, new_investment, data['Total Borrow'], data['Total Supply'],
#                             data['Borrow APY'], data['Utilization'], enforce_v2_utilization
#                         )
#                         if profit is None:
#                             continue
#                         current_profit, _, _ = calculate_profit_for_project(
#                             data, current_investment, data['Total Borrow'], data['Total Supply'],
#                             data['Borrow APY'], data['Utilization'], enforce_v2_utilization
#                         ) or (0, 0, 0)
#                         additional_profit = profit - current_profit
#                         if additional_profit > best_additional_profit:
#                             best_additional_profit = additional_profit
#                             best_project = data
#                             best_new_investment = new_investment
#
#                     if best_project is None:
#                         break
#
#                     current_investments[best_project['Link']] = best_new_investment
#                     remaining_capital -= step
#                     total_profit += best_additional_profit
#
#                 # Сохраняем результаты для комбинации
#                 combo_investments = {}
#                 for data in combo:
#                     investment = current_investments[data['Link']]
#                     if investment > 0:
#                         profit, lend_apr, utilization = calculate_profit_for_project(
#                             data, investment, data['Total Borrow'], data['Total Supply'],
#                             data['Borrow APY'], data['Utilization'], enforce_v2_utilization
#                         )
#                         if profit is not None:
#                             combo_investments[data['Link']] = {
#                                 'investment': investment,
#                                 'daily_profit': profit,
#                                 'lend_apr': lend_apr,
#                                 'utilization': utilization
#                             }
#
#                 if total_profit > best_total_profit:
#                     best_total_profit = total_profit
#                     best_investments = combo_investments
#
#                 logger.debug(f"Комбинация {num_projects} проектов: Total Profit={total_profit:.2f}, "
#                              f"Investments={current_investments}")
#
#         return best_total_profit, best_investments
#
#     # Распределяем капитал для обоих сценариев
#     total_profit_default, investments_default = distribute_capital(valid_projects, enforce_v2_utilization=False)
#     total_profit_constrained, investments_constrained = distribute_capital(
#         valid_projects, enforce_v2_utilization=True
#     )
#
#     # Формируем результаты для всех проектов
#     for data in projects_data:
#         link = data['Link']
#         default_result = investments_default.get(link,
#                                                  {'investment': 0, 'daily_profit': 0, 'lend_apr': 0, 'utilization': 0})
#         constrained_result = investments_constrained.get(link,
#                                                         {'investment': 0, 'daily_profit': 0, 'lend_apr': 0,
#                                                          'utilization': 0})
#         results.append({
#             'Market': data['Market'],
#             'Link': link,
#             'Trusted By': data['Trusted By'],
#             'default': {
#                 'optimal_investment': default_result['investment'],
#                 'max_profit': default_result['daily_profit'],
#                 'optimal_lend_apr': default_result['lend_apr'],
#                 'optimal_utilization': default_result['utilization'] * 100,
#                 'total_profit': total_profit_default
#             },
#             'v2_utilization_constrained': {
#                 'optimal_investment': constrained_result['investment'],
#                 'max_profit': constrained_result['daily_profit'],
#                 'optimal_lend_apr': constrained_result['lend_apr'],
#                 'optimal_utilization': constrained_result['utilization'] * 100,
#                 'total_profit': total_profit_constrained
#             }
#         })
#
#     logger.info(f"Результаты распределения: {results}")
#     return results
#
#
#
# def main():
#     # Параметры для фильтрации
#     url = "https://app.morpho.org/ethereum/borrow"
#     rate_threshold = 8.0  # Порог для Rate (в процентах)
#     liquidity_threshold = 1_0_000  # Порог для Total Liquidity (1M)
#     trusted_by_allowed = ['Steakhouse', 'Gauntlet']  # Фильтрация по Trusted By
#     max_pages = 7  # Максимальное количество страниц
#     max_total_investment = 200_000
#     chain_id = 1
#
#     try:
#         # Парсинг и фильтрация данных
#         print("Начинаем парсинг данных...")
#         filtered_data = parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages)
#         if filtered_data is None or filtered_data.empty:
#             logger.error("Не удалось получить данные или данные не соответствуют фильтрам.")
#             print("Ошибка: Данные не получены или пусты. Проверьте лог-файл 'morpho_scraper.log'.")
#             return
#
#         print("Отфильтрованные данные:\n", filtered_data.to_string())
#
#         # Инициализация драйвера для API-запросов
#         chrome_options = Options()
#         chrome_options.add_argument("--headless")
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         chrome_options.add_argument("--disable-gpu")
#         chrome_options.add_argument("--window-size=1920,1080")
#         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#         print("WebDriver для API успешно инициализирован.")
#
#         # Подготовка данных для распределения капитала
#         projects_data = []
#         for index, row in filtered_data.iterrows():
#             market_name = row['text Collateral']
#             market_url = row['link Collateral']
#             trusted_by = ', '.join(row['Trusted By'])
#             print(f"\nОбработка рынка: {market_name}, URL: {market_url}")
#
#             unique_key = extract_unique_key(market_url, driver)
#             print(f"Извлеченный unique_key: {unique_key}")
#             if not unique_key:
#                 logger.warning(f"Пропуск рынка {market_name}: не удалось извлечь unique_key.")
#                 print(f"Пропуск рынка {market_name}: не удалось извлечь unique_key.")
#                 continue
#
#             total_borrow, total_supply, borrow_apy, utilization = get_morpho_data(unique_key, chain_id)
#             print(f"API данные для {market_name}: Total Borrow={total_borrow}, Total Supply={total_supply}, "
#                   f"Borrow APY={borrow_apy}, Utilization={utilization}")
#             if all(v is not None for v in [total_borrow, total_supply, borrow_apy, utilization]):
#                 projects_data.append({
#                     'Market': market_name,
#                     'Link': market_url,
#                     'Trusted By': trusted_by,
#                     'Rate': row['Rate'],
#                     'Total Borrow': total_borrow,
#                     'Total Supply': total_supply,
#                     'Borrow APY': borrow_apy,
#                     'Utilization': utilization
#                 })
#                 logger.info(f"Добавлен рынок: {market_name}, Total Borrow={total_borrow:,.2f}, "
#                             f"Total Supply={total_supply:,.2f}, Borrow APY={borrow_apy:.2f}%, Utilization={utilization:.2f}%")
#                 print(f"Добавлен рынок: {market_name}")
#             else:
#                 logger.warning(f"Не удалось получить данные для рынка: {market_name}")
#                 print(f"Не удалось получить данные для рынка: {market_name}")
#
#         driver.quit()
#         print("WebDriver для API закрыт.")
#
#         if not projects_data:
#             logger.error("Список projects_data пуст. Прерываем выполнение.")
#             print("Ошибка: Список projects_data пуст. Проверьте API или фильтры.")
#             return
#
#         print(f"Собранные проекты ({len(projects_data)}):", projects_data)
#
#         # Распределение капитала
#         print("\nЗапускаем распределение капитала...")
#         results = calculate_optimal_investment(projects_data, max_total_investment=max_total_investment)
#         print("Результаты распределения капитала:", results)
#
#         if results:
#             results_df = pd.DataFrame([
#                 {
#                     'Market': r['Market'],
#                     'Trusted By': r['Trusted By'],
#                     'Link': r['Link'],
#                     'Default Investment (USD)': r['default']['optimal_investment'],
#                     'Default Daily Profit (USD)': r['default']['max_profit'],
#                     'Default Lend APR (%)': r['default']['optimal_lend_apr'],
#                     'Default Utilization (%)': r['default']['optimal_utilization'],
#                     'Default Total Profit (USD)': r['default']['total_profit'],
#                     'Constrained Investment (USD)': r['v2_utilization_constrained']['optimal_investment'],
#                     'Constrained Daily Profit (USD)': r['v2_utilization_constrained']['max_profit'],
#                     'Constrained Lend APR (%)': r['v2_utilization_constrained']['optimal_lend_apr'],
#                     'Constrained Utilization (%)': r['v2_utilization_constrained']['optimal_utilization'],
#                     'Constrained Total Profit (USD)': r['v2_utilization_constrained']['total_profit']
#                 }
#                 for r in results
#             ])
#             print("\nИтоговые результаты:\n", results_df.to_string(index=False))
#             results_df.to_csv('morpho_investment_allocation.csv', index=False)
#             logger.info("Результаты сохранены в 'morpho_investment_allocation.csv'.")
#             print("Результаты сохранены в 'morpho_investment_allocation.csv'.")
#         else:
#             logger.warning("Результаты распределения капитала пусты.")
#             print("Ошибка: Результаты распределения капитала пусты.")
#
#     except Exception as e:
#         logger.error(f"Критическая ошибка в main: {e}")
#         print(f"Критическая ошибка: {e}")
#         if 'driver' in locals():
#             driver.quit()
#             print("WebDriver закрыт в случае ошибки.")
#
#
# if __name__ == "__main__":
#     main()

import os
import time
import requests
import pandas as pd
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import math

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Настройки
#BOT_TOKEN = os.getenv("BOT_TOKEN") #8060812740:AAEBXpMOoCZ2RdD8JY7pO0EXf0aQQN6jQJg

#CHAT_IDS = [6192278046, 306507209]
#CHAT_IDS = [int(chat_id) for chat_id in os.getenv("CHAT_IDS", "").split(",") if chat_id]


BOT_TOKEN = "8060812740:AAEBXpMOoCZ2RdD8JY7pO0EXf0aQQN6jQJg"
CHAT_IDS = [6192278046]
RATE_THRESHOLD = 5.0
LIQUIDITY_THRESHOLD = 1_0_000
#TRUSTED_BY_ALLOWED = ['Steakhouse', 'Gauntlet', 'Mevcapital']
TRUSTED_BY_ALLOWED = ['Mevcapital']
MAX_PAGES = 10
MAX_TOTAL_INVESTMENT = 200_000
CHAIN_ID = 1
URL = "https://app.morpho.org/ethereum/borrow"
MAX_ITERATION_TIME = 1800  # 30 минут в секундах
MIN_INTERVAL = 7200 - 1800  # 2 hours minus 30 minutes


# def send_to_telegram(message):
#     """Отправляет сообщение в Telegram."""
#     if not BOT_TOKEN or not CHAT_IDS:
#         logger.error("BOT_TOKEN или CHAT_IDS не заданы")
#         return
#     url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#     for chat_id in CHAT_IDS:
#         for attempt in range(3):
#             try:
#                 payload = {"chat_id": chat_id, "text": message[:4096]}
#                 response = requests.post(url, json=payload, timeout=10)
#                 response.raise_for_status()
#                 logger.info(f"Сообщение отправлено в Telegram для chat_id {chat_id}")
#                 break
#             except requests.RequestException as e:
#                 logger.error(f"Попытка {attempt + 1}/3: Ошибка отправки в Telegram: {e}")
#                 if attempt == 2:
#                     logger.error(f"Не удалось отправить сообщение для chat_id {chat_id}")
#                 time.sleep(2)

def send_to_telegram(message):
    """Отправляет сообщение в Telegram."""
    if not BOT_TOKEN or not CHAT_IDS:
        logger.error("BOT_TOKEN или CHAT_IDS не заданы")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            payload = {"chat_id": chat_id, "text": message[:4096], "parse_mode": "Markdown"}
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Сообщение отправлено в Telegram для chat_id {chat_id}")
        except requests.RequestException as e:
            logger.error(f"Ошибка отправки в Telegram для chat_id {chat_id}: {e}")
            if isinstance(e, requests.HTTPError) and e.response.status_code == 400:
                logger.warning(f"Пропуск chat_id {chat_id} из-за ошибки 400 Bad Request")
                continue
            for attempt in range(1, 3):
                try:
                    time.sleep(2)
                    response = requests.post(url, json=payload, timeout=10)
                    response.raise_for_status()
                    logger.info(f"Сообщение отправлено в Telegram для chat_id {chat_id} после попытки {attempt + 1}")
                    break
                except requests.RequestException as retry_e:
                    logger.error(f"Попытка {attempt + 1}/3: Ошибка отправки в Telegram: {retry_e}")
            else:
                logger.error(f"Не удалось отправить сообщение для chat_id {chat_id}")


def get_driver():
    """Инициализирует WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.binary_location = '/usr/bin/chromium'
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
    logger.info("WebDriver успешно инициализирован.")
    return driver

def convert_to_number(value):
    try:
        if isinstance(value, (int, float)):
            return float(value)
        value = value.replace('$', '').replace(',', '').strip()
        value = value.split('\n')[0]
        value = re.sub(r'\s*(USDC|WETH|DAI|rUSD|USDT|wUSDL|USDS|[a-zA-Z]+)$', '', value).strip()
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'k' in value or 'K' in value:
            return float(value.replace('k', '').replace('K', '')) * 1_000
        elif '%' in value:
            return float(value.replace('%', ''))
        return float(value)
    except (ValueError, AttributeError) as e:
        logger.error(f"Ошибка преобразования значения '{value}': {e}")
        return None

def clean_name(name):
    name = name.replace('.Svg', '').replace('.svg', '').replace('-', ' ').title()
    #if name.lower() == 'mevcapital':
     #   return 'MEV Capital'
    return name

def extract_trusted_by_names(col, driver):
    try:
        tooltip = (col.get_attribute('title') or
                  col.get_attribute('data-tooltip') or
                  col.get_attribute('aria-label') or
                  col.get_attribute('data-tip') or
                  col.get_attribute('data-tooltip-content'))
        if tooltip:
            names = [clean_name(name.strip()) for name in tooltip.split(',')]
            logger.debug(f"Tooltip найден (атрибут): {names}")
            return names
        images = col.find_elements(By.TAG_NAME, 'img')
        names = []
        for img in images:
            src = img.get_attribute('src') or ''
            alt = img.get_attribute('alt') or ''
            if src and 'morpho.org' in src:
                name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
                names.append(clean_name(name))
            elif alt and alt != 'avatar':
                names.append(clean_name(alt))
        if names:
            logger.debug(f"Изображения найдены в ячейке: {names}")
            return names
        ActionChains(driver).move_to_element(col).perform()
        time.sleep(1)
        try:
            tooltip_element = driver.find_element(
                By.CSS_SELECTOR,
                '[class*="tooltip"], [class*="popover"], [role="tooltip"], [data-tip], [data-tooltip-content], .css-11nlj6z, .eedv4i31'
            )
            tooltip_images = tooltip_element.find_elements(By.TAG_NAME, 'img')
            for img in tooltip_images:
                src = img.get_attribute('src') or ''
                alt = img.get_attribute('alt') or ''
                if src and 'morpho.org' in src:
                    name = src.split('/')[-1].replace('.png', '').replace('.svg', '').replace('-', ' ').title()
                    names.append(clean_name(name))
                elif alt and alt != 'avatar':
                    names.append(clean_name(alt))
            if names:
                logger.debug(f"Изображения найдены в tooltip: {names}")
                return names
            text = tooltip_element.text.strip()
            if text:
                names = [clean_name(name.strip()) for name in text.split(',')]
                logger.debug(f"Tooltip найден (текст): {names}")
                return names
        except:
            logger.debug(f"Tooltip не найден для Trusted By в строке: {col.text}")
        return [col.text.strip()]
    except Exception as e:
        logger.error(f"Ошибка при извлечении Trusted By: {e}")
        return [col.text.strip()]

def extract_collateral_links(col):
    try:
        link_elements = col.find_elements(By.TAG_NAME, 'a')
        for link in link_elements:
            href = link.get_attribute('href')
            if href:
                return href
        href = (col.get_attribute('href') or
                col.get_attribute('data-href') or
                col.get_attribute('data-url'))
        if href:
            return href
        images = col.find_elements(By.TAG_NAME, 'img')
        for img in images:
            parent = img.find_element(By.XPATH, './..')
            href = parent.get_attribute('href') or parent.get_attribute('data-href')
            if href:
                return href
        return None
    except Exception as e:
        logger.error(f"Ошибка при извлечении ссылки из Collateral: {e}")
        return None

# def parse_and_filter_data(url, rate_threshold, liquidity_threshold, trusted_by_allowed, max_pages, driver, iteration_start_time):
#     try:
#         logger.info(f"Открываем страницу: {url}")
#         driver.get(url)
#         logger.info("Ожидаем загрузки таблицы...")
#         table_selector = 'table'
#         WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#         )
#         logger.info("Таблица найдена.")
#         table = driver.find_element(By.CSS_SELECTOR, table_selector)
#         headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
#         logger.info(f"Заголовки: {headers}")
#         all_data = []
#
#         for page in range(1, max_pages + 1):
#             if time.time() - iteration_start_time > MAX_ITERATION_TIME:
#                 logger.warning(f"Превышено время итерации ({MAX_ITERATION_TIME} сек). Прерываем парсинг.")
#                 send_to_telegram(f"Превышено время парсинга ({MAX_ITERATION_TIME} сек).")
#                 break
#             logger.info(f"Обработка страницы {page}...")
#             try:
#                 WebDriverWait(driver, 20).until(
#                     EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                 )
#                 table = driver.find_element(By.CSS_SELECTOR, table_selector)
#                 rows = table.find_elements(By.TAG_NAME, 'tr')[1:]
#                 if not rows:
#                     logger.info(f"На странице {page} нет строк данных.")
#                     break
#                 for row in rows:
#                     cols = row.find_elements(By.TAG_NAME, 'td')
#                     row_data = []
#                     for i, col in enumerate(cols):
#                         if headers[i] == 'Trusted By':
#                             row_data.append(extract_trusted_by_names(col, driver))
#                         elif headers[i] == 'Collateral':
#                             row_data.append({
#                                 'text': col.text.strip(),
#                                 'link': extract_collateral_links(col)
#                             })
#                         else:
#                             row_data.append(col.text.strip())
#                     all_data.append(row_data)
#                 logger.info(f"Собрано {len(rows)} строк на странице {page}")
#                 if page < max_pages:
#                     try:
#                         next_button = WebDriverWait(driver, 10).until(
#                             EC.element_to_be_clickable(
#                                 (By.CSS_SELECTOR,
#                                  'button.css-ktorsf:not([disabled]):not([style*="rotate(180deg)"]), '
#                                  'button:has(svg[icon="ArrowPlain20"]):not([disabled]):not([style*="rotate(180deg)"]), '
#                                  '[class*="pagination"] button:not([style*="rotate(180deg)"])')
#                             )
#                         )
#                         logger.info("Кнопка 'Next' найдена.")
#                         next_button.click()
#                         logger.info("Кликаем по кнопке 'Next'...")
#                         time.sleep(1)
#                         WebDriverWait(driver, 20).until(EC.staleness_of(table))
#                     except Exception as e:
#                         logger.info(f"Кнопка 'Next' не найдена: {e}")
#                         next_url = f"{url}?page={page + 1}"
#                         logger.info(f"Пробуем перейти на страницу {page + 1} через URL: {next_url}")
#                         driver.get(next_url)
#                         try:
#                             WebDriverWait(driver, 20).until(
#                                 EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
#                             )
#                         except Exception as url_e:
#                             logger.info(f"Ошибка перехода через URL: {url_e}")
#                             break
#             except Exception as page_e:
#                 logger.error(f"Ошибка обработки страницы {page}: {page_e}")
#                 break
#         if not all_data:
#             logger.warning("Нет собранных данных.")
#             send_to_telegram("Ошибка: Нет собранных данных с сайта.")
#             return None
#         df = pd.DataFrame(all_data, columns=headers)
#         logger.info(f"Исходный DataFrame:\n{df.to_string()}")
#         df['text Collateral'] = df['Collateral'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
#         df['link Collateral'] = df['Collateral'].apply(lambda x: x['link'] if isinstance(x, dict) else None)
#         df['Rate'] = df['Rate'].apply(convert_to_number)
#         df['Total Liquidity'] = df['Total Liquidity'].apply(convert_to_number)
#         logger.info(f"DataFrame после преобразования:\n{df.to_string()}")
#         df = df.dropna(subset=['Rate', 'Total Liquidity'])
#         logger.info(f"DataFrame после удаления пустых значений:\n{df.to_string()}")
#         filtered_df = df[
#             (df['Rate'] > rate_threshold) &
#             (df['Total Liquidity'] > liquidity_threshold) &
#             (df['Trusted By'].apply(lambda x: not trusted_by_allowed or any(name in trusted_by_allowed for name in x)))
#         ]
#         logger.info(f"Отфильтрованный DataFrame:\n{filtered_df.to_string()}")
#         if filtered_df.empty:
#             logger.warning("Нет строк, удовлетворяющих условиям фильтрации.")
#             send_to_telegram("Ошибка: Нет строк, удовлетворяющих условиям фильтрации.")
#             return None
#         result_df = filtered_df[['text Collateral', 'link Collateral', 'Rate', 'Trusted By', 'Total Liquidity']]
#         sorted_df = result_df.sort_values(by='Rate', ascending=False)
#         return sorted_df
#     except Exception as e:
#         logger.error(f"Критическая ошибка в parse_and_filter_data: {e}")
#         send_to_telegram(f"Критическая ошибка в parse_and_filter_data: {e}")
#         return None

def parse_and_filter_data(url: str, rate_threshold: float, liquidity_threshold: float,
                          trusted_by_allowed: List[str], max_pages: int, driver,
                          iteration_start_time: float) -> Optional[pd.DataFrame]:
    """
    Парсит данные с сайта Morpho, фильтрует их по заданным критериям и возвращает отфильтрованный DataFrame.

    Args:
        url: URL страницы для парсинга.
        rate_threshold: Минимальная процентная ставка для фильтрации.
        liquidity_threshold: Минимальная ликвидность для фильтрации.
        trusted_by_allowed: Список разрешенных имен для Trusted By.
        max_pages: Максимальное количество страниц для парсинга.
        driver: WebDriver для Selenium.
        iteration_start_time: Время начала итерации для контроля времени выполнения.

    Returns:
        Отфильтрованный DataFrame или None, если данные не получены или не прошли фильтры.
    """
    try:
        logger.info(f"Открываем страницу: {url}")
        driver.get(url)
        logger.info("Ожидаем загрузки таблицы...")
        table_selector = 'table'
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
        )
        logger.info("Таблица найдена.")
        table = driver.find_element(By.CSS_SELECTOR, table_selector)
        headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, 'th')]
        logger.info(f"Заголовки таблицы: {headers}")
        all_data = []

        for page in range(1, max_pages + 1):
            if time.time() - iteration_start_time > MAX_ITERATION_TIME:
                logger.warning(f"Превышено время итерации ({MAX_ITERATION_TIME} сек). Прерываем парсинг.")
                send_to_telegram(f"Превышено время парсинга ({MAX_ITERATION_TIME} сек).")
                break
            logger.info(f"Обработка страницы {page}...")
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
                )
                table = driver.find_element(By.CSS_SELECTOR, table_selector)
                rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # Пропускаем заголовок
                if not rows:
                    logger.info(f"На странице {page} нет строк данных.")
                    break
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, 'td')
                    row_data = []
                    for i, col in enumerate(cols):
                        if headers[i] == 'Trusted By':
                            trusted_by_names = extract_trusted_by_names(col, driver)
                            logger.debug(f"Извлечены Trusted By для строки: {trusted_by_names}")
                            row_data.append(trusted_by_names)
                        elif headers[i] == 'Collateral':
                            row_data.append({
                                'text': col.text.strip(),
                                'link': extract_collateral_links(col)
                            })
                        else:
                            row_data.append(col.text.strip())
                    all_data.append(row_data)
                logger.info(f"Собрано {len(rows)} строк на странице {page}")
                if page < max_pages:
                    try:
                        next_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR,
                                 'button.css-ktorsf:not([disabled]):not([style*="rotate(180deg)"]), '
                                 'button:has(svg[icon="ArrowPlain20"]):not([disabled]):not([style*="rotate(180deg)"]), '
                                 '[class*="pagination"] button:not([style*="rotate(180deg)"])')
                            )
                        )
                        logger.info("Кнопка 'Next' найдена.")
                        next_button.click()
                        logger.info("Кликаем по кнопке 'Next'...")
                        time.sleep(1)
                        WebDriverWait(driver, 20).until(EC.staleness_of(table))
                    except Exception as e:
                        logger.info(f"Кнопка 'Next' не найдена: {e}")
                        next_url = f"{url}?page={page + 1}"
                        logger.info(f"Пробуем перейти на страницу {page + 1} через URL: {next_url}")
                        driver.get(next_url)
                        try:
                            WebDriverWait(driver, 20).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
                            )
                        except Exception as url_e:
                            logger.info(f"Ошибка перехода через URL: {url_e}")
                            break
            except Exception as page_e:
                logger.error(f"Ошибка обработки страницы {page}: {page_e}")
                break
        if not all_data:
            logger.warning("Нет собранных данных.")
            send_to_telegram("Ошибка: Нет собранных данных с сайта.")
            return None
        df = pd.DataFrame(all_data, columns=headers)
        logger.info(f"Исходный DataFrame:\n{df.to_string()}")
        df['text Collateral'] = df['Collateral'].apply(lambda x: x['text'] if isinstance(x, dict) else x)
        df['link Collateral'] = df['Collateral'].apply(lambda x: x['link'] if isinstance(x, dict) else None)
        df['Rate'] = df['Rate'].apply(convert_to_number)
        df['Total Liquidity'] = df['Total Liquidity'].apply(convert_to_number)
        logger.info(f"DataFrame после преобразования:\n{df.to_string()}")
        df = df.dropna(subset=['Rate', 'Total Liquidity'])
        logger.info(f"DataFrame после удаления пустых значений:\n{df.to_string()}")
        logger.info(f"Фильтрация по Trusted By: {trusted_by_allowed}")
        logger.info(f"Trusted By в DataFrame:\n{df[['text Collateral', 'Trusted By']].to_string()}")
        filtered_df = df[
            (df['Rate'] > rate_threshold) &
            (df['Total Liquidity'] > liquidity_threshold) &
            (df['Trusted By'].apply(lambda x: any(name in trusted_by_allowed for name in x)))
            ]
        logger.info(f"Отфильтрованный DataFrame:\n{filtered_df.to_string()}")
        if filtered_df.empty:
            logger.warning("Нет строк, удовлетворяющих условиям фильтрации.")
            send_to_telegram("Ошибка: Нет строк, удовлетворяющих условиям фильтрации.")
            return None
        result_df = filtered_df[['text Collateral', 'link Collateral', 'Rate', 'Trusted By', 'Total Liquidity']]
        sorted_df = result_df.sort_values(by='Rate', ascending=False)
        logger.info(f"Итоговый DataFrame после сортировки:\n{sorted_df.to_string()}")
        return sorted_df
    except Exception as e:
        logger.error(f"Критическая ошибка в parse_and_filter_data: {e}")
        send_to_telegram(f"Критическая ошибка в parse_and_filter_data: {e}")
        return None

def format_percentage(value):
    if value is not None:
        return float(value) * 100
    return None

def extract_unique_key(market_url, driver):
    try:
        match = re.search(r'market/([^/]+)/?', market_url)
        if match:
            unique_key = match.group(1)
            logger.debug(f"Извлечён unique_key из URL: {unique_key}")
            return unique_key
        driver.get(market_url)
        logger.debug(f"Открываем страницу рынка для извлечения unique_key: {market_url}")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )
        scripts = driver.find_elements(By.TAG_NAME, 'script')
        for script in scripts:
            script_content = script.get_attribute('innerHTML')
            if script_content and 'uniqueKey' in script_content:
                match = re.search(r'"uniqueKey"\s*:\s*"([^"]+)"', script_content)
                if match:
                    unique_key = match.group(1)
                    logger.debug(f"Извлечён unique_key из скрипта: {unique_key}")
                    return unique_key
        logger.warning(f"Не удалось извлечь unique_key для {market_url}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при извлечении unique_key для {market_url}: {e}")
        return None

def get_morpho_data(unique_key, chain_id=1):
    if not unique_key:
        logger.error("unique_key не предоставлен.")
        return None, None, None, None
    url = "https://api.morpho.org/graphql"
    query = """
    query {
        marketByUniqueKey(uniqueKey: "%s", chainId: %d) {
            state {
                borrowAssetsUsd
                supplyAssetsUsd
                borrowApy
                utilization
            }
        }
    }
    """ % (unique_key, chain_id)
    try:
        payload = {"query": query}
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/graphql-response+json, application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        logger.info(f"API response for unique_key {unique_key}: {response.text}")
        response.raise_for_status()
        data = response.json()
        if 'errors' in data:
            logger.error(f"Ошибки в ответе API для unique_key {unique_key}: {data['errors']}")
            return None, None, None, None
        market_data = data.get('data', {}).get('marketByUniqueKey', {})
        if not market_data:
            logger.warning(f"Рынок с unique_key {unique_key} не найден.")
            return None, None, None, None
        state = market_data.get('state', {})
        total_borrow = state.get('borrowAssetsUsd')
        total_supply = state.get('supplyAssetsUsd')
        borrow_apy = state.get('borrowApy')
        utilization = state.get('utilization')
        logger.info(f"Raw API values for {unique_key}: Total Borrow={total_borrow}, Total Supply={total_supply}, "
                    f"Borrow APY={borrow_apy}, Utilization={utilization}")
        if any(v is None for v in [total_borrow, total_supply, borrow_apy, utilization]):
            logger.warning(f"Некоторые данные отсутствуют для unique_key {unique_key}")
            return None, None, None, None
        total_borrow = convert_to_number(total_borrow)
        total_supply = convert_to_number(total_supply)
        borrow_apy = format_percentage(borrow_apy)
        utilization = format_percentage(utilization)
        logger.info(f"Converted values for {unique_key}: Total Borrow={total_borrow}, Total Supply={total_supply}, "
                    f"Borrow APY={borrow_apy}, Utilization={utilization}")
        if any(v is None for v in [total_borrow, total_supply, borrow_apy, utilization]):
            logger.warning(f"Ошибка преобразования данных для unique_key {unique_key}")
            return None, None, None, None
        return total_borrow, total_supply, borrow_apy, utilization
    except requests.RequestException as e:
        logger.error(f"Ошибка запроса для unique_key {unique_key}: {e}")
        return None, None, None, None

def calculate_new_rates(total_borrow, total_supply, borrow_apy, utilization, deposit_amount):
    u_target = 0.9
    k_d = 4
    k_p = 50 / 31_536_000
    seconds_per_year = 31_536_000
    fee = 0
    time_since_last_interaction = 1000
    borrow_apy_decimal = borrow_apy / 100
    r_prev = math.log(1 + borrow_apy_decimal) / seconds_per_year
    u_current = utilization / 100
    new_total_supply = total_supply + deposit_amount
    u_new = total_borrow / new_total_supply
    if u_current > u_target:
        e_u_current = (u_current - u_target) / (1 - u_target)
    else:
        e_u_current = (u_current - u_target) / u_target
    if u_new > u_target:
        e_u_new = (u_new - u_target) / (1 - u_target)
    else:
        e_u_new = (u_new - u_target) / u_target
    if u_current > u_target:
        curve_u_current = (k_d - 1) * e_u_current + 1
    else:
        curve_u_current = (1 - 1 / k_d) * e_u_current + 1
    if u_new > u_target:
        curve_u_new = (k_d - 1) * e_u_new + 1
    else:
        curve_u_new = (1 - 1 / k_d) * e_u_new + 1
    r_T_last = r_prev / curve_u_current
    speed_t = math.exp(k_p * e_u_current * time_since_last_interaction)
    r_T_t = r_T_last * speed_t
    r_new = r_T_t * curve_u_new
    borrow_apy_new = (math.exp(r_new * seconds_per_year) - 1) * 100
    supply_apy_new = borrow_apy_new / 100 * u_new * (1 - fee) * 100
    r_supply_t = r_new * u_new * (1 - fee)
    instantaneous_supply_rate_annual = r_supply_t * seconds_per_year * 100
    u_new_percent = u_new * 100
    return borrow_apy_new, supply_apy_new, instantaneous_supply_rate_annual, u_new_percent

def get_max_investment_for_v2_utilization(total_borrow: float, total_supply: float) -> float:
    available_liquidity = total_supply - total_borrow
    reserve_size = total_supply
    max_investment = (0.10 * reserve_size - available_liquidity) / 0.90
    return max_investment if max_investment > 0 else 0

def calculate_profit_for_project(data: Dict, investment: float, total_borrow: float, total_supply: float,
                                borrow_apy: float, utilization: float, enforce_v2_utilization: bool = False) -> Tuple[
    Optional[float], Optional[float], Optional[float]]:
    try:
        utilization = utilization / 100
        available_liquidity = total_supply - total_borrow
        reserve_size = total_supply
        lend_apr = borrow_apy * utilization
        logger.debug(
            f"Данные: Market={data['Market']}, Lend APR={lend_apr:.2f}%, Utilization={utilization * 100:.2f}%, "
            f"Available Liquidity={available_liquidity:.2f}, Reserve Size={reserve_size:.2f}, "
            f"Investment={investment:.2f}, Enforce V2={enforce_v2_utilization}"
        )
        MIN_APR = 0
        if lend_apr <= MIN_APR:
            logger.info(f"Пара отфильтрована: {data.get('Link')} (Lend APR={lend_apr} <= {MIN_APR}%)")
            return None, None, None
        if utilization >= 1.01:
            logger.info(f"Пара отфильтрована: {data.get('Link')} (Utilization Rate={utilization * 100}% >= 101%)")
            return None, None, None
        if reserve_size == 0:
            logger.info(f"Пара отфильтрована: {data.get('Link')} (Reserve Size={reserve_size} == 0)")
            return None, None, None
        if enforce_v2_utilization and utilization < 0.91:
            logger.info(f"Пара отфильтрована: {data.get('Link')} (Initial Utilization={utilization * 100}% < 91%)")
            return None, None, None
        new_utilization = total_borrow / (total_supply + investment)
        if enforce_v2_utilization and new_utilization <= 0.90:
            logger.debug(f"Пропущено: new_utilization={new_utilization:.4f} <= 0.90")
            return None, None, None
        _, new_lend_apr, _, new_utilization_percent = calculate_new_rates(
            total_borrow, total_supply, borrow_apy, utilization * 100, investment
        )
        daily_profit = (investment * new_lend_apr / 100) / 365.24
        return daily_profit, new_lend_apr, new_utilization_percent / 100
    except Exception as e:
        logger.error(f"Ошибка парсинга данных для {data.get('Link')}: {e}")
        return None, None, None

def calculate_optimal_investment(projects_data: List[Dict], max_total_investment: float = 200_000) -> List[Dict]:
    for data in projects_data:
        data['initial_lend_apr'] = data['Borrow APY'] * (data['Utilization'] / 100)
    sorted_projects = sorted(projects_data, key=lambda x: x['initial_lend_apr'], reverse=True)
    valid_projects = sorted_projects[:4]
    if not valid_projects:
        logger.info("Ни один проект не прошел фильтрацию")
        return []
    step = 5000
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
                        max_investment = get_max_investment_for_v2_utilization(
                            data['Total Borrow'], data['Total Supply']
                        ) if enforce_v2_utilization else max_total_investment
                        if new_investment > max_investment:
                            continue
                        profit, lend_apr, utilization = calculate_profit_for_project(
                            data, new_investment, data['Total Borrow'], data['Total Supply'],
                            data['Borrow APY'], data['Utilization'], enforce_v2_utilization
                        )
                        if profit is None:
                            continue
                        current_profit, _, _ = calculate_profit_for_project(
                            data, current_investment, data['Total Borrow'], data['Total Supply'],
                            data['Borrow APY'], data['Utilization'], enforce_v2_utilization
                        ) or (0, 0, 0)
                        additional_profit = profit - current_profit
                        if additional_profit > best_additional_profit:
                            best_additional_profit = additional_profit
                            best_project = data
                            best_new_investment = new_investment
                    if best_project is None:
                        break
                    current_investments[best_project['Link']] = best_new_investment
                    remaining_capital -= step
                    total_profit += best_additional_profit
                combo_investments = {}
                for data in combo:
                    investment = current_investments[data['Link']]
                    if investment > 0:
                        profit, lend_apr, utilization = calculate_profit_for_project(
                            data, investment, data['Total Borrow'], data['Total Supply'],
                            data['Borrow APY'], data['Utilization'], enforce_v2_utilization
                        )
                        if profit is not None:
                            combo_investments[data['Link']] = {
                                'investment': investment,
                                'daily_profit': profit,
                                'lend_apr': lend_apr,
                                'utilization': utilization
                            }
                if total_profit > best_total_profit:
                    best_total_profit = total_profit
                    best_investments = combo_investments
        return best_total_profit, best_investments
    total_profit_default, investments_default = distribute_capital(valid_projects, enforce_v2_utilization=False)
    total_profit_constrained, investments_constrained = distribute_capital(valid_projects, enforce_v2_utilization=True)
    for data in projects_data:
        link = data['Link']
        default_result = investments_default.get(link,
                                                {'investment': 0, 'daily_profit': 0, 'lend_apr': 0, 'utilization': 0})
        constrained_result = investments_constrained.get(link,
                                                        {'investment': 0, 'daily_profit': 0, 'lend_apr': 0,
                                                         'utilization': 0})
        results.append({
            'Market': data['Market'],
            'Link': link,
            'Trusted By': data['Trusted By'],
            'default': {
                'optimal_investment': default_result['investment'],
                'max_profit': default_result['daily_profit'],
                'optimal_lend_apr': default_result['lend_apr'],
                'optimal_utilization': default_result['utilization'] * 100,
                'total_profit': total_profit_default
            },
            'v2_utilization_constrained': {
                'optimal_investment': constrained_result['investment'],
                'max_profit': constrained_result['daily_profit'],
                'optimal_lend_apr': constrained_result['lend_apr'],
                'optimal_utilization': constrained_result['utilization'] * 100,
                'total_profit': total_profit_constrained
            }
        })
    logger.info(f"Результаты распределения: {results}")
    return results

def main():
    # Check if another instance ran too recently
    lock_file = "/tmp/morpho_scraper.lock"
    if os.path.exists(lock_file):
        with open(lock_file, "r") as f:
            last_run = float(f.read())
        if time.time() - last_run < MIN_INTERVAL:
            logger.info(f"Пропуск: Последний запуск был {time.time() - last_run:.2f} секунд назад, менее {MIN_INTERVAL} секунд.")
            send_to_telegram(f"Пропуск: Последний запуск был недавно. Следующий запуск через ~{int((MIN_INTERVAL - (time.time() - last_run)) / 60)} минут.")
            return

    # Update lock file
    with open(lock_file, "w") as f:
        f.write(str(time.time()))

    iteration_start_time = time.time()
    logger.info("Запуск Background Worker")
    send_to_telegram("Тест: Background Worker запущен")
    driver = None
    try:
        driver = get_driver()
        filtered_data = parse_and_filter_data(URL, RATE_THRESHOLD, LIQUIDITY_THRESHOLD, TRUSTED_BY_ALLOWED, MAX_PAGES, driver, iteration_start_time)
        if filtered_data is None or filtered_data.empty:
            logger.error("Не удалось получить данные или данные не соответствуют фильтрам.")
            send_to_telegram("Ошибка: Данные не получены или пусты.")
            return
        logger.info(f"Отфильтрованные данные:\n{filtered_data.to_string()}")
        projects_data = []
        for index, row in filtered_data.iterrows():
            if time.time() - iteration_start_time > MAX_ITERATION_TIME:
                logger.warning(f"Превышено время итерации ({MAX_ITERATION_TIME} сек). Прерываем обработку.")
                send_to_telegram(f"Превышено время обработки ({MAX_ITERATION_TIME} сек).")
                break
            market_name = row['text Collateral']
            market_url = row['link Collateral']
            trusted_by = ', '.join(row['Trusted By'])
            logger.info(f"Обработка рынка: {market_name}, URL: {market_url}")
            unique_key = extract_unique_key(market_url, driver)
            logger.info(f"Извлеченный unique_key: {unique_key}")
            if not unique_key:
                logger.warning(f"Пропуск рынка {market_name}: не удалось извлечь unique_key.")
                continue
            total_borrow, total_supply, borrow_apy, utilization = get_morpho_data(unique_key, CHAIN_ID)
            logger.info(f"API данные для {market_name}: Total Borrow={total_borrow}, Total Supply={total_supply}, "
                        f"Borrow APY={borrow_apy}, Utilization={utilization}")
            if all(v is not None for v in [total_borrow, total_supply, borrow_apy, utilization]):
                projects_data.append({
                    'Market': market_name,
                    'Link': market_url,
                    'Trusted By': trusted_by,
                    'Rate': row['Rate'],
                    'Total Borrow': total_borrow,
                    'Total Supply': total_supply,
                    'Borrow APY': borrow_apy,
                    'Utilization': utilization
                })
                logger.info(f"Добавлен рынок: {market_name}, Total Borrow={total_borrow:,.2f}, "
                            f"Total Supply={total_supply:,.2f}, Borrow APY={borrow_apy:.2f}%, Utilization={utilization:.2f}%")
            else:
                logger.warning(f"Не удалось получить данные для рынка: {market_name}")
        if not projects_data:
            logger.error("Список projects_data пуст.")
            send_to_telegram("Ошибка: Список markets пуст. Проверьте API или фильтры.")
            return
        logger.info(f"Собранные проекты ({len(projects_data)}): {projects_data}")
        results = calculate_optimal_investment(projects_data, max_total_investment=MAX_TOTAL_INVESTMENT)
        logger.info(f"Результаты распределения капитала: {results}")
        if results:
            results_df = pd.DataFrame([
                {
                    'Market': r['Market'],
                    'Trusted By': r['Trusted By'],
                    'Link': r['Link'],
                    'Default Investment (USD)': r['default']['optimal_investment'],
                    'Default Daily Profit (USD)': r['default']['max_profit'],
                    'Default Lend APR (%)': r['default']['optimal_lend_apr'],
                    'Default Utilization (%)': r['default']['optimal_utilization'],
                    'Default Total Profit (USD)': r['default']['total_profit'],
                    'Constrained Investment (USD)': r['v2_utilization_constrained']['optimal_investment'],
                    'Constrained Daily Profit (USD)': r['v2_utilization_constrained']['max_profit'],
                    'Constrained Lend APR (%)': r['v2_utilization_constrained']['optimal_lend_apr'],
                    'Constrained Utilization (%)': r['v2_utilization_constrained']['optimal_utilization'],
                    'Constrained Total Profit (USD)': r['v2_utilization_constrained']['total_profit']
                }
                for r in results
            ])
            logger.info(f"Итоговые результаты:\n{results_df.to_string(index=False)}")
            message = "=== Итоговые результаты распределения капитала ===\n\n"
            message += "*Вариант 1 (без ограничений на утилизацию)*:\n"
            has_default_investments = False
            for _, row in results_df.iterrows():
                if row['Default Investment (USD)'] > 0:
                    has_default_investments = True
                    message += (
                        f"*Рынок*: {row['Market']}\n"
                        f"*Trusted By*: {row['Trusted By']}\n"
                        f"*Ссылка*: {row['Link']}\n"
                        f"*Инвестиция*: ${row['Default Investment (USD)']:,.2f}\n"
                        f"*Дневная прибыль*: ${row['Default Daily Profit (USD)']:,.2f}\n"
                        f"*Lend APR*: {row['Default Lend APR (%)']:.2f}%\n"
                        f"*Утилизация*: {row['Default Utilization (%)']:.2f}%\n"
                        f"---\n"
                    )
            if not has_default_investments:
                message += "Нет инвестиций для Варианта 1.\n"
            message += f"*Общая дневная прибыль*: ${results_df['Default Total Profit (USD)'].iloc[0]:,.2f}\n\n"
            message += "*Вариант 2 (ограничение утилизации > 90%)*:\n"
            has_constrained_investments = False
            for _, row in results_df.iterrows():
                if row['Constrained Investment (USD)'] > 0:
                    has_constrained_investments = True
                    message += (
                        f"*Рынок*: {row['Market']}\n"
                        f"*Trusted By*: {row['Trusted By']}\n"
                        f"*Ссылка*: {row['Link']}\n"
                        f"*Инвестиция*: ${row['Constrained Investment (USD)']:,.2f}\n"
                        f"*Дневная прибыль*: ${row['Constrained Daily Profit (USD)']:,.2f}\n"
                        f"*Lend APR*: {row['Constrained Lend APR (%)']:.2f}%\n"
                        f"*Утилизация*: {row['Constrained Utilization (%)']:.2f}%\n"
                        f"---\n"
                    )
            if not has_constrained_investments:
                message += "Нет инвестиций для Варианта 2.\n"
            message += f"*Общая дневная прибыль*: ${results_df['Constrained Total Profit (USD)'].iloc[0]:,.2f}\n"
            send_to_telegram(message)
        else:
            logger.warning("Результаты распределения капитала пусты.")
            send_to_telegram("Ошибка: Результаты распределения капитала пусты.")
    except Exception as e:
        logger.error(f"Критическая ошибка в main: {e}")
        send_to_telegram(f"Критическая ошибка: {e}")
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver закрыт.")
        elapsed_time = time.time() - iteration_start_time
        logger.info(f"Итерация завершена за {elapsed_time:.2f} секунд")
        if os.path.exists(lock_file):
            os.remove(lock_file)


if __name__ == "__main__":
    main()