import streamlit as st
import requests
import time
import json
from dotenv import load_dotenv
import os

# Настройка
load_dotenv()
VELORA_API_URL = "https://api.paraswap.io"  # Базовый URL
USER_ADDRESS = "0x5929d94eb4c41d91955154465a58a531ad4cf7b8"  # Ваш адрес
CHAIN_ID = 1  # Ethereum Mainnet
VERSION = "6.2"  # Пробуем v6.2
INCLUDE_DEXS = "CurveV1StableNg,FluidDex,Ekubo"  # Предпочтительные DEX
EXCLUDE_DEXS = "UniswapV3,UniswapV4Pool,FluidDexLite"  # Исключаем нежелательные DEX
MAX_SPLITS = 1  # Строго 1 DEX

def print_and_save_swap_route(price_route):
    """Выводит и возвращает путь свопа из priceRoute."""
    swap_routes = []
    st.write("\n=== Swap Route (edited in v6.2) ===")

    for route in price_route.get("bestRoute", []):
        route_percent = route.get("percent", 0)
        st.write(f"Main Route: {route_percent}%")
        for swap in route.get("swaps", []):
            src_token = swap.get("srcToken", "Unknown")
            dest_token = swap.get("destToken", "Unknown")
            st.write(f"  Swap: {src_token} -> {dest_token}")
            for exchange in swap.get("swapExchanges", []):
                dex = exchange.get("exchange", "Unknown")
                percent = exchange.get("percent", 0)
                src_amount = exchange.get("srcAmount", "N/A")
                dest_amount = exchange.get("destAmount", "N/A")
                pool_addresses = exchange.get("poolAddresses", [])
                st.write(f"    DEX: {dex}, Percent: {percent}%")
                st.write(f"      Source Amount: {src_amount}")
                st.write(f"      Destination Amount: {dest_amount}")
                st.write(f"      Pool Addresses: {pool_addresses}")
                swap_routes.append({
                    "dex": dex,
                    "percent": percent,
                    "srcAmount": src_amount,
                    "destAmount": dest_amount,
                    "poolAddresses": pool_addresses,
                    "srcToken": src_token,
                    "destToken": dest_token
                })
    st.write("===============================\n")

    if not swap_routes:
        st.warning("Warning: No valid swap routes found after editing!")
    return swap_routes

def get_velora_data(src_token, dest_token, src_amount, slippage, max_attempts=3, version=VERSION):
    """Получает данные Velora API с фильтрацией DEX и fallback на v5."""
    attempt = 1
    while attempt <= max_attempts:
        try:
            # Шаг 1: Получение priceRoute
            prices_url = (
                f"{VELORA_API_URL}/prices"
                f"?srcToken={src_token}"
                f"&destToken={dest_token}"
                f"&amount={src_amount}"
                f"&srcDecimals=6"
                f"&destDecimals=6"
                f"&side=SELL"
                f"&network={CHAIN_ID}"
                f"&userAddress={USER_ADDRESS}"
                f"&slippage={slippage}"
                f"&version={version}"
                f"&includeDEXs={INCLUDE_DEXS}"
                f"&excludeDEXs={EXCLUDE_DEXS}"
                f"&excludeIndirectRoutes=true"
                f"&maxSplits={MAX_SPLITS}"
            )
            st.write(f"Attempt {attempt}: Prices URL (v{version}): {prices_url}")
            response = requests.get(prices_url)
            response.raise_for_status()
            data = response.json()
            price_route = data.get("priceRoute")
            if not price_route:
                raise Exception(f"Failed to get price route in v{version}")
            st.write("Price Route (edited):", price_route)

            # Вывод и сохранение пути свопа
            swap_routes = print_and_save_swap_route(price_route)

            # Шаг 2: Построение /transactions
            tx_url = f"{VELORA_API_URL}/transactions/{CHAIN_ID}"
            deadline = int(time.time()) + 48 * 3600  # 48 часов
            payload = {
                "srcToken": src_token,
                "srcDecimals": price_route.get("srcDecimals", 6),
                "destToken": dest_token,
                "destDecimals": price_route.get("destDecimals", 6),
                "srcAmount": src_amount,
                "destAmount": str(price_route["destAmount"]),
                "priceRoute": price_route,
                "userAddress": USER_ADDRESS,
                "deadline": deadline,
                "partner": "anon",
                "ignoreChecks": True
            }
            st.write("Payload for /transactions:", payload)

            response = requests.post(tx_url, json=payload)
            st.write("Response status:", response.status_code)
            st.write("Response text:", response.text)
            response.raise_for_status()
            tx_params = response.json()
            velora_data = tx_params.get("data")

            st.write(f"veloraData (v{version}):", velora_data)

            # Сохранение результатов
            with open(f"swap_routes_v{version}.json", "w") as f:
                json.dump(swap_routes, f, indent=2)
            with open(f"price_route_v{version}.json", "w") as f:
                json.dump(price_route, f, indent=2)
            with open(f"velora_data_v{version}.json", "w") as f:
                json.dump({"velora_data": velora_data}, f, indent=2)

            return {"velora_data": velora_data, "swap_routes": swap_routes}

        except requests.RequestException as e:
            st.error(f"HTTP Error in v{version}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response content: {e.response.text}")
            if attempt < max_attempts:
                st.warning(f"Attempt {attempt} failed. Retrying...")
                attempt += 1
                time.sleep(1)
            elif version == "6.2":
                st.warning(f"Failed after {max_attempts} attempts in v6.2. Falling back to v5...")
                return get_velora_data(src_token, dest_token, src_amount, slippage, max_attempts=3, version="5")
            else:
                st.error(f"Failed after {max_attempts} attempts in v5.")
                return None
        except Exception as e:
            st.error(f"Error in v{version}: {e}")
            if attempt < max_attempts:
                st.warning(f"Attempt {attempt} failed. Retrying...")
                attempt += 1
                time.sleep(1)
            elif version == "6.2":
                st.warning(f"Failed after {max_attempts} attempts in v6.2. Falling back to v5...")
                return get_velora_data(src_token, dest_token, src_amount, slippage, max_attempts=3, version="5")
            else:
                st.error(f"Failed after {max_attempts} attempts in v5.")
                return None

# Streamlit интерфейс
st.title("Velora Swap Data Generator")
st.write("Enter the parameters below to generate veloraData for a swap.")

# Ввод параметров
src_token = st.text_input("Source Token Address (SRC_TOKEN)", value="0xdac17f958d2ee523a2206206994597c13d831ec7")
dest_token = st.text_input("Destination Token Address (DEST_TOKEN)", value="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
src_amount = st.text_input("Source Amount (SRC_AMOUNT)", value="1778069697038")
slippage = st.number_input("Slippage (e.g., 0.0003 for 0.03%)", min_value=0.0, max_value=1.0, value=0.0003, step=0.0001)

# Кнопка для запуска
if st.button("Generate veloraData"):
    st.write(f"Current date and time: 12:55 PM EEST, Sunday, September 21, 2025")
    result = get_velora_data(src_token, dest_token, src_amount, slippage)
    if result:
        st.success(f"Generated veloraData for Aave swapDebt (v{VERSION}): {result['velora_data']}")
        st.write(f"Edited Swap Routes: {result['swap_routes']}")
        st.info(
            f"Results saved to swap_routes_v{VERSION}.json, price_route_v{VERSION}.json, and velora_data_v{VERSION}.json")
    else:
        st.error("Failed to generate veloraData for swap.")