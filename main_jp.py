import os
import time
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from google.genai.errors import ServerError, ClientError

# ==========================================
# 1. 決算速報の最速検知関数（株探 スクレイピング）
# ==========================================
def has_recent_earnings_filing_jp(ticker_code: str) -> bool:
    """株探の決算速報を監視し、本日または昨日に決算発表があったかを判定"""
    url = f"https://kabutan.jp/stock/news?code={ticker_code}&ncategory=2"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        # ニュース一覧を取得
        news_items = soup.select("table.s_news_list tr")
        today = datetime.now(timezone.utc).astimezone().date()
        
        for item in news_items[:5]: # 最新5件をチェック
            time_td = item.select_one("td time")
            title_a = item.select_one("td a")
            if not time_td or not title_a:
                continue
                
            time_str = time_td.text.strip()
            title = title_a.text.strip()
            
            # 「決算」に関する適時開示かチェック
            if "決算" in title or "業績" in title or "上方修正" in title:
                # "本日 15:00" や "08/14" などの文字列判定
                if "本日" in time_str or "昨日" in time_str:
                    print(f"[{ticker_code}] 最新の決算開示を検知: {title} ({time_str})")
                    return True
                elif "/" in time_str: # 例: 08/14
                    month_day = time_str.split(" ")[0]
                    current_md = today.strftime("%m/%d")
                    yesterday_md = (today - timedelta(days=1)).strftime("%m/%d")
                    if month_day == current_md or month_day == yesterday_md:
                        print(f"[{ticker_code}] 最新の決算開示を検知: {title} ({time_str})")
                        return True
    except Exception as e:
        print(f"[{ticker_code}] 決算情報の取得に失敗しました: {e}")
    return False

# ==========================================
# 2. 日本株 財務データ取得関数（yfinance）
# ==========================================
def get_japan_financials(ticker_code: str):
    """yfinanceを用いて直近数年分の財務データ（B/S・P/L）を取得"""
    symbol = f"{ticker_code}.T" # 日本株は末尾に.Tをつける
    stock = yf.Ticker(symbol)
    
    df_in = stock.financials.T
    if df_in.empty:
        return None
        
    metrics = {}
    if "Total Revenue" in df_in.columns: metrics["売上高"] = df_in["Total Revenue"]
    if "Gross Profit" in df_in.columns: metrics["粗利益"] = df_in["Gross Profit"]
    if "Operating Income" in df_in.columns: metrics["営業利益"] = df_in["Operating Income"]
    if "Net Income" in df_in.columns: metrics["純利益"] = df_in["Net Income"]
    
    df_combined = pd.DataFrame(metrics)
    
    # 粗利率・営業利益率の計算
    if "売上高" in df_combined.columns and "粗利益" in df_combined.columns:
        df_combined["粗利率(%)"] = (df_combined["粗利益"] / df_combined["売上高"]) * 100
    if "売上高" in df_combined.columns and "営業利益" in df_combined.columns:
        df_combined["営業利益率(%)"] = (df_combined["営業利益"] / df_combined["売上高"]) * 100
        
    df_combined = df_combined.sort_index(ascending=True) # 古い順に並び替え
    df_combined.index = df_combined.index.strftime('%Y-%m') # 日付フォーマット
    
    return df_combined

# ==========================================
# 3. Gemini API レポート生成
# ==========================================
def generate_analysis_report_jp(ticker: str, df_financials: pd.DataFrame, api_key: str):
    client = genai.Client(api_key=api_key)
    stock = yf.Ticker(f"{ticker}.T")
    curr_price = stock.info.get("currentPrice", stock.info.get("regularMarketPrice", "N/A"))
    
    prompt = f"""
あなたは世界トップクラスの投資コンサルタントです。
以下の日本株の財務推移をもとに、企業の「経済的な堀（モート）」の維持状況を判定し、バフェット基準に基づくレポートを作成してください。

【厳守するバフェット基準ルール】
1. エコノミック・モートの維持確認：粗利率が40%以上を維持しているか、営業利益率が10%以上を維持しているかを最重視してください。
2. 短期的な減益（先行投資によるもの等）に対する寛容さ：株価が一時的に下落（ノイズ）していても、モートが崩れていなければ「絶好の買い増しチャンス（バーゲンセール）」として判定してください。
3. 専門用語を避けた自然な日本語を用い、結論ファーストで記載してください。

【入力データ】
・銘柄コード: {ticker}
・現在の参考株価: {curr_price} 円
・財務諸表推移（直近数年分）:
{df_financials.to_string()}
"""

    candidate_models = ["gemini-3.8-flash", "gemini-3.1-pro-preview"]

    for model_name in candidate_models:
        for attempt in range(4):
            try:
                print(f"[{model_name}] レポート生成を試行中 (試行回数: {attempt + 1})...")
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except (ServerError, ClientError) as e:
                err_str = str(e)
                if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 15 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    break
            except Exception as e:
                break
    raise RuntimeError("利用可能なGeminiモデルが応答しませんでした。")

# ==========================================
# 4. メール送信関数
# ==========================================
def send_email(subject, body, sender_email, sender_password, receiver_email):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()

# ==========================================
# 5. メイン処理（日本株 保有銘柄）
# ==========================================
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_pwd = os.environ.get("GMAIL_APP_PASSWORD")

    # 保有する日本株の証券コード（4桁）をここに並べます
    TARGET_TICKERS_JP = ["3923", "6036"]  # ラクス, KeePer技研

    for ticker in TARGET_TICKERS_JP:
        print(f"\n==========================================")
        print(f"[{ticker}] 決算発表状況を確認中...")
        
        # 決算速報が直近出ているか判定
        if not has_recent_earnings_filing_jp(ticker):
            print(f"{ticker}: 直近の決算開示（適時開示）はありません。スキップします。")
            continue

        print(f"{ticker}: 新しい決算開示を検知しました！データを取得します。")
        df_fin = get_japan_financials(ticker)
        if df_fin is None or df_fin.empty:
            print(f"{ticker}: 財務データの取得に失敗しました。")
            continue
            
        report = generate_analysis_report_jp(ticker, df_fin, api_key)
        subject = f"【日本株 決算速報】{ticker} バフェット基準 投資判定レポート ({datetime.now(timezone.utc).strftime('%Y/%m/%d')})"
        send_email(subject, report, sender_email, sender_pwd, sender_email)
        print(f"{ticker} のレポート送信が完了しました。")
        
        time.sleep(10)

if __name__ == "__main__":
    main()
