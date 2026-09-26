import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from google.genai.errors import ServerError

# ==========================================
# 0. 共通設定・SEC CIK取得
# ==========================================
SEC_HEADERS = {"User-Agent": "StockAnalyzerApp user@example.com"}

def get_cik(ticker: str):
    """ティッカーシンボルからSEC CIKコードを取得"""
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    res = requests.get(tickers_url, headers=SEC_HEADERS)
    res.raise_for_status()
    for entry in res.json().values():
        if entry["ticker"] == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None

# ==========================================
# 1. 決算発表（10-K / 10-Q 提出）判定関数
# ==========================================
def has_recent_earnings_filing(cik_str: str, days_threshold: int = 2) -> bool:
    """直近指定日数以内に 10-K または 10-Q が提出されたかを判定"""
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    res = requests.get(url, headers=SEC_HEADERS)
    res.raise_for_status()
    
    recent_filings = res.json().get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    
    today = datetime.now(timezone.utc).date()
    
    for form, date_str in zip(forms, filing_dates):
        if form in ["10-K", "10-Q"]:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_ago = (today - filing_date).days
            if 0 <= days_ago <= days_threshold:
                print(f"決算提出を検知: Form {form}, 提出日: {date_str} ({days_ago}日前)")
                return True
            else:
                break
    return False

# ==========================================
# 2. SEC 財務データ取得関数（B/S・P/L項目）
# ==========================================
def get_buffett_metrics(ticker: str, cik_str: str):
    """SECから過去10年分の主要財務諸表データを取得結合"""
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    facts_res = requests.get(facts_url, headers=SEC_HEADERS)
    facts_res.raise_for_status()
    us_gaap = facts_res.json()["facts"].get("us-gaap", {})

    def extract_yearly_data(tag_name):
        if tag_name not in us_gaap:
            return pd.DataFrame()
        units_key = list(us_gaap[tag_name]["units"].keys())[0]
        dataList = us_gaap[tag_name]["units"][units_key]
        df = pd.DataFrame([d for d in dataList if d.get("form") == "10-K"])
        if df.empty:
            return df
        return df.drop_duplicates(subset=["fy"], keep="last")[["fy", "val"]].rename(columns={"val": tag_name})

    tags = [
        "NetIncomeLoss", 
        "PaymentsToAcquirePropertyPlantAndEquipment", 
        "OperatingIncomeLoss", 
        "InterestExpense",
        "GrossProfit",
        "EarningsPerShareDiluted",
        "CashAndCashEquivalentsAtCarryingValue",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations"
    ]

    df_combined = pd.DataFrame(columns=["fy"])
    for tag in tags:
        df_tag = extract_yearly_data(tag)
        if not df_tag.empty:
            df_combined = pd.merge(df_combined, df_tag, on="fy", how="outer")
            
    df_combined = df_combined.sort_values("fy").dropna(subset=["NetIncomeLoss"]).tail(10)

    # 過去株式数の推計 (純利益 / EPS)
    if "EarningsPerShareDiluted" in df_combined.columns and "NetIncomeLoss" in df_combined.columns:
        df_combined["推計株式数"] = df_combined.apply(
            lambda r: r["NetIncomeLoss"] / r["EarningsPerShareDiluted"] if pd.notnull(r["EarningsPerShareDiluted"]) and r["EarningsPerShareDiluted"] != 0 else None,
            axis=1
        )

    # 期末株価の取得結合
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10y", interval="1mo")
        prices = []
        for _, row in df_combined.iterrows():
            fy = int(row["fy"])
            yearly = hist[hist.index.year == fy]
            prices.append(round(yearly.iloc[-1]["Close"], 2) if not yearly.empty else None)
        df_combined["実際の株価(期末)"] = prices
    except Exception:
        df_combined["実際の株価(期末)"] = None

    rename_dict = {
        "fy": "会計年度",
        "NetIncomeLoss": "純利益",
        "PaymentsToAcquirePropertyPlantAndEquipment": "資本的支出",
        "OperatingIncomeLoss": "営業利益",
        "InterestExpense": "支払利息",
        "GrossProfit": "粗利益",
        "EarningsPerShareDiluted": "EPS",
        "CashAndCashEquivalentsAtCarryingValue": "現金および現金同等物",
        "LongTermDebtNoncurrent": "長期有利子負債",
        "LongTermDebtAndCapitalLeaseObligations": "長期負債・リース債務"
    }
    return df_combined.rename(columns=rename_dict)

# ==========================================
# 3. Gemini API レポート生成
# ==========================================
def generate_analysis_report(ticker: str, df_financials: pd.DataFrame, api_key: str):
    client = genai.Client(api_key=api_key)
    stock = yf.Ticker(ticker)
    curr_price = stock.info.get("currentPrice", stock.info.get("regularMarketPrice", "N/A"))
    shares = stock.info.get("sharesOutstanding", "N/A")

    prompt = f"""
あなたは世界トップクラスの投資コンサルタントです。
以下の入力データ（過去10年分の財務推移と最新市場データ）をもとに、企業の「経済的な堀」の維持状況を判定し、時系列での想定株価を算出してレポートを作成してください。
専門用語を避けた自然な日本語を用い、ROICなどの専門用語には簡単な注釈を添え、結論ファーストで記載してください。

【追加タスク：時系列での適正株価算出と推移表示】
※絶対に単純平均は行わないでください。最も正確な「観点A」を主軸（コア）とし、「観点B」「観点C」は市場評価レンジとして提示してください。
■ 観点A：本質的価値（エコノミック・プロフィット法）
・事業価値 = 投下資産 + (次期EP ÷ (WACC - 長期成長率g))
・エコノミック・プロフィット(EP) = 投下資産 × (ROIC - WACC)
・株主価値 = 事業価値 + 現金等 - 有利子負債。想定株価 = 株主価値 ÷ 発行済株式数。
■ 観点B：収益性マルチプル（PER・PEGレシオ法）
・急成長企業はPEG=1.0基準で妥当PERを設定し算出。
■ 観点C：純資産・売上高マルチプル（PBR/PSR法）

【厳守出力ルール】
1. 株式数の単位換算：Billion等の数字はそのまま「〇〇億株」と直訳せず、正確な日本の単位（例：2.8億株など）に換算すること。
2. 数値の完全一致：「1. 総合結論」の最新想定株価・参考株価と、「時系列トレンド推移表」最終行(現在)の数値は完全に一致させること。

【入力データ】
・企業名（ティッカー）: {ticker}
・現在の参考株価: ${curr_price}
・最新発行済株式数: {shares}
・財務諸表推移（過去10年）:
{df_financials.to_string()}
"""

    candidate_models = ["gemini-3.8-flash", "gemini-3.1-pro-preview"]

    for model_name in candidate_models:
        for attempt in range(3):
            try:
                print(f"[{model_name}] レポート生成を試行中 (試行回数: {attempt + 1})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except ServerError as e:
                print(f"503 混雑エラー検知: {e}. 10秒待機して再試行します...")
                time.sleep(10)
            except Exception as e:
                print(f"予期しないエラー ({model_name}): {e}")
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
# 5. メイン処理（保有7銘柄の決算判定）
# ==========================================
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_pwd = os.environ.get("GMAIL_APP_PASSWORD")

    # 保有7銘柄
    TARGET_TICKERS = ["BR", "CDNS", "CPRT", "GOOGL", "INTU", "NVDA", "ZTS"]

    for ticker in TARGET_TICKERS:
        print(f"\n==========================================")
        print(f"[{ticker}] 決算発表状況を確認中...")
        
        cik_str = get_cik(ticker)
        if not cik_str:
            print(f"{ticker}: CIKコードの取得に失敗しました。")
            continue
            
        # 直近2日以内に 10-K または 10-Q の開示があるか判定
        if not has_recent_earnings_filing(cik_str, days_threshold=2):
            print(f"{ticker}: 直近2日以内の決算開示（10-K/10-Q）はありません。スキップします。")
            continue

        print(f"{ticker}: 新しい決算開示を確認しました。分析レポートを作成します。")
        df_fin = get_buffett_metrics(ticker, cik_str)
        if df_fin is None or df_fin.empty:
            print(f"{ticker}: 財務データの取得に失敗しました。")
            continue
            
        report = generate_analysis_report(ticker, df_fin, api_key)
        subject = f"【決算速報診断】{ticker} 企業価値・バリュー投資レポート ({datetime.now(timezone.utc).strftime('%Y/%m/%d')})"
        send_email(subject, report, sender_email, sender_pwd, sender_email)
        print(f"{ticker} のレポート送信が完了しました。")

if __name__ == "__main__":
    main()
