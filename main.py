import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from google.genai.errors import ServerError

# ==========================================
# 1. SEC 財務データ取得関数（B/S・P/L項目）
# ==========================================
def get_buffett_metrics(ticker: str):
    headers = {"User-Agent": "StockAnalyzerApp user@example.com"}
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    res = requests.get(tickers_url, headers=headers)
    res.raise_for_status()
    
    cik_str = None
    for entry in res.json().values():
        if entry["ticker"] == ticker.upper():
            cik_str = str(entry["cik_str"]).zfill(10)
            break
            
    if not cik_str:
        return None

    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    facts_res = requests.get(facts_url, headers=headers)
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
# 2. Gemini API レポート生成（自動リトライ＆フォールバック付き）
# ==========================================
def generate_analysis_report(ticker: str, df_financials: pd.DataFrame, api_key: str):
    client = genai.Client(api_key=api_key)
    
    # リアルタイム市場データ
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
    # 優先モデル順
    candidate_models = ["gemini-3.8-flash", "gemini-3.1-pro-preview""]

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

    raise RuntimeError("利用可能なGeminiモデルが混雑のため応答しませんでした。")

# ==========================================
# 3. メール送信関数
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
# 4. メイン処理（実行エントリーポイント）
# ==========================================
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_pwd = os.environ.get("GMAIL_APP_PASSWORD")

    TARGET_TICKERS = ["INTU"]

    for ticker in TARGET_TICKERS:
        print(f"--- {ticker} の分析を開始 ---")
        df_fin = get_buffett_metrics(ticker)
        if df_fin is None or df_fin.empty:
            print(f"{ticker}: 財務データが取得できませんでした。")
            continue
            
        report = generate_analysis_report(ticker, df_fin, api_key)
        
        subject = f"【自動決算診断】{ticker} 企業価値・バリュー投資レポート ({datetime.now(timezone.utc).strftime('%Y/%m/%d')})"
        send_email(subject, report, sender_email, sender_pwd, sender_email)
        print(f"{ticker} のメール送信が完了しました。")

if __name__ == "__main__":
    main()
