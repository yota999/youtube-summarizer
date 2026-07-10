import os
import re

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

load_dotenv()

st.set_page_config(page_title="YouTube要約くん", page_icon="📺", layout="centered")

st.markdown(
    """
    <style>
    /* 読みやすさ優先: 行間を広め、1行の長さを読みやすい範囲に抑える */
    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stMarkdown p, .stMarkdown li {
        line-height: 1.9;
        font-size: 1rem;
    }
    .stMarkdown ul {
        padding-left: 1.2rem;
    }
    .stMarkdown li {
        margin-bottom: 0.5rem;
    }
    /* 「詳しい内容」のカード見出し */
    .stExpander .detail-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    /* スマホでの余白を少し詰めて画面を有効活用 */
    @media (max-width: 480px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 動画の長さに応じてモデルを自動で使い分ける。
# 予算内に収まるなら質の良いSonnet 5、長すぎる場合は低コストなHaiku 4.5に切り替えて
# 60分程度の動画でも字幕全体を対象にできるようにする。
QUALITY_MODEL = "claude-sonnet-5"
QUALITY_MAX_OUTPUT_TOKENS = 6000
QUALITY_PRICE_INPUT_PER_MTOK = 3.0  # $ / 100万トークン
QUALITY_PRICE_OUTPUT_PER_MTOK = 15.0  # $ / 100万トークン

FALLBACK_MODEL = "claude-haiku-4-5"
FALLBACK_MAX_OUTPUT_TOKENS = 7000
FALLBACK_PRICE_INPUT_PER_MTOK = 1.0  # $ / 100万トークン
FALLBACK_PRICE_OUTPUT_PER_MTOK = 5.0  # $ / 100万トークン

JPY_PER_USD = 160  # 円安方向に多めに見積もった為替レート（目安）

# 予算は動画の長さに応じて変える（10分なら20円、40分なら50円、60分なら70円…の感覚）。
BUDGET_BASE_JPY = 10.0
BUDGET_PER_MINUTE_JPY = 1.0


def budget_jpy_for(duration_minutes: float) -> float:
    return BUDGET_BASE_JPY + BUDGET_PER_MINUTE_JPY * duration_minutes


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> tuple[str, float]:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript(["ja", "ja-JP"])
    except NoTranscriptFound:
        transcript = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )
        if transcript.language_code != "ja" and transcript.is_translatable:
            transcript = transcript.translate("ja")

    fetched = transcript.fetch()
    text = "\n".join(snippet.text for snippet in fetched)

    duration_minutes = 0.0
    if len(fetched) > 0:
        last_snippet = fetched[-1]
        duration_minutes = (last_snippet.start + last_snippet.duration) / 60

    return text, duration_minutes


def build_prompt(transcript_text: str) -> str:
    return f"""以下はYouTube動画の字幕（文字起こし）です。
この内容を日本語で詳しく要約してください。要点を省略しすぎず、具体的な話の流れが分かるようにしてください。

重要: 必ず次の順番で書いてください。「まとめ」を先に書き切ってから、最後に「詳しい内容」を
書いてください。文字数には余裕があるので、「詳しい内容」は遠慮せずにこの要約の中で
一番文字数の多いメインパートとして書いてください。
出力の上限に近づいて文字数が本当に厳しくなった場合だけ、「詳しい内容」の記述量を
減らして調整し、「まとめ」は必ず完成させてください。

出力形式:
## まとめ
（4〜6文で、動画全体の内容・背景と、結論や視聴者へのメッセージをまとめる）

## 詳しい内容
（話題ごとに整理された「小見出し＋箇条書き」の形式で書いてください。ただし、
箇条書きを事実の羅列にするだけだと、動画を見ていない人には文脈が伝わらず
理解しづらくなるので、次のルールを守ってください。
- 話の中に出てきたテーマ・話題・時系列の区切りごとに「### 小見出し」を立てる
- 各小見出しの直後に1〜2文で「これが何の話で、なぜこの話題になったのか」という
  前置き・背景を書いてから箇条書きに入る
- 箇条書きの各項目も、単なる事実だけでなく「なぜそうなったか」「どういう経緯・理由か」
  が分かるように書く（悪い例:「Aさんが〜と発言」／良い例:「〜という経緯があり、
  Aさんは〜と発言」）
- 前提知識が必要な固有名詞・専門用語・人物が出てきたら、（）で簡単に補足する
- 1つの箇条書きは1〜3文程度。長い説明が必要な場合は文を増やしてよいので、
  短くしすぎて意味が飛ばないようにする
- 動画を見ていない人が読んでも、話の流れとその理由がきちんと理解できることを
  最優先にする）

--- 字幕 ---
{transcript_text}
"""


def choose_model_and_fit(client: Anthropic, transcript_text: str, budget_usd: float) -> dict:
    """予算内であれば高品質モデル(Sonnet 5)を字幕全体に使う。
    予算を超える場合は、低コストなHaiku 4.5に切り替えて必要なら字幕を切り詰める。"""
    quality_output_cost = (
        QUALITY_MAX_OUTPUT_TOKENS * QUALITY_PRICE_OUTPUT_PER_MTOK / 1_000_000
    )
    quality_count = client.messages.count_tokens(
        model=QUALITY_MODEL,
        messages=[{"role": "user", "content": build_prompt(transcript_text)}],
    )
    quality_input_cost = quality_count.input_tokens * QUALITY_PRICE_INPUT_PER_MTOK / 1_000_000

    if quality_input_cost + quality_output_cost <= budget_usd:
        return {
            "model": QUALITY_MODEL,
            "max_output_tokens": QUALITY_MAX_OUTPUT_TOKENS,
            "price_input": QUALITY_PRICE_INPUT_PER_MTOK,
            "price_output": QUALITY_PRICE_OUTPUT_PER_MTOK,
            "text": transcript_text,
            "truncated": False,
        }

    # 予算に収まらないほど長い動画は、低コストなモデルに切り替えて全体をカバーする
    fallback_output_cost = (
        FALLBACK_MAX_OUTPUT_TOKENS * FALLBACK_PRICE_OUTPUT_PER_MTOK / 1_000_000
    )
    max_input_tokens = int(
        (budget_usd - fallback_output_cost) / FALLBACK_PRICE_INPUT_PER_MTOK * 1_000_000
    )

    text = transcript_text
    truncated = False
    for _ in range(3):
        fallback_count = client.messages.count_tokens(
            model=FALLBACK_MODEL,
            messages=[{"role": "user", "content": build_prompt(text)}],
        )
        if fallback_count.input_tokens <= max_input_tokens:
            break
        truncated = True
        ratio = max_input_tokens / fallback_count.input_tokens * 0.9  # 少し余裕を持たせる
        text = text[: int(len(text) * ratio)]

    return {
        "model": FALLBACK_MODEL,
        "max_output_tokens": FALLBACK_MAX_OUTPUT_TOKENS,
        "price_input": FALLBACK_PRICE_INPUT_PER_MTOK,
        "price_output": FALLBACK_PRICE_OUTPUT_PER_MTOK,
        "text": text,
        "truncated": truncated,
    }


def parse_summary_sections(summary_text: str) -> dict[str, str]:
    """「## 見出し」で区切られた要約テキストを、見出しごとの辞書に分割する。"""
    sections: dict[str, str] = {}
    current_title = None
    current_lines: list[str] = []

    for line in summary_text.splitlines():
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = heading_match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()

    return sections


def parse_detail_subsections(detail_text: str) -> list[tuple[str, str]]:
    """「### 見出し」で区切られた詳しい内容を (見出し, 本文) のリストに分割する。"""
    subsections: list[tuple[str, str]] = []
    current_title = None
    current_lines: list[str] = []

    for line in detail_text.splitlines():
        heading_match = re.match(r"^###\s+(.+?)\s*$", line)
        if heading_match:
            if current_title is not None:
                subsections.append((current_title, "\n".join(current_lines).strip()))
            current_title = heading_match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        subsections.append((current_title, "\n".join(current_lines).strip()))

    return subsections


def summarize_with_claude(client: Anthropic, config: dict) -> tuple[str, float, bool]:
    prompt = build_prompt(config["text"])
    response = client.messages.create(
        model=config["model"],
        max_tokens=config["max_output_tokens"],
        messages=[{"role": "user", "content": prompt}],
    )
    cost_usd = (
        response.usage.input_tokens * config["price_input"]
        + response.usage.output_tokens * config["price_output"]
    ) / 1_000_000
    cost_jpy = cost_usd * JPY_PER_USD
    summary_text = next(
        block.text for block in response.content if block.type == "text"
    )
    cut_off = response.stop_reason == "max_tokens"
    return summary_text, cost_jpy, cut_off


st.title("📺 YouTube動画 要約くん")
st.write("YouTubeのURLを貼るだけで、動画の内容をAIが要約します。")

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error(
        "Claude APIキーが設定されていません。ローカルで動かす場合はプロジェクトフォルダの "
        "`.env` ファイルに `ANTHROPIC_API_KEY=あなたのキー` を書いてください。"
        "Streamlit Community Cloudの場合はアプリの「Secrets」に設定してください。"
    )
    st.stop()

url = st.text_input("YouTube動画のURL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("要約する", type="primary"):
    if not url:
        st.warning("URLを入力してください。")
        st.stop()

    video_id = extract_video_id(url)
    if not video_id:
        st.error("YouTubeのURLとして認識できませんでした。URLを確認してください。")
        st.stop()

    try:
        with st.spinner("字幕を取得しています..."):
            transcript_text, duration_minutes = fetch_transcript(video_id)
    except TranscriptsDisabled:
        st.error("この動画は字幕が無効になっているため、要約できません。")
        st.stop()
    except VideoUnavailable:
        st.error("動画が見つかりませんでした。URLを確認してください。")
        st.stop()
    except Exception as e:
        st.error(f"字幕の取得に失敗しました: {e}")
        st.stop()

    budget_jpy = budget_jpy_for(duration_minutes)
    budget_usd = budget_jpy / JPY_PER_USD

    try:
        client = Anthropic(api_key=api_key)
        with st.spinner("最適なAIモデルと料金を確認しています..."):
            config = choose_model_and_fit(client, transcript_text, budget_usd)
        with st.spinner("AIが要約を作成しています..."):
            summary, cost_jpy, cut_off = summarize_with_claude(client, config)
    except Exception as e:
        st.error(f"要約の作成に失敗しました: {e}")
        st.stop()

    st.success("要約が完成しました！")

    if config["truncated"]:
        st.warning(
            f"動画が長いため、料金を{budget_jpy:.0f}円以内に抑える都合上、"
            "字幕の前半部分のみを要約対象にしています。"
        )
    if cut_off:
        st.warning(
            "文字数の上限に達したため、「詳しい内容」の後半が途中で切れている可能性が"
            "あります（まとめは完成しています）。"
        )

    sections = parse_summary_sections(summary)

    if not sections:
        st.markdown(summary)

    if sections.get("まとめ"):
        st.markdown("#### 🎯 まとめ")
        st.success(sections["まとめ"])

    if sections.get("詳しい内容"):
        with st.expander("📖 詳しい内容を見る"):
            subsections = parse_detail_subsections(sections["詳しい内容"])
            if subsections:
                for heading, body in subsections:
                    with st.container(border=True):
                        st.markdown(f'<p class="detail-card-title">🔸 {heading}</p>', unsafe_allow_html=True)
                        st.markdown(body)
            else:
                st.markdown(sections["詳しい内容"])

    model_label = (
        "Claude Sonnet 5（高品質モデル）"
        if config["model"] == QUALITY_MODEL
        else "Claude Haiku 4.5（長時間動画向け）"
    )
    st.caption(
        f"動画の長さ: 約{duration_minutes:.0f}分（今回の予算上限: {budget_jpy:.0f}円） / "
        f"使用モデル: {model_label} / 今回のAPI利用料金 目安: "
        f"約{cost_jpy:.2f}円（1ドル={JPY_PER_USD}円換算）"
    )
