import streamlit as st
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="전생 탐색기",
    page_icon="🔮",
    layout="centered"
)

# ── 커스텀 CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600;700&family=Cinzel:wght@400;600&display=swap');

/* 배경 */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a0a2e 0%, #0d0d1a 50%, #0a0a14 100%);
    min-height: 100vh;
}

/* 별똥별 효과 */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1px 1px at 20% 15%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 25%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 60%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 80%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 70%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 35% 40%, rgba(200,180,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 65% 85%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 5%,  rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 95%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 5%  45%, rgba(180,160,255,0.5) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* 전체 텍스트 */
html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
    color: #e8e0f5 !important;
    font-family: 'Noto Serif KR', serif !important;
}

/* 타이틀 */
.main-title {
    font-family: 'Cinzel', serif !important;
    font-size: 2.6rem;
    font-weight: 600;
    text-align: center;
    background: linear-gradient(135deg, #c9a8f5 0%, #f0c6ff 40%, #a78bfa 70%, #c9a8f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
    text-shadow: none;
    padding-top: 1rem;
}

.sub-title {
    text-align: center;
    color: #a89bc2 !important;
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 0.08em;
}

/* 구슬 이미지 컨테이너 */
.orb-container {
    display: flex;
    justify-content: center;
    margin: 1.5rem 0;
}
.orb-container img {
    border-radius: 50%;
    width: 200px;
    height: 200px;
    object-fit: cover;
    border: 2px solid rgba(167,139,250,0.4);
    box-shadow:
        0 0 30px rgba(167,139,250,0.3),
        0 0 60px rgba(167,139,250,0.15),
        inset 0 0 30px rgba(0,0,0,0.5);
    animation: pulse-orb 3s ease-in-out infinite;
}
@keyframes pulse-orb {
    0%, 100% { box-shadow: 0 0 30px rgba(167,139,250,0.3), 0 0 60px rgba(167,139,250,0.15); }
    50%       { box-shadow: 0 0 45px rgba(196,167,255,0.5), 0 0 90px rgba(167,139,250,0.25); }
}

/* 배지 */
.stBadge, [data-testid="stBadge"] {
    background: rgba(139,92,246,0.2) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    color: #c4b5fd !important;
    border-radius: 20px !important;
}

/* 입력창 */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 10px !important;
    color: #e8e0f5 !important;
    font-family: 'Noto Serif KR', serif !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(196,167,255,0.7) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}
.stTextInput label {
    color: #361969 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em;
}

/* 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
    color: #f5f0ff !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Noto Serif KR', serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em;
    width: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(109,40,217,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    box-shadow: 0 6px 28px rgba(139,92,246,0.45) !important;
    transform: translateY(-1px);
}

/* 결과 카드 */
.result-card {
    background: linear-gradient(135deg, rgba(109,40,217,0.12), rgba(76,29,149,0.08));
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(196,167,255,0.6), transparent);
}
.result-label {
    font-size: 0.78rem;
    color: #9b89b8 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-value {
    font-size: 1.15rem;
    color: #e8d5ff !important;
    font-weight: 600;
}
.die-reason {
    color: #f0abfc !important;
    font-style: italic;
}

/* 스토리 박스 */
.story-box {
    background: rgba(15, 10, 30, 0.6);
    border: 1px solid rgba(167,139,250,0.2);
    border-left: 3px solid rgba(167,139,250,0.6);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    line-height: 2.0;
    font-size: 1rem;
    color: #ddd0f0 !important;
    backdrop-filter: blur(8px);
    position: relative;
}
.story-box::before {
    content: '❝';
    position: absolute;
    top: 0.8rem; left: 1.2rem;
    font-size: 2rem;
    color: rgba(167,139,250,0.25);
    font-family: Georgia, serif;
    line-height: 1;
}

/* 구분선 */
.stDivider hr, hr {
    border-color: rgba(167,139,250,0.2) !important;
    margin: 1.5rem 0 !important;
}

/* 성공/정보 메시지 */
.stSuccess {
    background: rgba(109,40,217,0.15) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 10px !important;
    color: #c4b5fd !important;
}
.stInfo {
    background: rgba(15,10,30,0.5) !important;
    border: 1px solid rgba(167,139,250,0.2) !important;
    border-left: 3px solid rgba(167,139,250,0.5) !important;
    border-radius: 12px !important;
    color: #ddd0f0 !important;
}

/* 스피너 */
.stSpinner p {
    color: #a89bc2 !important;
    font-style: italic;
    letter-spacing: 0.05em;
}

/* 서브헤더 */
h3 {
    font-family: 'Cinzel', serif !important;
    color: #c4b5fd !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(167,139,250,0.2);
    padding-bottom: 0.5rem;
}

/* 섹션 헤더 */
.section-header {
    font-family: 'Cinzel', serif;
    font-size: 1rem;
    color: #a78bfa !important;
    letter-spacing: 0.12em;
    text-align: center;
    margin: 1.5rem 0 1rem 0;
}

/* 스크롤바 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 ───────────────────────────────────────────────────────
world_list = [
    '원시', '고대', '중세', '르네상스', '근대', '근현대', '미래',
    '빙하기', '공룡', '신화', '대항해', '산업혁명',
    '전국', '냉전', '근미래'
]
reign_list = [
    '유럽', '동아시아', '아프리카', '아메리카', '남극', '중동', '인도', '동남아시아',
    '북극', '지중해', '잉카제국', '바이킹 땅', '오스만 제국', '몽골 초원', '실크로드',
    '카리브해', '시베리아', '사하라 사막', '아마존 밀림', '히말라야 산중'
]
job_list = [
    '펭귄', '사자', '호랑이', '토끼', '코끼리', '독수리', '고래', '늑대', '판다', '낙타',
    '기린', '부엉이', '공룡', '매머드', '라쿤', '돌고래', '여우', '곰', '악어', '공작새',
    '농부', '대장장이', '의사', '광대', '사냥꾼', '기사', '음유시인', '성직자', '귀족', '호위무사',
    '학자', '요리사', '예술가', '학생', '수사관', '나무꾼', '상인', '전사', '일반인', '하인',
    '선생님', '왕', '점쟁이', '해적', '닌자', '마법사', '연금술사', '자객', '탐험가', '스파이',
    '광부', '어부', '목동', '궁수', '마부', '도굴꾼', '무당', '약장수', '거지', '도적', '첩자',
    '용병', '인형사', '불꽃 곡예사', '뱀 조련사', '점성술사', '해몽가', '가면 무희', '독 제조사'
]
personal_list = [
    '친절한', '다정한', '선량한', '온화한', '냉혹한', '용맹한', '비겁한', '성실한', '나태한', '겸손한',
    '오만한', '신중한', '경솔한', '교활한', '대범한', '소심한', '잔인한', '정직한', '냉철한',
    '엉뚱한', '괴짜인', '낙천적인', '염세적인', '충동적인', '집착적인', '무관심한', '허풍스러운',
    '눈물이 많은', '웃음이 많은', '말이 없는', '수다스러운', '욕심 많은', '겁 없는',
    '눈치 없는', '지나치게 진지한', '만사 귀찮은', '자기애 넘치는', '복수심 강한', '과묵한', '느끼한'
]
die_list = [
    '발을 헛디딤', '전투 중 사망', '자다가 사망', '익사', '암살', '역병', '굶주림',
    '맹수의 습격', '노환', '저주', '독살', '낙뢰', '동사', '화형', '상심', '식도 폐쇄',
    '석화', '과로사', '실종', '과식', '화병', '반역으로 처벌', '웃다가 사망',
    '발명품 오작동', '자기 함정에 걸림', '지도를 거꾸로 봄', '독버섯 오해', '폭발 사고',
    '술에 취해 바다에 빠짐', '벌집 건드림', '연금술 실험 실패', '본인이 만든 독에 중독',
    '엉뚱한 사람과 결투', '무대에서 추락', '책더미에 깔림', '항해 중 조난',
    '용을 잡으러 갔다가 역관광', '왕에게 직언하다 처형', '너무 빨리 달리다 사망',
    '하품하다 턱 빠짐 후 합병증', '혼자 웃다가 숨막힘', '보물지도 분실 후 충격사',
    '발명한 기계에 깔림', '시식 담당이었음', '이름 모를 이유로 사라짐', '료이키 텐카이'
]


# ── 모델 ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="bfloat16", trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


def generate_story(name, world, reign, personal, job, die_reason):
    tokenizer, model = load_model()

    system_prompt = "당신은 흥미로운 전생 이야기를 쓰는 스토리텔러이다. 반드시 한국어로만 답한다."
    prompt = f"""아래 조건에 맞는 전생 이야기를 작성

- 시대: {world} 시대
- 지역: {reign}
- 직업/정체: {job}
- 성격: {personal}
- 이름: {name}
- 사망 원인: {die_reason}

조건:
1. 반드시 한국어로만 출력
2. {name}이 주인공인 3인칭 전지적 작가 시점
3. {die_reason}이 사망 원인으로 자연스럽게 등장
4. 극적이거나 특별한 사건 1개 필수 추가
5. 불필요한 설명 없이 바로 이야기 출력
6. 500토큰 이상 1000토큰 이하 출력"""

    input_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.8,
            temperature=0.4,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    story = tokenizer.decode(
        output[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()
    return story


# ── UI ───────────────────────────────────────────────────────────

# 헤더
st.markdown('<div class="main-title">🔮 나는 전생에 무엇이었을까?</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">수정구슬이 당신의 전생을 알려드립니다.</div>', unsafe_allow_html=True)

# 구슬 이미지
st.markdown("""
<div class="orb-container">
  <img src="https://cdn.pixabay.com/photo/2021/10/26/19/03/moon-6744954_1280.jpg" />
</div>
""", unsafe_allow_html=True)

st.badge('🔍 약 30만 개의 전생이 대기 중')

st.markdown("---")

# 입력
if 'my_past' not in st.session_state:
    st.session_state.my_past = ''

name_input = st.text_input('✍️ 이름을 입력하세요', placeholder='예: 홍길동')
st.session_state.my_past = name_input

if st.button('🔮 전생 탐색 시작'):
    if st.session_state.my_past == '':
        st.warning('이름을 먼저 입력해주세요.')
    else:
        result_world    = random.choice(world_list)
        result_reign    = random.choice(reign_list)
        result_personal = random.choice(personal_list)
        result_job      = random.choice(job_list)
        result_die      = random.choice(die_list)
        name            = st.session_state.my_past

        with st.spinner('🫳 수정구슬이 당신의 전생을 탐색하고 있습니다...'):
            time.sleep(3)

        st.success('✨ 전생의 기억을 찾았습니다!')

        # 결과 카드
        st.markdown(f"""
        <div class="result-card">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.2rem;">
                <div>
                    <div class="result-label">✦ 이름</div>
                    <div class="result-value">{name}</div>
                </div>
                <div>
                    <div class="result-label">✦ 시대</div>
                    <div class="result-value">{result_world} 시대</div>
                </div>
                <div>
                    <div class="result-label">✦ 지역</div>
                    <div class="result-value">{result_reign}</div>
                </div>
                <div>
                    <div class="result-label">✦ 직업 / 정체</div>
                    <div class="result-value">{result_personal} {result_job}</div>
                </div>
            </div>
            <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid rgba(167,139,250,0.15);">
                <div class="result-label">✦ 사망 사유</div>
                <div class="result-value die-reason">⚰️ {result_die}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        # 스토리
        st.markdown('<div class="section-header">— 전생의 기억 —</div>', unsafe_allow_html=True)

        with st.spinner('📜 전생의 이야기를 불러오는 중...'):
            story = generate_story(name, result_world, result_reign,
                                   result_personal, result_job, result_die)

        st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button('🔄 다시 탐색하기'):
            st.session_state.my_past = ''
            st.rerun()
