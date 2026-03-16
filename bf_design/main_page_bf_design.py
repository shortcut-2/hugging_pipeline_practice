import streamlit as st
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title('🔮나는 전생에 무엇이었을까?🔮')
st.subheader('이름을 입력하고 결과를 받아보세요!')
st.image("https://cdn.pixabay.com/photo/2021/10/26/19/03/moon-6744954_1280.jpg")


def past_world(world):
    return random.choice(world)

world_list = [
    '원시', '고대', '중세', '르네상스', '근대', '근현대', '미래',
    '빙하기', '공룡시대', '신화시대', '대항해시대', '산업혁명시대',
    '전국시대', '냉전시대', '근미래'
    ]

def past_reign(reign):
    return random.choice(reign)

reign_list = [
    '유럽', '동아시아', '아프리카', '아메리카', '남극', '중동', '인도', '동남아시아',
    '북극', '지중해', '잉카제국', '바이킹 땅', '오스만 제국', '몽골 초원', '실크로드',
    '카리브해', '시베리아', '사하라 사막', '아마존 밀림', '히말라야 산중'
    ]

def past_job(job):
    return random.choice(job)

job_list = [
    '펭귄', '사자', '호랑이', '토끼', '코끼리', '독수리', '고래', '늑대', '판다', '낙타',
    '기린', '부엉이', '공룡', '매머드', '라쿤', '돌고래', '여우', '곰', '악어', '공작새',
    '농부', '대장장이', '의사', '광대', '사냥꾼', '기사', '음유시인', '성직자', '귀족', '호위무사',
    '학자', '요리사', '예술가', '학생', '수사관', '나무꾼', '상인', '전사', '일반인', '하인', '선생님', '왕',
    '점쟁이', '해적', '닌자', '마법사', '연금술사', '자객', '탐험가', '스파이', '광부', '어부',
    '목동', '궁수', '마부', '도굴꾼', '무당', '약장수', '거지', '도적', '첩자', '용병',
    '인형사', '불꽃 곡예사', '뱀 조련사', '점성술사', '해몽가', '가면 무희', '독 제조사'
]

def past_personal(personal):
    return random.choice(personal)

personal_list = [
    '친절한', '다정한', '선량한', '온화한', '냉혹한', '용맹한', '비겁한', '성실한', '나태한', '겸손한',
    '오만한', '신중한', '경솔한', '교활한', '대범한', '소심한', '잔인한', '정직한', '냉철한',
    '엉뚱한', '괴짜인', '낙천적인', '염세적인', '충동적인', '집착적인', '무관심한', '허풍스러운',
    '눈물이 많은', '웃음이 많은', '말이 없는', '수다스러운', '욕심 많은', '겁 없는',
    '눈치 없는', '지나치게 진지한', '만사 귀찮은', '자기애 넘치는', '복수심 강한', '과묵한', '느끼한'
]

def past_die(die):
    return random.choice(die)

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

@st.cache_resource
def load_model():
    model_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="bfloat16",
        trust_remote_code=True
        )
    model.eval()
    return tokenizer, model


def generate_story(name, world, reign, personal, job, die_reason):
    tokenizer, model = load_model()

    system_prompt = "당신은 흥미로운 전생 이야기를 쓰는 스토리텔러입니다. 반드시 한국어로만 답하세요."
    prompt = f"""아래 조건에 맞는 전생 이야기를 3~4문장으로 써줘.

    -시대: {world} 시대
    -지역: {reign}
    -직업/정체: {job}
    -성격: {personal}
    -이름: {name}
    -사망 원인: {die_reason}

    -조건:
    1. 반드시 한국어로만 출력
    2. {name}이 주인공인 3인칭 전지적 작가 시점
    3. {die_reason}이 사망 원인으로 자연스럽게 등장
    4. 극적이거나 특별한 사건 1개 필수 추가
    5. 불필요한 설명 없이 바로 이야기 출력
    6. 500토큰 이상 1000토큰 이하 출력"""

    input_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content":system_prompt},
        {"role": "user", "content": prompt}],
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


st.badge('🔍약 30만 개의 결과가 대기 중입니다!')

if 'my_past' not in st.session_state:
    st.session_state.my_past = ''

st.session_state.my_past = st.text_input('✍️이름 입력')

if st.button('✅ 입력 완료') and st.session_state.my_past != '':

    result_world = past_world(world_list)
    result_reign = past_reign(reign_list)
    result_personal = past_personal(personal_list)
    result_job = past_job(job_list)
    result_die = past_die(die_list)
    name = st.session_state.my_past

    with st.spinner('🫳🫳🫳수정구슬이 당신의 전생을 탐색하고 있습니다...', show_time=True):
        time.sleep(3)

    st.success('찾았다‼️')
    st.markdown(f'**{name}**님의 결과는~')
    st.markdown(
        f'**{result_world} 시대**의 **{result_reign}**에 태어난 '
        f'**{result_personal} {result_job}**였습니다.'
    )
    st.markdown(f'사망 사유는 **{result_die}**입니다!')
    st.balloons()

    st.divider()
    st.subheader('📖 전생 스토리')

    with st.spinner('✨ 전생의 기억을 불러오는 중...'):
        story = generate_story(name, result_world, result_reign,
                               result_personal, result_job, result_die)

    st.info(story)

    if st.button('🔮수정구슬 다시 보기'):
        st.session_state.my_past = ''
        st.rerun()
