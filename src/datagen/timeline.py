"""Persona timeline generator.

Creates a 60-day event timeline with:
- Recurring topic anchors (rose_price_concern, grandma_park_visit, …)
  distributed across days so the retriever has multi-day memory chains.
- One-off events (supply shocks, unusual visits).
- Explicit links_back_to_day / links_back_historical so each day's
  utterances can reference earlier days or persona.historical_events
  (DATASET §5.1, IMPL_DETAILS §2.3).

Deterministic given a seed — same seed, same persona → same timeline.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Event:
    day: int
    time: str
    event_type: str
    trigger: str
    topic_key: str
    links_back_to_day: list[int] = field(default_factory=list)
    links_back_historical: list[int] = field(default_factory=list)  # days_ago values
    details: dict[str, Any] = field(default_factory=dict)


# Topic pools per persona — each topic will recur 3–5 times across 60 days.
# Additional one-off events are sampled from the same pool with lower frequency.
_TOPIC_POOLS: dict[str, list[dict[str, Any]]] = {
    "yejin_florist": [
        {"key": "rose_price_concern", "type": "pricing_deliberation",
         "triggers": ["도매가 +800원 인상 통보", "경쟁 꽃집 가격 인상 소문",
                      "단골이 가격 넌지시 물어봄"], "recur": 5},
        {"key": "grandma_park_visit", "type": "customer_interaction",
         "triggers": ["박 할머니 수요일 오전 방문", "박 할머니 빈손 귀가",
                      "박 할머니 손녀 선물 주문"], "recur": 8},
        {"key": "stock_shortage_pink_rose", "type": "stock_issue",
         "triggers": ["핑크 장미 재고 0", "발주 잘못해서 과잉재고"], "recur": 3},
        {"key": "husband_conversation", "type": "personal_rumination",
         "triggers": ["남편에게 이번 달 매출 말 못함", "남편이 매출 물어봄"], "recur": 4},
        {"key": "sns_reel_pressure", "type": "sns_pressure",
         "triggers": ["릴스 업로드 타이밍 놓침", "팔로워 증가 없음"], "recur": 4},
        {"key": "office_kim_order", "type": "customer_interaction",
         "triggers": ["김 과장 정기 납품 리셉션용", "김 과장 이번 달 캔슬"], "recur": 4},
        {"key": "weekend_event_prep", "type": "operational",
         "triggers": ["주말 어버이날 대비", "발주 결정 고민"], "recur": 4},
    ],
    "minseok_cafe": [
        {"key": "bean_price_concern", "type": "pricing_deliberation",
         "triggers": ["에티오피아 환율 급등", "콜롬비아 공급처 단가 조정"], "recur": 5},
        {"key": "franchise_competition", "type": "competitor_threat",
         "triggers": ["길 건너 프랜차이즈 프로모션", "오전 손님 빠진 듯 느낌"], "recur": 5},
        {"key": "morning_dev_daily", "type": "customer_interaction",
         "triggers": ["박 개발자 오전 8시 테이크아웃", "원두 추천 요청"], "recur": 10},
        {"key": "studio_jung_subscription", "type": "customer_interaction",
         "triggers": ["정 실장 콜드브루 정기 배송", "정 실장 추가 원두 주문"], "recur": 4},
        {"key": "elderly_kim_sunday", "type": "customer_interaction",
         "triggers": ["김 선생님 부부 일요일 오후", "디카페인 재고 확인"], "recur": 5},
        {"key": "roasting_inconsistency", "type": "quality_issue",
         "triggers": ["습도 급변해서 로스팅 발현 차이", "원두 향미 일관성 의문"], "recur": 4},
        {"key": "daughter_school_fee", "type": "personal_rumination",
         "triggers": ["딸 학원비 고지서", "아내 대신 지출 메모"], "recur": 3},
        {"key": "table_turnover_debate", "type": "operational",
         "triggers": ["장시간 체류 손님 vs 회전율", "주말 테이블 부족"], "recur": 3},
    ],
    "sunhee_hair": [
        {"key": "teacher_choi_cycle", "type": "customer_interaction",
         "triggers": ["최 선생님 6주차 방문 예정", "최 선생님 매직 문의"], "recur": 6},
        {"key": "mom_lim_weekly", "type": "customer_interaction",
         "triggers": ["임 어머님 3주 주기 새치 커버", "임 어머님 사위 험담"], "recur": 10},
        {"key": "bride_ahn_wedding_prep", "type": "special_event",
         "triggers": ["안 신부 리허설 예약", "업스타일 시안 재검토"], "recur": 5},
        {"key": "no_show_stress", "type": "operational",
         "triggers": ["주말 노쇼 1건 발생", "예약봇 알림 안 감"], "recur": 4},
        {"key": "perm_chemical_price", "type": "pricing_deliberation",
         "triggers": ["펌제 도매가 +15%", "염색제 제조사 변경 고려"], "recur": 4},
        {"key": "son_tuition", "type": "personal_rumination",
         "triggers": ["아들 등록금 마감일", "아들 이번 학기 장학금 확인"], "recur": 3},
        {"key": "lunch_break_fail", "type": "operational",
         "triggers": ["점심 거르고 3시까지 연속 시술", "화장실 타이밍 놓침"], "recur": 4},
        {"key": "past_mistake_worry", "type": "personal_rumination",
         "triggers": ["작년 매직 실패 손님 다시 오심", "비슷한 시술 앞두고 긴장"], "recur": 2},
    ],
}


def _load_persona(persona_path: Path) -> dict[str, Any]:
    return yaml.safe_load(persona_path.read_text(encoding="utf-8"))


def _pick_time(rng: random.Random, event_type: str) -> str:
    pools = {
        "pricing_deliberation": ["22:40", "23:15", "23:30", "06:20"],
        "customer_interaction": ["10:15", "14:30", "16:00", "17:45"],
        "stock_issue": ["06:00", "07:30", "21:00"],
        "personal_rumination": ["23:00", "23:50", "00:20", "05:45"],
        "sns_pressure": ["22:00", "23:30"],
        "operational": ["09:00", "18:00", "20:30"],
        "competitor_threat": ["08:30", "22:15"],
        "quality_issue": ["11:00", "15:00"],
        "special_event": ["14:00", "19:00"],
    }
    return rng.choice(pools.get(event_type, ["22:00"]))


def generate_timeline(
    persona: dict[str, Any],
    *,
    days: int = 60,
    seed: int = 42,
) -> list[Event]:
    persona_id = persona["persona"]["id"]
    pool = _TOPIC_POOLS.get(persona_id)
    if pool is None:
        raise ValueError(f"No topic pool defined for persona id={persona_id!r}")

    rng = random.Random(seed)
    historical = persona.get("historical_events", [])

    # Place each recurring topic on N distinct days across the horizon.
    day_events: dict[int, list[Event]] = {d: [] for d in range(1, days + 1)}
    topic_day_map: dict[str, list[int]] = {}

    for topic in pool:
        recur = min(int(topic["recur"]), days)
        if recur < 1:
            continue
        chosen_days = sorted(rng.sample(range(1, days + 1), recur))
        topic_day_map[topic["key"]] = chosen_days
        for i, day in enumerate(chosen_days):
            links_back = [d for d in chosen_days[:i]]
            links_hist = []
            # Some topics should also nod at the persona's historical_events
            # (days_ago anchors). Sample up to 1 with probability 0.3.
            if historical and rng.random() < 0.3:
                pick = rng.choice(historical)
                links_hist = [int(pick["days_ago"])]
            ev = Event(
                day=day,
                time=_pick_time(rng, topic["type"]),
                event_type=topic["type"],
                trigger=rng.choice(topic["triggers"]),
                topic_key=topic["key"],
                links_back_to_day=links_back,
                links_back_historical=links_hist,
                details={"recur_index": i, "recur_total": recur},
            )
            day_events[day].append(ev)

    # Flatten, sort by day then time.
    all_events: list[Event] = []
    for day in sorted(day_events):
        sorted_ev = sorted(day_events[day], key=lambda e: e.time)
        all_events.extend(sorted_ev)

    return all_events


def save_timeline(events: list[Event], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(yaml_line(asdict(ev)) + "\n")


def yaml_line(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, ensure_ascii=False)


def load_timeline(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("persona_yaml", type=Path)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    persona = _load_persona(args.persona_yaml)
    events = generate_timeline(persona, days=args.days, seed=args.seed)
    save_timeline(events, args.out)
    print(f"{len(events)} events for {persona['persona']['id']} -> {args.out}", file=sys.stderr)
