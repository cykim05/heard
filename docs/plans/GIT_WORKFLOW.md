# Heard — Git Workflow & Documentation Protocol

> 이 문서는 Claude Code가 **작업하면서 지켜야 하는 git·문서화 규약**입니다. 목적은 두 가지: (1) 나중에 본인·다른 사람이 repo를 보고 "튜토리얼처럼" 재현할 수 있게 만들기, (2) 모든 중간 산출물을 commit으로 잠가서 분실 방지.

---

## 0. Core Principles

1. **Real-time commit**: 작업 단위가 완료되는 **즉시** commit + push. 하루치 모아서 commit 하지 말 것.
2. **Narrative documentation**: 코드만 commit하지 말고 **"왜 이렇게 했는지"**를 `docs/lab_notebook/` 에 Markdown으로 동시에 기록.
3. **No Claude attribution**: commit author, PR author, code comment, README 어디에도 Claude/Anthropic 언급 금지. AI 사용 사실 자체를 숨기라는 게 아니라, **code authorship**은 사용자 한 사람으로 유지.
4. **Tutorial-ready**: 처음 이 repo를 보는 사람이 README + `docs/` 만 읽고 전체 파이프라인을 재현할 수 있어야 함.

---

## 1. Author Identity & Contributor Policy

### 1.1 Git config (repo 초기화 시 반드시)

```bash
cd heard-bench
git config user.name "Chanyoung Kim"
git config user.email "<user-provided email>"
# GitHub 계정과 동일하게 맞출 것
```

### 1.2 No co-author / no Claude attribution

모든 commit은 다음 규칙을 따름:

- **Commit message에 다음 문구 절대 금지**:
  - `Co-Authored-By: Claude <...>`
  - `🤖 Generated with Claude Code`
  - `with assistance from AI` / `AI-generated` / `Anthropic`
- **Code comment에도 금지**: `# Written by Claude`, `// AI-generated`, 등.
- **README·docs에도 금지**: 도구 언급은 가능하지만 authorship으로는 올리지 않음.

### 1.3 Auto-scrubbing (Claude Code가 스스로 지켜야 함)

Claude Code는 commit 생성 직전에 message를 self-check하여 위 문구가 포함되면 제거 후 commit. 만약 이미 들어간 commit이 있다면:

```bash
# 최근 1개 commit message 수정
git commit --amend -m "<cleaned message>"

# 여러 commit이 섞여 있으면
git rebase -i HEAD~N
# → pick → reword로 변경해서 수정
```

### 1.4 Pre-commit hook (권장, 자동화)

`.githooks/pre-commit` 에 다음 script를 두고 `git config core.hooksPath .githooks` 설정:

```bash
#!/usr/bin/env bash
# Block commits that mention Claude / AI co-authorship
msg_file="$1"
if [ -z "$msg_file" ]; then
  msg_file=".git/COMMIT_EDITMSG"
fi
if grep -iE "claude|anthropic|co-authored-by.*claude|🤖" "$msg_file" > /dev/null 2>&1; then
  echo "❌ commit message contains forbidden AI attribution. Edit and retry."
  exit 1
fi
exit 0
```

Claude Code가 repo 초기화 직후 이 hook을 만들고 실행 권한을 부여할 것 (`chmod +x .githooks/pre-commit`).

---

## 2. Commit Frequency & Granularity

### 2.1 When to commit

**즉시 commit**:
- 하나의 파일·모듈이 동작하는 최소 단위로 완성됐을 때
- 실험 결과가 생성됐을 때 (결과 파일도 반드시 커밋)
- 설계 결정이 바뀌었을 때 (README·docs 업데이트 포함)
- 30분 이상 작업한 경우는 무조건 중간 commit

**금지 사항**:
- 하루치 몰아서 commit
- 여러 논리적 변경을 하나의 commit으로 묶기
- 미완성 코드를 push하지 않고 로컬에만 방치

### 2.2 Commit 크기 목표

- **50–300 line diff** 를 한 commit의 정상 범위로.
- 1000 line 넘으면 쪼갤 것 (데이터 dump는 예외).
- **대형 데이터(>10MB)는 git-lfs**로 분리. 일반 diff와 섞지 말 것.

### 2.3 Push 주기

- 매 commit 후 `git push` (remote 손실 시 복구 가능).
- CI가 없으니 push는 자유. merge conflict는 거의 발생 안 함 (단일 작업자).

---

## 3. Commit Message Convention

### 3.1 Format

```
<type>(<scope>): <summary (50자 이내, 한글 OK)>

<body (optional, 왜 이렇게 했는지·무엇을 시도했는지, 줄당 72자 권장)>

<footer (optional, 참조 이슈·PR·외부 링크)>
```

### 3.2 Types

| type | 사용 시점 | 예시 |
|---|---|---|
| `feat` | 새 기능·모듈 추가 | `feat(node): add relation graph indexing` |
| `fix` | 버그 수정 | `fix(retriever): handle empty query edge case` |
| `data` | 데이터 생성·수정 | `data(ko_native): generate 40 yejin scenarios` |
| `exp` | 실험 실행·결과 저장 | `exp(sweep): run kanana-2.1b no-NODE baseline` |
| `docs` | 문서화 | `docs(lab): record reasoning for probe ratio` |
| `refactor` | 동작 변경 없는 구조 개선 | `refactor(eval): split runner into phases` |
| `chore` | 설정·의존성·잡일 | `chore: add bitsandbytes to requirements` |

### 3.3 Good vs Bad

✅ **Good**:
```
exp(sweep): run 9-config SUT sweep on ko_native

All 5 SUTs x (fp16+int4 where applicable) x 3 conditions
complete. Results in experiments/20260426_sweep_1/.
Kanana-2.1B int4 hit OOM at context=12k; switched to
flash-attn 2 and retried. Logged in lab notebook.
```

❌ **Bad**:
```
updates
```
```
done some work 🤖
```
```
Implemented the retrieval module

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 4. Documentation Philosophy — "Tutorial-Ready Repository"

이 repo는 **논문 supplementary + 오픈소스 프로젝트 + 본인의 미래 참고자료**의 3중 역할을 합니다. 따라서 다음 4종 문서를 유지:

### 4.1 `README.md` — 관문

구성:
1. Project one-liner (2–3줄)
2. Motivation & key claim (Part 1 프로포절 요약)
3. **Quickstart** (5분 안에 결과 하나 재현)
4. Repo layout 그림
5. How to reproduce (섹션 별 링크)
6. Dataset release (HF 링크)
7. Citation
8. License

**반드시 tutorial 형식**. "이 프로젝트는…" 식 설명보다 "다음 명령어를 실행하면 X가 생성됩니다" 식 실행 가능한 문장 중심.

### 4.2 `docs/lab_notebook/` — 연구 일지 (**핵심**)

**매일 1개 이상 파일**을 날짜별로 쌓는 narrative 기록:

```
docs/lab_notebook/
├── 2026-04-23_day1_setup.md
├── 2026-04-24_day2_data_generation.md
├── 2026-04-25_day3_filtering_and_sweep.md
├── 2026-04-26_day4_eval_and_report.md
└── index.md
```

각 파일 구성:
```markdown
# Day 2 — Data Generation (2026-04-24)

## Goals today
- [ ] Utterance corpus 3 personas, 60 days
- [ ] Scenario candidates 150
- [ ] Adversarial filter pass 1

## What I did
### 09:00 Persona prompt tuning
이유: Yejin 프롬프트에 "완벽주의" 특성이 반영이 약해서
trial 1은 너무 중립적 말투. "남편에게 약한 소리 못함" 구체화함.

commit: `5a3f1c2 feat(datagen): strengthen yejin perfectionism cue`

### 10:30 Generator rotation bug
GPT-4o가 첫 5 items에서만 사용되고 이후 skip되던 버그.
원인: rotation counter가 persona 내에서 reset 안 됨.
수정 후 distribution 균일함 확인.

commit: `9b2e403 fix(datagen): reset generator counter per persona`

## Results / artifacts
- `data/raw/utterances.jsonl` (2,430 utterances)
- `data/scenarios/candidates.jsonl` (147)

## Numbers
| metric | value |
|---|---|
| Utterances generated | 2,430 |
| API cost (USD) | 4.72 |
| Filter survival rate | 68/147 = 46.3% |

## Decisions made
- Scenario count 100 → 유지 (filter survival 충분)
- Judge 2개 agreement check는 Day 3로 연기

## Tomorrow
- [ ] Human spot check 20 items (~1h)
- [ ] Full SUT sweep start
```

이 lab notebook이 리포트의 Methods 섹션 초안 소스가 됩니다. 기억에 의존하지 말고 **실시간으로** 적을 것.

### 4.3 `docs/guides/` — 튜토리얼 (재현 가능성)

| 파일 | 내용 |
|---|---|
| `01_environment_setup.md` | CUDA·PyTorch·bitsandbytes 설치, L40S 전용 팁 |
| `02_data_generation.md` | API 키 설정, 생성 명령, 예상 비용·시간 |
| `03_running_experiments.md` | SUT 다운로드, sweep 실행, 로그 위치 |
| `04_evaluation_and_figures.md` | Metric 해석, figure 재생성 |
| `05_hf_upload.md` | HuggingFace 업로드 절차 |

각 파일은 **복붙 가능한 명령어 + 예상 출력**을 포함한 step-by-step.

### 4.4 `docs/decisions/` — ADR (Architectural Decision Records)

중요한 설계 결정 하나당 파일 1개:

```
docs/decisions/
├── 0001-why-kanana-and-exaone-as-sut.md
├── 0002-adversarial-filter-threshold.md
├── 0003-reflective-rubric-design.md
└── 0004-hf-dataset-schema.md
```

각 ADR 구성:
```markdown
# ADR 0002 — Adversarial Filter Threshold

## Status
Accepted (2026-04-25)

## Context
초기 후보 150개를 no-NODE baseline으로 검증. 3개 SUT x 3 trials.

## Decision
모든 9 runs가 fail한 item만 최종 통과. "6/9 이상 fail"로 완화하면
60 items 추가 확보 가능했으나, baseline이 0%에 수렴하는 극명한
대비를 우선시.

## Consequences
- Final item count 68 (목표 100 미달)
- 리포트에서 "harder subset" 으로 positioning 가능
- 재생성 필요 시 candidate 추가 200개 필요

## Alternatives considered
- 7/9 threshold → 92 items 가능, but weaker delta
- 모델 1개만으로 filter → 빠르지만 over-fit 우려
```

**ADR는 마감 전에 쓰는 게 아니라, 결정하는 순간 쓴다**. 나중에 Discussion 섹션을 바로 생성할 수 있는 소스.

---

## 5. Experiment Logging (재현성 핵심)

### 5.1 모든 실험 = 하나의 디렉터리

```
experiments/
└── 20260426_1430_ko_native_sweep/
    ├── config.yaml           # 실험 설정 전체 (SUT, quant, conditions)
    ├── model_registry.json   # 사용된 모든 모델의 HF ID + revision hash
    ├── code_snapshot.patch   # 실험 시작 시점 git diff (uncommitted 있다면)
    ├── git_sha.txt           # 실험 시점 HEAD sha
    ├── results.jsonl         # 모든 run의 입출력 + latency + VRAM
    ├── metrics.json          # 집계 metric
    ├── stdout.log
    └── stderr.log
```

### 5.2 실험 시작 시 자동 기록 (Claude Code가 짜야 할 wrapper)

```python
# scripts/run_experiment.py
def run_experiment(config_path):
    run_id = f"{datetime.now():%Y%m%d_%H%M}_{config['name']}"
    run_dir = Path(f"experiments/{run_id}")
    run_dir.mkdir(parents=True)
    
    # 재현성 snapshot
    subprocess.run(["git", "rev-parse", "HEAD"], 
                   stdout=open(run_dir/"git_sha.txt", "w"))
    subprocess.run(["git", "diff", "HEAD"], 
                   stdout=open(run_dir/"code_snapshot.patch", "w"))
    shutil.copy(config_path, run_dir/"config.yaml")
    
    # 실제 실행
    results = execute(config)
    
    # 저장 + commit
    save_all(run_dir, results)
    subprocess.run(["git", "add", str(run_dir)])
    subprocess.run(["git", "commit", "-m", 
                    f"exp({config['name']}): complete sweep with {len(results)} runs"])
    subprocess.run(["git", "push"])
```

### 5.3 실험 결과도 commit 대상

작은 artifacts (metrics.json, figures)는 git으로 관리. 큰 것 (개별 response jsonl)은 git-lfs 또는 HF dataset repo 분리.

---

## 6. Directory Layout (최종)

```
heard-bench/
├── README.md                          # tutorial 관문
├── LICENSE                            # MIT
├── requirements.txt
├── pyproject.toml                     # 선택
├── .githooks/
│   └── pre-commit                     # AI attribution blocker
├── .gitignore
├── configs/
│   ├── models.yaml
│   ├── sweeps/
│   │   ├── reduced.yaml
│   │   └── full.yaml
│   └── personas/
│       ├── yejin_florist.yaml
│       ├── minseok_cafe.yaml
│       └── sunhee_hair.yaml
├── src/                               # 실제 코드
│   ├── datagen/
│   ├── node/
│   ├── mirror/
│   ├── eval/
│   └── utils/
├── scripts/                           # CLI entrypoints
│   ├── 01_generate_data.py
│   ├── 02_run_filter.py
│   ├── 03_run_sweep.py
│   ├── 04_judge.py
│   ├── 05_make_figures.py
│   └── 06_upload_hf.py
├── data/                              # 생성된 데이터
│   ├── raw/
│   ├── scenarios/
│   └── final/
├── experiments/                       # 실험 결과 (자동 생성)
├── report/
│   ├── 20243053.tex 또는 .md
│   ├── figures/
│   └── references.bib
└── docs/
    ├── guides/                        # tutorial
    ├── lab_notebook/                  # 일지
    └── decisions/                     # ADR
```

---

## 7. Day-by-Day Git Discipline

각 day의 **최소 commit 수**:

| Day | 최소 commits | 반드시 포함되는 artifact |
|---|---|---|
| Day 1 | 8+ | repo init, pre-commit hook, README skeleton, personas×3, utterance gen script, day1 lab notebook |
| Day 2 | 10+ | 2400 utterances, 150 scenarios, filter code, 100 final items, day2 lab notebook |
| Day 3 | 6+ | sweep configs, 9000+ run results, metrics, day3 lab notebook |
| Day 4 | 10+ | figures×6, report draft, HF upload, day4 lab notebook, ADR×4 |

Claude Code는 하루 종료 시 `git log --oneline --since="today"` 로 commit 수를 확인하고 최소 수 미달이면 사용자에게 경고.

---

## 8. HuggingFace Dataset Repo (별도)

Dataset은 **메인 repo와 분리된 HF dataset repo**에 올림:

```
heard-bench/                           # GitHub (code)
└── data/final/                        # 여기엔 snapshot만, 원본은 HF

heard-bench-dataset/                   # HuggingFace (data)
├── README.md                          # dataset card
├── en_subset/test.jsonl
├── ko_translated/test.jsonl
├── ko_native/test.jsonl
└── dataset_script.py
```

HF 업로드도 git (HF Hub가 git 기반). 여기도 동일한 no-Claude-attribution 규칙 적용.

---

## 9. Error Recovery Protocols

### 9.1 실수로 Claude attribution 들어간 commit 발견

```bash
# 최근 1개
git commit --amend  # editor 열고 수정

# 여러 개
git rebase -i origin/main  # pick → reword
git push --force-with-lease  # force push (단일 작업자라 안전)
```

### 9.2 Push 못한 채 인스턴스 종료

Claude Code는 **작업 종료 직전** `git status && git push` 를 실행해야 함. 이를 매 session 마지막 action으로.

### 9.3 Large file 실수로 commit

```bash
git rm --cached path/to/large
git commit -m "chore: remove accidentally tracked large file"
# git-lfs로 재추가
git lfs track "*.jsonl"
git add .gitattributes path/to/large
git commit -m "chore: migrate large file to lfs"
```

---

## 10. Summary: What Claude Code Must Do

1. **Repo 초기화 직후**:
   - git config name/email 사용자 값으로
   - `.githooks/pre-commit` 설치
   - README.md skeleton + docs/ 디렉터리 구조 생성
   - 첫 commit: `chore: initial repo layout`

2. **작업 중 매 30분–1시간**:
   - 완료된 단위 commit + push
   - Commit message는 §3 규약
   - Claude/AI 문구 절대 금지 (pre-commit hook이 막지만 self-check도)

3. **주요 의사결정 시**:
   - ADR 작성 (`docs/decisions/000N-*.md`)
   - 해당 commit과 함께 push

4. **매일 업무 종료 시**:
   - `docs/lab_notebook/YYYY-MM-DD_dayN_*.md` 작성
   - 오늘 commits 나열 + 내일 할 일
   - `git push` 로 마감

5. **실험 실행 시**:
   - `experiments/{run_id}/` 에 config/snapshot/results 자동 저장
   - 완료 후 즉시 commit

6. **세션 종료 직전 반드시**:
   - `git status` 로 uncommitted 없는지 확인
   - `git push` 로 로컬만 있는 것 없는지 확인

---

**End of git workflow. 이 문서의 규칙을 어기면 나중에 본인이 repo를 이해 못 하게 됩니다. Claude Code, discipline.**
