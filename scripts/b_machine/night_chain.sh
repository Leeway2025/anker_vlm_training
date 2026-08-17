#!/bin/bash
# ============================================================================
# 今晚串行单锁总编排 —— 单容器进程内顺序跑完两活,永不并发抢 8 芯。
#   周末崩因回顾: 多条夜链并发 → seed-2 被 SIGTERM 4×; /data 掉链; .git root占。
# 队列(单锁串行,任一时刻只有一个 TPU 作业):
#   Job1  rt-w=8 全链: SFT(唯一变量=RT首2字符loss×8, WEEK_PLAN 08-03晚"RT修复第一发")
#         → eval(base-gate) → dump裸logits → 双模+三模集成(rt-w 模型入池当第三模)
#         (跳过已证必败的权重汤 —— §5 法则#1)
#   Job2  S5-RT 全链: rtfocus 体检 → 去字母化(修 S5 泄题死因)→ CoT-SFT(喂去字母资产)
#         → eval(base-gate) → 手术先验交付候选
# 特性: flock 单锁 / unset WDS_DIR / 崩溃残留挪走 / 分段幂等 / 时间戳日志 /
#       Job1 失败不挡 Job2(两活独立,各自 set -e 子壳)
# 启动(宿主机,后台脱离):
#   sudo docker exec -d tpu_train bash -lc \
#     'cd /workspace && nohup bash scripts/night_chain.sh \
#        > logs/night_chain.$(date +%m%d_%H%M).log 2>&1 &'
# 明早巡检: tail logs/night_chain.*.log,并看文末"明早看"清单。
# ============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs outputs

# —— 单锁: 防误触双起夜链(周末崩因#1) ——
exec 200>/tmp/night_chain.lock
flock -n 200 || { echo "[FATAL] 另一夜链已持锁 /tmp/night_chain.lock,退出"; exit 1; }

# —— 环境净化(周末崩因#2: WDS_DIR 继承致测试集导向错误 tar,KeyError) ——
unset WDS_DIR

TS(){ date '+%F %T'; }
# 从 eval 报告里抠 SubKS 百分数(如 "SubKS acc = 73.83%")
subks(){ grep -oE 'SubKS[[:space:]]+acc[[:space:]]*=[[:space:]]*[0-9.]+' "$1" 2>/dev/null \
         | grep -oE '[0-9.]+$' | head -1; }
# RT 百分数(rt-w 疗效判读: 对 seed-1 RT 应 ≥+0.3 且 SubKS 回撤 ≤0.2)
rtacc(){ grep -oE '(RoleType|RT)[[:space:]]+acc[[:space:]]*=[[:space:]]*[0-9.]+' "$1" 2>/dev/null \
         | grep -oE '[0-9.]+$' | head -1; }

echo "[chain] 开跑 $(TS) | 代码: $(git log --oneline -1 2>/dev/null)"
if python3 -c "import jax,sys;sys.exit(0 if len(jax.devices())==8 else 1)" 2>/dev/null; then
  echo "[chain] JAX 见 8 芯 ✓"
else
  echo "[chain][WARN] JAX 未见满 8 芯 —— 巡检时重点看"
fi

# ============================ Job1: rt-w=8 全链 ============================
job1(){
  echo "[J1] ==== rt-w=8 全链 开始 $(TS) ===="
  # 崩溃残留挪走(step 未完即被杀,best.npz 是废的),保留供事后查,幂等重跑
  if [ -e outputs/jax_5b_rtw ] && [ ! -s outputs/jax_5b_rtw/eval_report.txt ]; then
    mv outputs/jax_5b_rtw "outputs/jax_5b_rtw.crashed.$(date +%s)"
    echo "[J1] 崩溃残留已挪走"
  fi
  # ① 训练(配方逐字=seed-1,唯一变量 --rt-w 8 = RT字母位 loss×8);幂等: 有 best.npz 则跳
  if [ ! -s outputs/jax_5b_rtw/train_params_best.npz ]; then
    python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
      --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
      --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
      --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
      --seed 1 --mu-dtype float32 --rt-w 8 --prefetch-workers 24 --out outputs/jax_5b_rtw
    echo "[J1] rt-w=8 训完 $(TS)"
  else echo "[J1] rt-w=8 已训,跳过训练"; fi
  # ② eval + 落报告(幂等)
  if [ ! -s outputs/jax_5b_rtw/eval_report.txt ]; then
    bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      outputs/jax_5b_rtw/eval_preds outputs/jax_5b_rtw/train_params_best.npz 8
    python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_rtw/eval_preds.jsonl \
      --labels /data/labels_test.jsonl --per-class \
      --exclude-ids /data/test_mislabel_exclude_ids_522.txt | tee outputs/jax_5b_rtw/eval_report.txt
  fi
  RTW=$(subks outputs/jax_5b_rtw/eval_report.txt)
  RTW_RT=$(rtacc outputs/jax_5b_rtw/eval_report.txt)
  echo "[J1] rt-w=8 裸 SubKS=${RTW:-?} RT=${RTW_RT:-?}(对 seed-1 SubKS73.52 —— 判读: RT≥+0.3 且 SubKS 回撤≤0.2 为疗效)"
  # ③ dump rt-w 裸 logits(供三模集成;幂等)
  if [ ! -s outputs/optin_rtw/preds.jsonl ]; then
    mkdir -p outputs/optin_rtw
    INFER_ARGS='--dump-letter-logits' \
    bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      outputs/optin_rtw/preds outputs/jax_5b_rtw/train_params_best.npz 8
  fi
  # ④ 集成: 双模(已证 74.44)必跑;三模按 rt-w base-gate(裸≥73.0)决定,rt-w 入池当第三模
  bash scripts/ensemble2.sh || echo "[J1][WARN] 双模集成失败(见上)"
  if awk "BEGIN{exit !(${RTW:-0}>=73.0)}"; then
    M3_DIR=outputs/optin_rtw M3_PARAMS=outputs/jax_5b_rtw/train_params_best.npz M3_NAME=rt-w \
      bash scripts/ensemble3.sh || echo "[J1][WARN] 三模集成失败(见上)"
  else
    echo "[J1] rt-w 裸分未过 base-gate(<73.0),跳过三模集成(仅留双模 74.44)"
  fi
  echo "[J1] ==== rt-w=8 全链 完成 $(TS) ===="
}

# ============================ Job2: S5-RT 全链 ============================
job2(){
  echo "[J2] ==== S5-RT 全链 开始 $(TS) ===="
  RAW=/data/assets_rat/asset_C_rtfocus.jsonl              # RT 强化 CoT(含字母)
  STRIP=/data/assets_rat/asset_C_rtfocus_nolttr.jsonl    # 去字母化产出(喂训练)
  test -s "$RAW" || { echo "[J2][FATAL] 缺 rtfocus 资产 $RAW"; return 1; }
  # ① 体检(在 raw 上跑: 含字母才能与 GT 对账);check_cot_asset 硬伤会 exit1 挡链
  ASSET="$RAW" bash scripts/check_cot_asset.sh
  ASSET="$RAW" bash scripts/check_cot_labels.sh  || echo "[J2] 链↔GT 对账仅供参考"
  ASSET="$RAW" bash scripts/check_cot_rt_bias.sh || true
  # ② 去字母化(修 S5 泄题死因 §5#3;幂等)
  [ -s "$STRIP" ] || bash scripts/strip_cot_letters.sh "$RAW" "$STRIP"
  # 去字母验收: 残留"带字母"答案指纹必须为 0,否则拒绝开训(不重犯 S5 泄题)。
  # 口径与 strip_cot_letters.sh 的 LEAK 一致 —— 无字母的短语提及不算泄漏。
  RES=$(python3 - "$STRIP" <<'PY'
import json, re, sys
p = sys.argv[1]
RT = r'(?i:Role\s*Type)'; SK = r'(?i:Sub[\s\-_]?keyscene)'
LEAK = re.compile(
    rf'{RT}[\s:,\-=]*(?:is|as|of)?\s*\(?\s*[A-E]\b'
    rf'|{SK}[\s:,\-=]*\(?\s*[a-u]\)|{SK}\s+[a-u]\b|\(\s*[A-Ea-u]\s*\)')
print(sum(1 for l in open(p) if LEAK.search(json.loads(l)['reasoning_chain'])))
PY
)
  [ "$RES" = "0" ] || { echo "[J2][FATAL] 去字母残留 $RES 条,拒绝开训"; return 1; }
  echo "[J2] 去字母化验收通过(残留 0)-> $STRIP"
  # ③ S5-RT 训练(配方逐字=seed-1;唯一变量=去字母 CoT 三参数)~5-6h
  if [ ! -s outputs/jax_5b_s5rt/train_params_best.npz ]; then
    python jax_impl/train_sft.py --labels /data/labels_dedup.jsonl \
      --layout /data/hf_layout.json --rank-scheme prod --train-vision --train-projector \
      --init-npz outputs/jax_5a/proj_a.npz --augment --early-stop-patience 3 \
      --accum 32 --steps 1500 --eval-every 100 --val-ids /data/val_ids_v2.txt \
      --seed 1 --mu-dtype float32 \
      --cot-file "$STRIP" --cot-ratio 0.6 --cot-anneal 0.5 \
      --prefetch-workers 24 --out outputs/jax_5b_s5rt
    echo "[J2] S5-RT 训完 $(TS)"
  else echo "[J2] S5-RT 已训,跳过训练"; fi
  # ④ eval + 落报告(幂等)
  if [ ! -s outputs/jax_5b_s5rt/eval_report.txt ]; then
    bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      outputs/jax_5b_s5rt/eval_preds outputs/jax_5b_s5rt/train_params_best.npz 8
    python3 jax_impl/eval_metrics.py --preds outputs/jax_5b_s5rt/eval_preds.jsonl \
      --labels /data/labels_test.jsonl --per-class \
      --exclude-ids /data/test_mislabel_exclude_ids_522.txt | tee outputs/jax_5b_s5rt/eval_report.txt
  fi
  SR=$(subks outputs/jax_5b_s5rt/eval_report.txt)
  echo "[J2] S5-RT 裸 SubKS=${SR:-?}(旧 S5 泄题版曾 -2.69,去字母后应回升)"
  # ⑤ 裸 logits → 手术先验 → 交付候选(独立目录,不覆盖 optin/)
  mkdir -p outputs/optin_s5rt
  if [ ! -s outputs/optin_s5rt/preds.jsonl ]; then
    INFER_ARGS='--dump-letter-logits' \
    bash jax_impl/infer_sharded.sh python /data/labels_test.jsonl /data/hf_layout.json \
      outputs/optin_s5rt/preds outputs/jax_5b_s5rt/train_params_best.npz 8
  fi
  python3 jax_impl/apply_surgical_prior.py --logits outputs/optin_s5rt/preds.jsonl \
    --labels /data/labels_test.jsonl \
    --fold-a /data/test_sfoldA.jsonl --fold-b /data/test_sfoldB.jsonl \
    --tau 0.7 --out outputs/optin_s5rt/preds_surg.jsonl
  python3 jax_impl/eval_metrics.py --preds outputs/optin_s5rt/preds_surg.jsonl \
    --labels /data/labels_test.jsonl --per-class \
    --exclude-ids /data/test_mislabel_exclude_ids_522.txt | tee outputs/optin_s5rt/eval_report.txt
  echo "[J2] ==== S5-RT 全链 完成 $(TS) ===="
}

# —— 串行执行: 各作业 set -e 子壳隔离,内部失败即止,作业间互不牵连 ——
( set -e; job1 ) || echo "[chain][ERR] Job1 异常终止(见上),继续 Job2"
( set -e; job2 ) || echo "[chain][ERR] Job2 异常终止(见上)"

echo "[chain] 全部完成 $(TS)"
echo "[chain] ===== 明早看 ====="
echo "  Job1 rt-w 裸   : outputs/jax_5b_rtw/eval_report.txt   (对 seed-1 SubKS73.52; RT≥+0.3&回撤≤0.2=疗效)"
echo "  Job1 双模集成  : outputs/optin/eval_report_ens.txt      (对 74.44)"
echo "  Job1 三模集成  : outputs/optin/eval_report_ens3.txt     (对 74.44; 第三模=rt-w)"
echo "  Job2 S5-RT 裸  : outputs/jax_5b_s5rt/eval_report.txt    (对 seed-1 73.52)"
echo "  Job2 S5-RT交付 : outputs/optin_s5rt/eval_report.txt     (对 73.77 / EunoVLM 73.83)"
