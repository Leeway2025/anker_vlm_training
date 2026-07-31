#!/bin/bash
# 白天批量生产线(一键): 自动判断 S5 状态 → 串起当日全部任务,点完即走。
#   ① 两个秒级体检(存档) ② TPU 主线状态机(S5在跑=挂接力/S5已完=收读数+KTO配对/
#   S5没跑=补跑全链) ③ RT强化重标(需 GEMINI_API_KEY 与 WDS_DIR)
# 用法: cd /workspace && git pull
#       export GEMINI_API_KEY=AQ.xxx  WDS_DIR=<训练集tar目录>   # ③需要
#       bash scripts/day_run.sh
set -e
cd "$(dirname "$0")/.."
echo "========== 白天生产线 $(date) =========="

# ① 体检两连(只读,失败不挡后续)
bash scripts/check_cot_labels.sh  | tee cot_labels_check.txt  || true
bash scripts/check_cot_rt_bias.sh | tee cot_rt_bias_check.txt || true

# ② TPU 主线状态机([n] 写法防 pgrep 匹配到本脚本/接力进程自身)
if pgrep -f '[n]ight_s5.sh' >/dev/null; then
  echo "[day] S5 夜链正在跑 → 挂接力: 结束后自动 KTO 配对挖掘"
  nohup bash -c 'while pgrep -f "[n]ight_s5.sh" >/dev/null; do sleep 300; done; \
                 bash scripts/build_kto_rt_pairs.sh' > kto_pairs.log 2>&1 &
elif [ -s outputs/jax_5b_s5/eval_report.txt ]; then
  echo "== S5 已完成,读数如下(晚上原样贴回) =="
  for f in outputs/jax_5b_s5/eval_report.txt outputs/optin_s5/eval_report.txt \
           outputs/optin/eval_report_joint.txt; do
    echo "---- $f"; head -8 "$f" 2>/dev/null || echo "(缺失)"
  done
  BASE=$(python3 - <<'PY'
import re
def score(p):
    try:
        t = open(p).read()
        return sum(float(re.search(rf'{k}\s+acc\s*=\s*([\d.]+)%', t).group(1))
                   for k in ('RoleType', 'SubKS'))
    except Exception:
        return 0.0
s5 = score('outputs/jax_5b_s5/eval_report.txt')
s1 = score('outputs/jax_5b_seed1/eval_report.txt')
print('outputs/jax_5b_s5/train_params_best.npz' if s5 > s1
      else 'outputs/jax_5b_seed1/train_params_best.npz')
PY
)
  echo "[day] KTO 配对底座(裸分 RT+SubKS 之和自动选优)= $BASE"
  if [ -s /data/kto_pairs_rt.jsonl ]; then
    echo "[day] /data/kto_pairs_rt.jsonl 已存在,跳过重复挖掘(要重挖先删该文件)"
  else
    BASE="$BASE" nohup bash scripts/build_kto_rt_pairs.sh > kto_pairs.log 2>&1 &
  fi
else
  echo "[day] S5 从未跑过 → 补跑全链: 资产门禁 + S5夜链 + KTO配对接力"
  bash scripts/check_cot_asset.sh
  nohup bash -c 'bash scripts/night_s5.sh && bash scripts/build_kto_rt_pairs.sh' \
      > day_s5_kto.log 2>&1 &
fi

# ③ RT 强化重标(网络任务,与 TPU 并行;缺环境变量则提示跳过)
if [ -n "$GEMINI_API_KEY" ] && [ -n "$WDS_DIR" ]; then
  if pgrep -f '[r]ationalize_cot' >/dev/null; then
    echo "[day] rationalize 已在跑,不重复启动"
  else
    nohup bash scripts/rationalize_rt.sh > rat_rt.log 2>&1 &
    echo "[day] RT 强化重标已启动(rat_rt.log)"
  fi
else
  echo "[day][跳过] RT重标: 先 export GEMINI_API_KEY=AQ.xxx WDS_DIR=<tar目录>"
  echo "           再单独执行: nohup bash scripts/rationalize_rt.sh > rat_rt.log 2>&1 &"
fi

sleep 10
echo "========== 存活确认 =========="
pgrep -af '[n]ight_s5|[b]uild_kto|[r]ationalize|[t]rain_sft|[i]nfer' || echo "(无后台任务?检查上方报错)"
echo "
========== 晚上贴回清单(五样) ==========
1. 本脚本开头打印的 S5 三份报告读数(或 outputs/*/eval_report.txt 的 head -8)
2. tail -20 kto_pairs.log     ← 重点: RT 错行方向分布(A->D 占比定 KTO 生死)
3. tail -5 rat_rt.log         ← 完成量 + unsupported 率(=懒标D浓度实测)
4. cot_labels_check.txt 头 6 行(一致率)
5. cot_rt_bias_check.txt 头 2 行(怯判教材比例)"
