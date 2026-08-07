#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规范重审(v3 judge)收工后重建清洗资产 v2(纯确定性, 无 API):
  keep 的修正才生效, keep 的单证嫌疑才降权/删除, retract 全部恢复原样。
产出:
  /data/labels_dedup_softclean_v2.jsonl + /data/suspect_weights_v2.json (软化臂)
  /data/labels_dedup_clean_v2.jsonl                      (硬删臂)
  /data/test_mislabel_exclude_ids_rt_v3.txt / _sk_v3.txt (评测剔除 v3, 新文件名)
  /data/val_ids_v2_denoised.txt                          (去噪 dev, 就地更新)
"""
import json

def keepset(path):
    s = set()
    for l in open(path, encoding="utf-8"):
        j = json.loads(l)
        if j["decision"] == "keep":
            s.add(j["video_id"])
    return s


def overlay(base, fix_path):
    """快递新规补审(08-04): →B 方向以 bfix 裁决为准, 覆盖第一轮。"""
    import os
    if not os.path.exists(fix_path):
        return base
    for l in open(fix_path, encoding="utf-8"):
        j = json.loads(l)
        (base.add if j["decision"] == "keep" else base.discard)(j["video_id"])
    return base

keep_rt_tr = overlay(keepset("/data/rejudge_rt_train.jsonl"),
                     "/data/rejudge_rt_train_bfix.jsonl")
keep_rt_te = overlay(keepset("/data/rejudge_rt_test.jsonl"),
                     "/data/rejudge_rt_test_bfix.jsonl")
keep_sk_tr = keepset("/data/rejudge_sk_train.jsonl")
keep_sk_te = keepset("/data/rejudge_sk_test.jsonl")

corr = {}
for l in open("/mnt/disks/data/anker_vlm_training/outputs/diag/gt_train_corrections.jsonl"):
    j = json.loads(l)
    if j["video_id"] in keep_rt_tr:
        corr[j["video_id"]] = j["correct_rt"]
dele = set()
for l in open("/mnt/disks/data/anker_vlm_training/outputs/diag/gt_train_deletes.jsonl"):
    j = json.loads(l)
    if j["video_id"] in keep_rt_tr:
        dele.add(j["video_id"])

nc = nw = n = 0
weights = {}
soft = open("/data/labels_dedup_softclean_v2.jsonl", "w", encoding="utf-8")
hard = open("/data/labels_dedup_clean_v2.jsonl", "w", encoding="utf-8")
for l in open("/data/labels_dedup.jsonl", encoding="utf-8"):
    j = json.loads(l); vid = j["video_id"]; n += 1
    if vid in corr:
        j["labels"]["role_type"] = corr[vid]; nc += 1
        s = json.dumps(j, ensure_ascii=False) + "\n"
        soft.write(s); hard.write(s)
    elif vid in dele:
        weights[vid] = 0.3; nw += 1
        soft.write(json.dumps(j, ensure_ascii=False) + "\n")  # 软化: 保留降权
        # 硬删: 不写入
    else:
        s = json.dumps(j, ensure_ascii=False) + "\n"
        soft.write(s); hard.write(s)
soft.close(); hard.close()
json.dump(weights, open("/data/suspect_weights_v2.json", "w"))

open("/data/test_mislabel_exclude_ids_rt_v3.txt", "w").write(
    "\n".join(sorted(keep_rt_te)) + "\n")
open("/data/test_mislabel_exclude_ids_sk_v3.txt", "w").write(
    "\n".join(sorted(keep_sk_te)) + "\n")

val = {x.strip() for x in open("/data/val_ids_v2.txt") if x.strip()}
bad = (keep_rt_tr | keep_sk_tr) & val
open("/data/val_ids_v2_denoised.txt", "w").write(
    "\n".join(sorted(val - bad)) + "\n")

print(f"[v2] 全量 {n} | 修正 {nc}(v1 1933) | 降权 {nw}(v1 2307) "
      f"| 硬删版保留 {n - nw - (2307 - nw) if False else n - nw} 条")
print(f"[v3口径] test RT 剔 {len(keep_rt_te)}(v1 522) | "
      f"SK 剔 {len(keep_sk_te)}(前版 129)")
print(f"[dev] 去噪 dev {len(val - bad)}/{len(val)}")
print("[done] assets_v2")
