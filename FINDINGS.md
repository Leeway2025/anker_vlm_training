
## v11(2026-07-23): GRPO 改良版真机门禁全绿

**结论**: `jax_impl/grpo.py`(Dr.GRPO 无偏优势 + DAPO 动态采样/Clip-Higher,
无 think 定格式)在 SPOT v6e-4 过三道门禁:
① fakedata 欠训 SFT 起点,6 轮 mean_reward 0.241→0.297(+23%),
   kept 94%→69%(策略确定化,动态过滤按设计工作);
② 产物 npz detect=prod(256/512),infer 严格加载器可直接消费;
③ 全流程端到端无崩溃(采样→过滤→PPO 裁剪更新→落盘)。

**八连修档案(shard_map 参数进场姿势,后来者防坑)**:
1. 闭包 numpy 索引数组 → Manual mesh 里常量 gather 炸(改 traced 入参);
2. 闭包 base 参数 → 同上('devices' mesh 分片进 Manual 上下文;
   **train_sft 把 base 作为操作数传不是风格,是必须**);
3. ckpt 加载的 base 布局与 jit 编译预期不一致 → 进循环前统一
   device_put(NamedSharding(mesh, P()));
4. 全尺寸 logits [B,L,V] 先转 fp32 再切片 → HBM OOM(先切后转);
5. ref 前向放在梯度图里 → OOM(冻结策略 logits 恒定,rollout 期预计算当数据喂);
6. f32[B,2] 列抽取致编译器选转置布局 → RuntimeProgramInputMismatch(拆一维);
7. 关 donate → 更新期 policy/opt 双份缓冲,程序加载 OOM(donate 必开);
8. SPOT 抢占即删盘 → 重建全重来(验证机一律 --instance-termination-action=STOP)。

**真数据上场配方**(最强 SFT 底座出炉后):
  先 --rounds 2 --chunk 256 试跑看 kept/KL → 正式 --rounds 12 --chunk 2048
  --group 6 --lr 2e-6(真数据比门禁保守一个量级)~3h;
  可选 --class-weight-json 把 m/E 先验矫正内化进奖励。
