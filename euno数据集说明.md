# anker_video_clips_wds_testset 数据集说明

## 📁 数据集概览

| 属性 | 值 |
|------|-----|
| **数据格式** | WebDataset (tar shards + pickle) |
| **测试集数据路径** | `anker_video_clips_wds_testset/` (11,022 样本, 23 分片) |
| **训练集数据路径** | `anker_video_clips_wds_full/` (1,082,100 样本, 2,168 分片) |
| **测试集标注文件** | `anker_video_clips/euno_test_v3.0.5_des25_updated_260323_merged_batch104_2800_test_format_v23_frames.json` |
| **训练集标注文件** | `anker_video_clips/euno_train_v3.0.18_balanced_100k_frames.json` |
| **每个样本帧数** | 16 帧 (所有样本统一) |
| **图像分辨率** | 384 × 384 pixels |
| **图像格式** | JPEG (RGB, 3通道) |

---

## 📂 目录结构

```
anker_video_clips_wds_testset/
├── index.json              # 样本索引文件 (key → shard_id 映射)
├── shard_stats.json        # 分片统计信息
├── shard-000000.tar        # 分片 0 (500 个样本)
├── shard-000001.tar        # 分片 1 (500 个样本)
├── shard-000002.tar        # 分片 2 (500 个样本)
│   ...
├── shard-000021.tar        # 分片 21 (500 个样本)
└── shard-000022.tar        # 分片 22 (22 个样本)
```

每个 `.tar` 文件包含 **500 个 `.pyd` 文件** (Python pickle),每个 `.pyd` 文件代表一个视频片段样本。

---

## 🏷️ 标签文件 (Annotations)

### 标注文件位置

标注文件为独立 JSON 文件,不在 tar 分片内:

**测试集标注**:
```
anker_video_clips/euno_test_v3.0.5_des25_updated_260323_merged_batch104_2800_test_format_v23_frames.json
```
- **样本数量**: 11,022 

**训练集标注**:
```
anker_video_clips/euno_train_v3.0.18_balanced_100k_frames.json
```
- **样本数量**: 98,395 (平衡采样版本)

- **格式**: LlamaFactory 对话格式

### 标注文件结构

```json
[
  {
    "id": "0",
    "videos": ["testset_1k_250730/0006f134-xxx_segment_1"],
    "conversations": [
      {
        "from": "human",
        "value": "<video>...prompt template..."
      },
      {
        "from": "gpt",
        "value": "D|g|A man in a hat approached a camera..."
      }
    ]
  },
  ...
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 样本唯一标识符,如 `"0"`, `"100"` |
| `videos` | `List[str]` | 视频相对路径列表 (通常只有 1 个),格式同 index.json 的 key |
| `conversations` | `List[dict]` | 对话列表,包含 human (提示) 和 gpt (标签) 两轮 |
| `conversations[0].from` | `str` | 固定为 `"human"` |
| `conversations[0].value` | `str` | 提示模板 (所有样本相同) |
| `conversations[1].from` | `str` | 固定为 `"gpt"` |
| `conversations[1].value` | `str` | **标签值**,格式为 `RoleType \| Sub-Keyscene \| short_description` |

---

## 🏷️ 标签值详解

### 标签格式

每个样本的标签 (gpt 回复) 为结构化文本,格式:

```
<RoleType代码> | <Sub-Keyscene代码> | <简短描述>
```

示例:
```
D|g|A man in a hat approached a camera at a residence, paused, then turned and left.
```

### RoleType (角色类型)

| 代码 | 含义 | 说明 |
|------|------|------|
| **A** | Family Member | 家庭成员 |
| **B** | Staff | 工作人员 (如快递员、维修工) |
| **C** | Suspicious Person | 可疑人员 |
| **D** | Unspecified | 未指定身份 |
| **E** | Non-Human | 非人类 (动物、车辆等) |

### Sub-Keyscene (子场景类型)

| 代码 | 含义 | 说明 |
|------|------|------|
| **a** | Vehicle Access | 车辆进出 |
| **b** | Dog Walking | 遛狗 |
| **c** | Kid Playing | 儿童玩耍 |
| **d** | Kid Studying | 儿童学习 |
| **e** | Leisure Activity | 休闲活动 |
| **f** | Home Chores | 家务活动 |
| **g** | Visitor Arrival | 访客到达 |
| **h** | Package Brought Home | 包裹被带回家 |
| **i** | Package Delivery | 包裹送达 |
| **j** | Person Falling | 人员摔倒 |
| **k** | Leaving Porch | 离开门廊 |
| **l** | Approaching Porch | 靠近门廊 |
| **m** | Other Normal Activity | 其他正常活动 |
| **n** | Package Taken Away | 包裹被取走 |
| **o** | Other Property Damage | 其他财产损坏 |
| **p** | Wildlife | 野生动物 |
| **q** | Weapon Threat | 武器威胁 |
| **r** | Other Hazards | 其他危险 |
| **s** | Loitering | 徘徊 |
| **t** | Vehicle Anomaly | 车辆异常 |
| **u** | Unauthorized Entry | 非法入侵 |

### 提示模板 (Prompt Template)

所有样本的 human 提示相同:

```
<video>You are an AI surveillance analyst.
Output: role type code | sub-keyscene code | short description.

RoleType: A=Family Member, B=Staff, C=Suspicious Person, D=Unspecified, E=Non-Human
Sub-Keyscene: a=Vehicle Access, b=Dog Walking, c=Kid Playing, d=Kid Studying, e=Leisure Activity, f=Home Chores, g=Visitor Arrival, h=Package Brought Home, i=Package Delivery, j=Person Falling, k=Leaving Porch, l=Approaching Porch, m=Other Normal Activity, n=Package Taken Away, o=Other Property Damage, p=Wildlife, q=Weapon Threat, r=Other Hazards, s=Loitering, t=Vehicle Anomaly, u=Unauthorized Entry

Use "|" to separate fields, no extra words.
```

---

## 📊 标签分布

### RoleType 分布对比

| 代码 | 含义 | 训练集 (98,395) | 测试集 (11,022) |
|------|------|----------------|----------------|
| A | Family Member | 29,122 (29.6%) | 3,060 (27.8%) |
| B | Staff | 14,600 (14.8%) | 1,053 (9.6%) |
| C | Suspicious Person | 18,397 (18.7%) | 1,384 (12.6%) |
| D | Unspecified | 29,243 (29.7%) | 3,846 (34.9%) |
| E | Non-Human | 7,033 (7.1%) | 1,679 (15.2%) |

**分布差异说明**:
- 训练集经过**平衡采样**,各类别分布较为均匀 (29-30% 集中在 A 和 D)
- 测试集中 **E (非人类)** 和 **D (未指定身份)** 占比更高,便于评估模型对稀有类别的识别能力
- 训练集 **B (工作人员)** 和 **C (可疑人员)** 占比高于测试集

### Sub-Keyscene 分布对比

| 代码 | 含义 | 训练集 | 测试集 |
|------|------|--------|--------|
| m | Other Normal Activity | 12,483 (12.7%) | 3,036 (27.5%) |
| a | Vehicle Access | 8,615 (8.8%) | 1,084 (9.8%) |
| k | Leaving Porch | 7,505 (7.6%) | 475 (4.3%) |
| l | Approaching Porch | 7,259 (7.4%) | 758 (6.9%) |
| g | Visitor Arrival | 6,591 (6.7%) | 493 (4.5%) |
| e | Leisure Activity | 6,240 (6.3%) | 954 (8.7%) |
| c | Kid Playing | 6,240 (6.3%) | 203 (1.8%) |
| f | Home Chores | 6,240 (6.3%) | 941 (8.5%) |
| s | Loitering | 5,425 (5.5%) | 487 (4.4%) |
| i | Package Delivery | 4,157 (4.2%) | 749 (6.8%) |
| n | Package Taken Away | 3,572 (3.6%) | 474 (4.3%) |
| t | Vehicle Anomaly | 3,409 (3.5%) | 333 (3.0%) |
| p | Wildlife | 3,142 (3.2%) | 103 (0.9%) |
| q | Weapon Threat | 3,156 (3.2%) | 63 (0.6%) |
| u | Unauthorized Entry | 3,122 (3.2%) | 117 (1.1%) |
| h | Package Brought Home | 3,126 (3.2%) | 148 (1.3%) |
| o | Other Property Damage | 3,137 (3.2%) | 366 (3.3%) |
| b | Dog Walking | 3,239 (3.3%) | 233 (2.1%) |
| r | Other Hazards | 1,580 (1.6%) | 2 (0.0%) |
| j | Person Falling | 132 (0.1%) | 3 (0.0%) |

**分布差异说明**:
- 训练集经过**平衡采样**,大多数场景类别占比在 3-9% 之间,分布较为均匀
- 测试集中 **m (其他正常活动)** 占比显著更高 (27.5% vs 12.7%),反映真实场景分布
- 测试集中 **i (包裹送达)**、**e (休闲活动)**、**f (家务活动)** 占比更高
- 稀有类别 (j, r) 在两个集合中样本数都很少

---

## 🔍 真实数据示例

### 数据格式 + 标签 完整示例

#### 示例 1: 未指定身份 + 访客到达 (D|g)

```python
# 数据 (from .pyd)
{
    "video_rel": "testset_1k_250730/0006f134-a7da-405e-a47a-7d0e8405bbe3_segment_1",
    "num_frames": 16,
    "frames": [b'\xff\xd8\xff...', ...],  # 16 帧 JPEG, 384x384 RGB
}

# 标签 (from annotation JSON)
{
    "id": "0",
    "videos": ["testset_1k_250730/0006f134-a7da-405e-a47a-7d0e8405bbe3_segment_1"],
    "conversations": [
        {
            "from": "human",
            "value": "<video>You are an AI surveillance analyst..."
        },
        {
            "from": "gpt",
            "value": "D|g|A man in a hat approached a camera at a residence, paused, then turned and left."
        }
    ]
}
# 解析: RoleType=D (未指定身份), Sub-Keyscene=g (访客到达)
# 描述: 一名戴帽子的男子在住宅处靠近摄像头,停顿后转身离开
```

#### 示例 2: 家庭成员 + 车辆进出 (A|a)

```python
# 标签
{
    "id": "500",
    "videos": ["testset_1k_250730/decrypt_T8020P1124100C90_..._segment_1"],
    "conversations": [
        {"from": "human", "value": "<video>You are an AI surveillance analyst..."},
        {"from": "gpt", "value": "A|a|At night, a woman in a dark coat enters a pickup. Another woman in a light suit walks down stairs, then goes back up."}
    ]
}
# 解析: RoleType=A (家庭成员), Sub-Keyscene=a (车辆进出)
# 描述: 夜间,一名穿深色外套的女性进入皮卡,另一名穿浅色套装的女性走下楼梯后又返回
```

#### 示例 3: 可疑人员 + 包裹被取走 (C|n)

```python
# 标签
{
    "id": "8000",
    "videos": ["trainset_4k_250828/5911b3bd-ae4d-4d1a-9aaf-91a1462a5ae6_segment_1"],
    "conversations": [
        {"from": "human", "value": "<video>You are an AI surveillance analyst..."},
        {"from": "gpt", "value": "C|n|During the day, a man in light-colored clothes took a package from a residential doorway and left."}
    ]
}
# 解析: RoleType=C (可疑人员), Sub-Keyscene=n (包裹被取走)
# 描述: 白天,一名穿浅色衣服的男子从住宅门口取走包裹后离开
```

#### 示例 4: 非人类 + 野生动物 (E|p)

```python
# 标签
{
    "id": "...",
    "videos": ["testset_1k_250730/..."],
    "conversations": [
        {"from": "human", "value": "<video>You are an AI surveillance analyst..."},
        {"from": "gpt", "value": "E|p|At night, outside the residence, a bear knocked over a blue bucket placed on the porch."}
    ]
}
# 解析: RoleType=E (非人类), Sub-Keyscene=p (野生动物)
# 描述: 夜间,住宅外一只熊撞翻了门廊上的蓝色水桶
```

#### 示例 5: 工作人员 + 包裹送达 (B|i)

```python
# 标签
{
    "id": "...",
    "videos": ["trainset_9k_260304_batch104/..."],
    "conversations": [
        {"from": "human", "value": "<video>You are an AI surveillance analyst..."},
        {"from": "gpt", "value": "B|i|Daytime at a residence, a male worker puts down a package, pauses, then picks it up and continues delivering."}
    ]
}
# 解析: RoleType=B (工作人员), Sub-Keyscene=i (包裹送达)
# 描述: 白天在住宅处,一名男性工作人员放下包裹,停顿后又拿起继续配送
```

---

## 🎞️ 数据格式 (tar 分片中的 .pyd 文件)

每个 `.pyd` 文件是一个 **Python pickle** 序列化的字典,包含以下字段:

### 字段定义

```python
{
    "frames": List[bytes],     # 视频帧列表,每帧为 JPEG 编码的字节数据
    "video_rel": str,          # 视频相对路径/名称 (与标注文件中的 videos[0] 对应)
    "num_frames": int          # 帧数量 (固定为 16)
}
```

### 字段详细说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `frames` | `List[bytes]` | 长度为 16 的列表,每个元素是一帧 JPEG 图片的原始字节 (以 `\xff\xd8\xff` 开头) |
| `video_rel` | `str` | 视频片段的相对路径,格式同 index.json 的 key |
| `num_frames` | `int` | 帧数量,所有样本均为 **16** |

### 单帧图像属性

- **分辨率**: 384 × 384 pixels
- **色彩模式**: RGB (3 通道)
- **编码格式**: JPEG
- **单帧大小**: ~40-100 KB (平均约 65 KB)
- **单样本总大小**: ~0.9-1.5 MB (16 帧合计)

---

## 🏷️ 样本索引 (index.json)

`index.json` 是一个 JSON 字典,将**样本键 (sample key)** 映射到**分片索引 (shard index)**,用于定位数据在哪个 tar 分片中。

### 结构定义

```json
{
  "<sample_key>": <shard_index>,
  ...
}
```

- **Key (`sample_key`)**: 字符串,格式为 `<子目录名>/<视频片段ID>`,与标注文件的 `videos[0]` 一致
- **Value (`shard_index`)**: 整数,表示该样本所在的 tar 分片编号 (0-22)

### 示例

```json
{
  "testset_1k_250730/0006f134-a7da-405e-a47a-7d0e8405bbe3_segment_1": 0,
  "testset_1k_250730/00d3de0b-648b-458b-9030-587331490225_segment_1": 0,
  "testset_1k_250922/decrypt_T8030T2324020F2F_T822451024451154_segment_1": 5,
  "testset_2k_250711/T8030P23224512B3_T8210P7422450C39_20240416_segment_1": 10,
  "trainset_4k_250825/T8030P2322510409_T8160P2122501B2C_20230719_segment_1": 15,
  "trainset_9k_260304_batch104/decrypt_T8N00520251803F1_T8E0051_segment_1": 22
}
```

### 命名约定

| 元素 | 说明 |
|------|------|
| `sample_key` 中的分隔符 | 使用 `/` (正斜杠),如 `testset_1k_250730/xxx` |
| tar 文件名中的分隔符 | 使用 `__` (双下划线),如 `testset_1k_250730__xxx.pyd` |
| 子目录命名 | `<split>_<size>_<date>`,如 `testset_1k_250730`, `trainset_9k_260304_batch104` |

---

## 📤 推理结果格式 (Inference Results)

推理结果文件用于模型评估,包含预测结果和置信度分数。

### 文件结构

```json
[
  {
    "id": "3",
    "video": "testset_1k_250730/00e22ff1-c5bc-40be-9c53-3c1aa38ff652_segment_1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "You are an AI surveillance analyst.\nOutput: role type code | sub-keyscene code | short description.\n\nRoleType: A=Family Member, B=Staff, C=Suspicious Person, D=Unspecified, E=Non-Human\nSub-Keyscene: a=Vehicle Access, b=Dog Walking, c=Kid Playing, d=Kid Studying, e=Leisure Activity, f=Home Chores, g=Visitor Arrival, h=Package Brought Home, i=Package Delivery, j=Person Falling, k=Leaving Porch, l=Approaching Porch, m=Other Normal Activity, n=Package Taken Away, o=Other Property Damage, p=Wildlife, q=Weapon Threat, r=Other Hazards, s=Loitering, t=Vehicle Anomaly, u=Unauthorized Entry\n\nUse \"|\" to separate fields, no extra words.\n"
      },
      {
        "from": "gpt",
        "value": "B|i|Daytime at a residence, a male worker puts down a package, pauses, then picks it up and leaves."
      }
    ],
    "pred": {
      "result": "B|i|A person in a blue vest approaches a walkway camera, checks a device, picks up a small package, and leaves along the walkway.",
      "score": [
        0.9472081661224365,
        1.0
      ]
    }
  },
  ...
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 样本唯一标识符,与标注文件中的 `id` 对应 |
| `video` | `str` | 视频文件路径 (带 `.mp4` 后缀),用于定位原始视频 |
| `conversations` | `List[dict]` | 对话列表,包含 human (提示) 和 gpt (ground truth 标签) |
| `conversations[0].from` | `str` | 固定为 `"human"` |
| `conversations[0].value` | `str` | 提示模板 |
| `conversations[1].from` | `str` | 固定为 `"gpt"` |
| `conversations[1].value` | `str` | **Ground truth 标签**,格式为 `RoleType \| Sub-Keyscene \| description` |
| `pred` | `dict` | 模型预测结果 |
| `pred.result` | `str` | **预测标签**,格式同 ground truth: `RoleType \| Sub-Keyscene \| description` |
| `pred.score` | `List[float]` | 置信度分数列表,通常包含 2 个值 |



## 💻 读取代码示例

### 加载标注文件

```python
import json

# 加载标注
ann_path = "/data_tmp/data/vlm_datasets/data/anker_video_clips/euno_test_v3.0.5_des25_updated_260323_merged_batch104_2800_test_format_v23_frames.json"
with open(ann_path) as f:
    annotations = json.load(f)

# 遍历样本
for ann in annotations:
    video_path = ann["videos"][0]
    label = ann["conversations"][1]["value"]  # gpt 回复
    role_type, sub_keyscene, description = label.split("|", 2)

    print(f"Video: {video_path}")
    print(f"RoleType: {role_type}, Sub-Keyscene: {sub_keyscene}")
    print(f"Description: {description}")
```

### 联合加载数据 + 标签

```python
import json
import tarfile
import pickle
import io
from PIL import Image

# 1. 加载标注
ann_path = "/data_tmp/data/vlm_datasets/data/anker_video_clips/euno_test_v3.0.5_des25_updated_260323_merged_batch104_2800_test_format_v23_frames.json"
with open(ann_path) as f:
    annotations = json.load(f)

# 2. 加载索引
idx_path = "/data_tmp/data/vlm_datasets/data/anker_video_clips_wds_testset/index.json"
with open(idx_path) as f:
    index = json.load(f)

# 3. 联合查询: 获取某个标注对应的视频帧和标签
ann = annotations[0]
video_key = ann["videos"][0]
label = ann["conversations"][1]["value"]

# 定位数据
shard_id = index[video_key]
tar_path = f"/data_tmp/data/vlm_datasets/data/anker_video_clips_wds_testset/shard-{shard_id:06d}.tar"
pyd_filename = video_key.replace("/", "__") + ".pyd"

with tarfile.open(tar_path) as tf:
    member = tf.getmember(pyd_filename)
    fp = tf.extractfile(member)
    data = pickle.load(fp)

    # 解析帧
    frames = [Image.open(io.BytesIO(f)) for f in data["frames"]]

    print(f"Video: {video_key}")
    print(f"Frames: {len(frames)}, Size: {frames[0].size}")
    print(f"Label: {label}")
```

### 使用 WebDataset 库读取

```python
import webdataset as wds
from PIL import Image
import io
import pickle

dataset = (
    wds.WebDataset(
        "/data_tmp/data/vlm_datasets/data/anker_video_clips_wds_testset/shard-{000000..000022}.tar"
    )
    .to_tuple("pyd")
    .map(lambda x: pickle.loads(x[0]))
)

for data in dataset:
    frames = [Image.open(io.BytesIO(f)) for f in data["frames"]]
    print(f"video: {data['video_rel']}, frames: {len(frames)}")
    break
```

### 使用 tarfile + pickle 直接读取

```python
import tarfile
import pickle
from PIL import Image
import io

tar_path = "/data_tmp/data/vlm_datasets/data/anker_video_clips_wds_testset/shard-000000.tar"

with tarfile.open(tar_path) as tf:
    for member in tf.getmembers():
        if member.name.endswith('.pyd'):
            fp = tf.extractfile(member)
            data = pickle.load(fp)

            # 解析帧
            frames = [Image.open(io.BytesIO(f)) for f in data["frames"]]

            print(f"Video: {data['video_rel']}")
            print(f"Num frames: {data['num_frames']}")
            print(f"Frame size: {frames[0].size}, mode: {frames[0].mode}")
            break  # 只读第一个样本
```

### 使用 index.json 定位样本

```python
import json
import tarfile
import pickle

# 加载索引
with open("/data_tmp/data/vlm_datasets/data/anker_video_clips_wds_testset/index.json") as f:
    index = json.load(f)

# 查找特定样本
sample_key = "testset_1k_250730/0006f134-a7da-405e-a47a-7d0e8405bbe3_segment_1"
shard_id = index[sample_key]  # 返回 0

# 根据分片 ID 构造 tar 路径
tar_path = f"/data_tmp/data/vlm_datasets/data/anker_video_clips_wds_testset/shard-{shard_id:06d}.tar"

# 注意: tar 内的文件名用 "__" 替代 "/"
pyd_filename = sample_key.replace("/", "__") + ".pyd"

with tarfile.open(tar_path) as tf:
    member = tf.getmember(pyd_filename)
    fp = tf.extractfile(member)
    data = pickle.load(fp)
    print(f"Loaded: {data['video_rel']}, frames: {data['num_frames']}")
```

---

## 📝 注意事项

1. **标注与数据的对应**: 标注文件中的 `videos[0]` 字段与 `index.json` 的 key 以及 `.pyd` 中的 `video_rel` 字段完全一致。

2. **Key 与文件名的转换**: index.json 中的 key 使用 `/` 分隔,但 tar 内的 `.pyd` 文件名使用 `__` 分隔。定位样本时需要转换。

3. **混合 train/test**: 尽管数据集名为 `testset`,实际包含了多个 `trainset_*` 子目录的数据。使用时需根据 `video_rel` 前缀过滤。

4. **帧数固定**: 所有样本均为 16 帧,不存在变长序列。

5. **图像尺寸固定**: 所有帧均为 384×384 RGB JPEG,无需 resize。

6. **标签格式**: 标签为结构化文本 `RoleType|Sub-Keyscene|description`,使用 `|` 分隔三个字段。

7. **Pickle 安全**: `.pyd` 文件使用 `pickle.load()` 加载,仅适用于可信数据来源。

---

*生成时间: 2026-07-08*
*生成者: yajie.hou@anker-in.com*
