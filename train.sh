# 下载数据
gcloud auth activate-service-account --key-file=/home/nas-tpu-poc/gcp-key.json
gcloud config set project nas-tpu-poc
mkdir -p data/zx_vlm_dataset
cd data/zx_vlm_dataset
gcloud storage cp -r gs://zx_vlm_dataset/anker_video_clips_wds_testset/ ./   # 下载测试集
gcloud storage cp -r gs://zx_vlm_dataset/anker_video_clips/ ./    # 下载标签
gcloud storage cp -r gs://zx_vlm_dataset/anker_video_clips_wds_100k  ./  # 下载100k 训练集


# 创建容器
sudo docker pull europe-west4-docker.pkg.dev/leeway-main/anker/jax:env-v1
cd /home/nas-tpu-poc/code/anker_vlm_training
sudo docker run -d \
  --name tpu_train \
  --privileged \
  --net=host \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 \
  -v /dev:/dev \
  -v /home:/home \
  -v $PWD:/workspace \
  -v /home/nas-tpu-poc/code/anker_vlm_training/DATA:/data \
  -w /workspace \
  -p 0.0.0.0:51022:22 \
  -p 0.0.0.0:51080:8000 \
  -p 0.0.0.0:51088:8888 \
  -p 0.0.0.0:51090:9000 \
  --restart unless-stopped \
  europe-west4-docker.pkg.dev/leeway-main/anker/jax:env-v1 \
  tail -f /dev/null


# 生成训练集
python3 -m data.euno_wds \
    --annotation /home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips/euno_train_v3.0.18_balanced_100k_frames.json \
    --wds-dir    /home/nas-tpu-poc/data/zx_vlm_dataset/anker_video_clips_wds_100k \
    --out        DATA/labels.jsonl
wc -l DATA/labels.jsonl        # 应为 98395
