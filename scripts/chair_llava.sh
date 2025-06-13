CUBLAS_WORKSPACE_CONFIG=:16:8 python scripts/caption_coco.py --model llava-1.5-7b -o outputs/llava_chair.json

python scripts/chair.py --cap_file outputs/llava_chair.json