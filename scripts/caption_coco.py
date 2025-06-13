import argparse
import logging
from pathlib import Path
import random
import json

from dotenv import dotenv_values, find_dotenv
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from torchvision.io import decode_image
from torchvision.transforms import CenterCrop
from tqdm import tqdm

from hallu_attn.utils import init_seeds
import hallu_attn.constants as constants

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Run Chair evaluation on COCO dataset.")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., llava-1.5-7b, minigpt4, shikra)",
        choices=["llava-1.5-7b"],
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, required=True,
        help="Path to save the output JSON file with image captions."
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic mode for reproducibility."
            "Defaults to False, which is faster but less reproducible.",
    )

    return parser.parse_args()

def load_model(model_name: str):
    if model_name == "llava-1.5-7b":
        handle = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(handle)
        model = LlavaForConditionalGeneration.from_pretrained(
            handle,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    logger.info("LM architecture is %s", model.config.text_config.architectures)
    logger.info("Name/path is %s", model.config.text_config._name_or_path)

    return processor, model

def build_prompt(query, model_name, processor) -> str:
    if model_name.startswith("llava-1.5"):
        system_message = constants.SYSTEM_MESSAGE_LLAVA
    else:
        raise ValueError(f"Model {model_name} is not supported for building prompt.")

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]
        }
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt

def create_coco_loader(coco_path: Path, processor, num_samples=500, batch_size=1):
    coco = COCO(coco_path / "annotations" / "instances_val2014.json")
    image_ids = random.choices(coco.getImgIds(), k=num_samples)

    assert processor.image_processor.do_center_crop, \
        "This model does not use center crop!"
    transform = CenterCrop(processor.image_processor.crop_size["height"])

    class CocoDataset(Dataset):
        def __init__(self, coco, image_ids, transform=transform):
            self.coco = coco
            self.image_ids = image_ids
            self.transform = transform
        def __len__(self):
            return len(self.image_ids)
        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = coco_path / "val2014" / image_info["file_name"]
            image = decode_image(image_path, mode="RGB")
            image = self.transform(image)
            return {
                "image": image,
                "image_id": image_id,
            }

    dataloader = DataLoader(
        CocoDataset(coco, image_ids, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return dataloader

def main(args, env_values):
    device = "cuda"

    processor, model = load_model(args.model)
    coco_loader = create_coco_loader(Path(env_values["COCO_PATH"]), processor, batch_size=4)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    all_outputs = []
    for batch in tqdm(coco_loader):
        prompt = build_prompt(constants.QUERY_CAPTION, args.model, processor)
        texts = [prompt] * len(batch["image_id"])
        inputs = processor(
            text=texts, images=batch["image"], return_tensors="pt"
        ).to(device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)

        # remove the prompt part from the output,
        # following https://github.com/YUECHE77/SPIN/blob/34c0565c9bda5ee46a39c9be90e584605e22c72c/model_loader.py#L302
        output_texts = [text.split("ASSISTANT:",1)[-1].strip() for text in outputs]

        all_outputs.extend([{
                "image_id": image_id.item(),
                "caption": caption,
            }
            for image_id, caption in zip(batch["image_id"], output_texts)
        ])

    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(all_outputs, f, indent=2)

if __name__ == "__main__":
    init_seeds()

    args = parse_args()

    if args.deterministic:
        # see https://docs.pytorch.org/docs/stable/notes/randomness.html
        logger.info("Using deterministic mode for reproducibility.")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        logger.info("Using non-deterministic mode for speed.")

    main(args=args, env_values=dotenv_values(find_dotenv(".env")))