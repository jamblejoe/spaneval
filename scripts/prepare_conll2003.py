#!/usr/bin/env python3
"""
Stream tner/conll2003 directly from the HuggingFace Hub (no loading script needed),
extract 150 sentences, convert BIO tags to character-span format, and save to
data/train.jsonl (100) and data/eval.jsonl (50).
"""

import json
import random
from pathlib import Path

from huggingface_hub import HfFileSystem

SEED = 42
N_TRAIN = 100
N_EVAL = 50
MIN_CHARS = 100
DATA_DIR = Path(__file__).parent.parent / "data"


def bio_to_spans(
    tokens: list[str], tag_ids: list[int], id_to_tag: dict[int, str]
) -> tuple[str, list[dict]]:
    """Reconstruct plain text and entity spans from BIO-tagged tokens.

    Returns the joined text and a list of entity dicts with keys:
    entity_type, start, end, original_text.
    """
    text = " ".join(tokens)

    char_starts: list[int] = []
    pos = 0
    for token in tokens:
        char_starts.append(pos)
        pos += len(token) + 1  # +1 for the space separator

    entities: list[dict] = []
    ent_type: str | None = None
    ent_start: int | None = None
    ent_end: int | None = None

    for i, tag_id in enumerate(tag_ids):
        tag = id_to_tag[tag_id]

        if tag.startswith("B-"):
            if ent_type is not None:
                entities.append(
                    {
                        "entity_type": ent_type,
                        "start": ent_start,
                        "end": ent_end,
                        "original_text": text[ent_start:ent_end],
                    }
                )
            ent_type = tag[2:]
            ent_start = char_starts[i]
            ent_end = char_starts[i] + len(tokens[i])

        elif tag.startswith("I-"):
            ent_end = char_starts[i] + len(tokens[i])

        else:  # O
            if ent_type is not None:
                entities.append(
                    {
                        "entity_type": ent_type,
                        "start": ent_start,
                        "end": ent_end,
                        "original_text": text[ent_start:ent_end],
                    }
                )
                ent_type = None

    # Close any entity that runs to the last token
    if ent_type is not None:
        entities.append(
            {
                "entity_type": ent_type,
                "start": ent_start,
                "end": ent_end,
                "original_text": text[ent_start:ent_end],
            }
        )

    return text, entities


def main() -> None:
    fs = HfFileSystem()

    print("Reading label map …")
    with fs.open("datasets/tner/conll2003/dataset/label.json") as f:
        tag_to_id: dict[str, int] = json.load(f)
    id_to_tag = {v: k for k, v in tag_to_id.items()}

    print("Streaming train split …")
    rows: list[dict] = []
    with fs.open("datasets/tner/conll2003/dataset/train.json") as f:
        for line in f:
            rows.append(json.loads(line))

    rows = [r for r in rows if len(" ".join(r["tokens"])) >= MIN_CHARS]
    print(f"  {len(rows)} sentences after filtering (>= {MIN_CHARS} chars)")

    rng = random.Random(SEED)
    indices = rng.sample(range(len(rows)), N_TRAIN + N_EVAL)
    train_indices = set(indices[:N_TRAIN])
    eval_indices = set(indices[N_TRAIN:])

    DATA_DIR.mkdir(exist_ok=True)

    for split_indices, filename in [
        (train_indices, "train.jsonl"),
        (eval_indices, "eval.jsonl"),
    ]:
        out_path = DATA_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for idx in sorted(split_indices):
                row = rows[idx]
                text, entities = bio_to_spans(row["tokens"], row["tags"], id_to_tag)
                entities = [e for e in entities if e["entity_type"] != "MISC"]
                f.write(
                    json.dumps(
                        {"text": text, "entities": entities}, ensure_ascii=False
                    )
                    + "\n"
                )
        print(f"Saved {len(split_indices)} examples → {out_path}")


if __name__ == "__main__":
    main()
