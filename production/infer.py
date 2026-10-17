"""Generate with an exported two-bit ternary model on CPU."""
import argparse

import torch

from production.export import load_export
from terminus.generate import generate
from terminus.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    model, tokenizer_path, _ = load_export(args.bundle, tile_rows=args.tile_rows)
    tokenizer = load_tokenizer(tokenizer_path)
    ids = tokenizer.encode(args.prompt)
    if not ids:
        parser.error("prompt must produce at least one token")
    output = generate(model, torch.tensor([ids], dtype=torch.long), max_new_tokens=args.max_new_tokens,
                      eos_token_id=tokenizer.eos_id(), temperature=args.temperature, top_p=args.top_p,
                      generator=torch.Generator().manual_seed(args.seed))
    print(tokenizer.decode(output[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
