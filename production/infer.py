"""Generate with an exported two-bit ternary model on CPU."""
import argparse
import sys

import torch

from production.export import load_export
from terminus.generate import generate
from terminus.tokenizer import load_tokenizer
from terminus.tokenizer65k import Frozen65kTokenizer
from terminus.tokenizer_decode import decode_continuation, unused_token_ids


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
    frozen = isinstance(tokenizer, Frozen65kTokenizer)
    stops = (2, 4) if frozen else (tokenizer.eos_id(),)
    suppress = (0, 1, 3) + unused_token_ids(tokenizer) if frozen else ()
    output = generate(model, torch.tensor([ids], dtype=torch.long), max_new_tokens=args.max_new_tokens,
                      eos_token_id=tokenizer.eos_id(), temperature=args.temperature, top_p=args.top_p,
                      stop_token_ids=stops, suppress_token_ids=suppress,
                      generator=torch.Generator().manual_seed(args.seed))
    continuation = output[0, len(ids):].tolist()
    if frozen:
        text, dropped = decode_continuation(tokenizer, ids, continuation, stop_ids=stops,
                                            max_token_cutoff=len(continuation) == args.max_new_tokens)
        if dropped:
            print(f"generation budget ended inside UTF-8; omitted {dropped} trailing bytes", file=sys.stderr)
        print(text)
    else:
        print(tokenizer.decode(continuation))


if __name__ == "__main__":
    main()
