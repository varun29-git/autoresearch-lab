# Frequent-English keep-sets, about 32k

No Indic pieces were added. No embedding rows were deleted. No tokenizer file was written.
The id lists are the analysis. A later cut can apply one of them.

Stock tokenizer: `research/spark_tokenizer/tokenizer.json`,
sha256 `710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c`.

The CI-1 training bucket is not on this machine, so the frequency count uses
local caches that are already downloaded. They are not the pretraining mix.

| Domain | Frequency documents | Frequency tokens | Held-out documents | Held-out tokens |
| --- | ---: | ---: | ---: | ---: |
| HellaSwag train / validation | 39905 | 7239395 | 1500 | 136591 |
| CodeSearchNet Python train / valid | 30000 | 10449016 | 1500 | 592820 |
| MATH train plus GSM8K train / MATH test plus GSM8K test | 14973 | 3685575 | 600 | 115044 |

Frequency count saw 47427 distinct ids across 21373986 tokens. The forced skeleton is 100 specials (ids 0–13 plus every added token Spark marks special) and 243 byte symbols present in the vocab. Spark does not allocate the other 13 byte symbols.

## Side by side

| | Method 1, earliest ids | Method 2, frequency | Method 2 plus ancestors |
| --- | ---: | ---: | ---: |
| Ids kept | 32101 | 32000 | 33752 |
| Overlap with method 1 | 32101 | 13497 | 14559 |
| Kept ids with a dropped merge ancestor | 0 | 2629 | 0 |
| Kept ids whose surface is a dropped script | 11816 | 214 | 367 |
| Kept ids whose surface is a target Indic script | 839 | 1 | 1 |
| English held-out distinct ids missing | 2141 | 198 | 170 |
| English held-out token occurrences missing | 9.67% | 0.2% | 0.17% |
| Code held-out distinct ids missing | 8002 | 1098 | 997 |
| Code held-out token occurrences missing | 10.5% | 0.58% | 0.52% |
| Math held-out distinct ids missing | 1284 | 140 | 129 |
| Math held-out token occurrences missing | 6.99% | 0.26% | 0.25% |
| English tokens per word, stock | 1.2525 | 1.2525 | 1.2525 |
| English tokens per word, after the cut | 1.382 | 1.2977 | 1.2548 |
| English fertility ratio, cut / stock | 1.1034 | 1.036 | 1.0018 |

A missing id is a stock token on the held-out text that is not in the keep list.
Fertility recomputes the length by walking merge parents whenever a token,
or one of its ancestors, is not kept. A kept id whose parent was dropped
cannot be emitted by the pruned merge table, so it is expanded too.

## Method 1

Non-special means every id outside the special set. The keep list is the
first 32000 of those ids, in id order, union the specials and the byte symbols.
That is 32101 ids. 0 of them have a merge ancestor outside the list. 11816 kept surfaces are in a script this
experiment is not treating as English.

## Method 2

Specials and byte symbols are forced in. The rest of the 32,000 slots are
the most common remaining ids on the frequency sample. Ties break toward
the lower id. The ancestor version starts from that same 32,000 and then
adds every missing merge parent, which brings it to 33752 ids.
Without that restore, 2629 kept ids are unreachable.
With the restore, that count is 0.

## Recommendation

For English, method 2 with ancestors is the safer of these three lists. Its English fertility ratio is 1.0018 and it misses 170 distinct held-out English ids. Method 1 spends keep-slots on whatever Spark merged first, including scripts that are not English. Method 2 spends them on tokens this English, code, and math sample actually emits. The ancestor restore is what makes those frequent tokens buildable. This sample is not the pretraining bucket, so the frequency ranking should be rerun on that bucket before a cut is applied.

Lists:

- `reports/keep_method1_bpe_order.json`
- `reports/keep_method2_frequency.json`
- `reports/keep_method2_frequency_ancestors.json`
