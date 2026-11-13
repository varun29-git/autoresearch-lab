# Keep-sets with Indic: fertility and step-0 loss

Measured on one AWS `g5.2xlarge` against `s3://nemotron-training-data-us-east-1`. No checkpoint was written. No tokenizer file was saved.
The id lists are the Spark rows that stay. Appended Sarvam pieces follow
those rows, in the order stored beside each list.

Stock tokenizer sha256 `710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c`.
Donor table sha256 `c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29`.
Model `XHToken/Spark-X2.6-1.7B-Base`, loaded in bf16. Instance `i-05b8ebb406c005f4a`.

Method 2's frequency count is the S3 sample below, not the earlier local
HellaSwag cache. Method 1 is still the earliest Spark ids and does not
depend on the corpus. After either English core is chosen, every Spark
row whose surface is in the seven target scripts is force-kept, the merge
ancestors of those rows are restored, and Sarvam pieces that are not
already a kept surface are appended. Method 2 is also reported without
restoring ancestors of the English frequency picks. Those tokens stay in
the list and the pruned merge table cannot emit them.

Each frequency document is capped at 100000 characters. Each evaluation document is the
first 4000 characters. Loss scores at most the first
2048 tokens of that slice. Bits per byte uses the
UTF-8 length of the text that those tokens cover. It is the loss figure
that stays comparable when the tokenizers cut the same text into different
numbers of pieces. Nats per token is the mean next-token loss on each
tokenizer's own sequence.

New rows use the scaled constituent mean: the mean of the stock Spark
rows of that surface, rescaled to the mean norm of those rows. No
optimizer step was run. A lower nats-per-token after a smaller softmax
is not, by itself, evidence that the model got better.

## Sample

| Domain | Role | S3 key | Documents | Stock tokens |
| --- | --- | --- | ---: | ---: |
| english | frequency | `datasets/nemotron-cc-hq/part_000000.parquet` | 28032 | 20013653 |
| code | frequency | `datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/part_000000.parquet` | 7840 | 4010749 |
| math | frequency | `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/part_000000.parquet` | 1920 | 4034883 |
| english | held-out | `datasets/nemotron-cc-hq/part_000400.parquet` | 774 | 300053 |
| code | held-out | `datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/part_000040.parquet` | 264 | 150321 |
| math | held-out | `datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/part_000008.parquet` | 134 | 150040 |
| hi | held-out | `datasets/sangraha/verified/hin/data-0.parquet` | 117 | 80058 |
| mr | held-out | `datasets/sangraha/verified/mar/data-0.parquet` | 101 | 80369 |
| kn | held-out | `datasets/sangraha/verified/kan/data-0.parquet` | 92 | 80003 |
| te | held-out | `datasets/sangraha/verified/tel/data-0.parquet` | 76 | 80436 |
| ta | held-out | `datasets/sangraha/verified/tam/data-0.parquet` | 97 | 80222 |
| gu | held-out | `datasets/sangraha/verified/guj/data-0.parquet` | 90 | 80032 |
| bn | held-out | `datasets/sangraha/verified/ben/data-0.parquet` | 98 | 80862 |

Frequency saw 55474 distinct stock ids across 28059285 tokens.

## Vocabulary

| | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic | Script cut + Indic |
| --- | ---: | ---: | ---: | ---: |
| English core ids | 32101 | 32000 | 33866 | n/a |
| Spark seven-script ids added beyond the core | 3003 | 3842 | 3842 | n/a |
| Ancestors of those rows added beyond the core | 21 | 99 | 99 | n/a |
| Spark rows kept | 35125 | 35941 | 37807 | 77842 |
| Sarvam pieces reused | 3284 | 3266 | 3268 | 3357 |
| Sarvam pieces appended | 36507 | 36525 | 36523 | 36434 |
| Finished vocabulary | 71632 | 72466 | 74330 | 114276 |
| Kept ids with a dropped merge ancestor | 0 | 3229 | 0 | 0 |
| Kept ids in a dropped script | 11816 | 19 | 22 | 0 |

Overlap of the method 2 English core with method 1: 13095.
Overlap of the ancestor-closed method 2 core with method 1: 13882.

## Fertility, tokens per word

| Domain | Stock Spark | Script cut + Indic | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic | Sarvam 65k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| english | 1.3586 | 1.3588 | 1.5124 | 1.4593 | 1.3719 | 1.4396 |
| code | 2.1744 | 2.1747 | 2.4106 | 2.2412 | 2.1781 | 2.4836 |
| math | 1.6228 | 1.6228 | 1.7793 | 1.7168 | 1.6265 | 1.7934 |
| hi | 2.0226 | 1.4500 | 1.4540 | 1.4524 | 1.4521 | 1.4453 |
| mr | 2.9335 | 1.9429 | 1.9494 | 1.9477 | 1.9454 | 1.9385 |
| kn | 3.9467 | 2.5028 | 2.5132 | 2.5109 | 2.5060 | 2.4925 |
| te | 4.5921 | 2.3909 | 2.3949 | 2.3935 | 2.3922 | 2.3822 |
| ta | 4.0654 | 2.5847 | 2.5951 | 2.5916 | 2.5895 | 2.5798 |
| gu | 2.8562 | 1.9986 | 2.0030 | 2.0016 | 2.0007 | 1.9986 |
| bn | 3.1510 | 2.0571 | 2.0596 | 2.0587 | 2.0585 | 2.0466 |

## Fertility ratio versus stock tokenization

Actual encoder length divided by the stock Spark length on the same
held-out slices. The parent-walk column expands a stock id whenever that
id, or one of its merge ancestors, is outside the Spark keep-set. The
loss uses the actual encoder, not the parent walk.

| Domain | Stock Spark | Script cut + Indic | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic | Sarvam 65k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| english | 1.0000 | 1.0001 | 1.1132 | 1.0741 | 1.0098 | 1.0596 |
| code | 1.0000 | 1.0001 | 1.1086 | 1.0307 | 1.0017 | 1.1422 |
| math | 1.0000 | 1.0000 | 1.0965 | 1.0580 | 1.0023 | 1.1052 |
| hi | 1.0000 | 0.7169 | 0.7189 | 0.7181 | 0.7179 | 0.7146 |
| mr | 1.0000 | 0.6623 | 0.6645 | 0.6639 | 0.6632 | 0.6608 |
| kn | 1.0000 | 0.6342 | 0.6368 | 0.6362 | 0.6350 | 0.6315 |
| te | 1.0000 | 0.5206 | 0.5215 | 0.5212 | 0.5209 | 0.5187 |
| ta | 1.0000 | 0.6358 | 0.6384 | 0.6375 | 0.6370 | 0.6346 |
| gu | 1.0000 | 0.6997 | 0.7013 | 0.7008 | 0.7005 | 0.6997 |
| bn | 1.0000 | 0.6528 | 0.6536 | 0.6533 | 0.6533 | 0.6495 |

| Domain | Method 1 parent-walk | Method 2 parent-walk | Method 2 + ancestors parent-walk |
| --- | ---: | ---: | ---: |
| english | 1.1132 | 1.0908 | 1.0104 |
| code | 1.1086 | 1.0456 | 1.0018 |
| math | 1.0965 | 1.0721 | 1.0023 |
| hi | 1.0020 | 1.0012 | 1.0010 |
| mr | 1.0022 | 1.0017 | 1.0008 |
| kn | 1.0036 | 1.0034 | 1.0018 |
| te | 1.0125 | 1.0121 | 1.0120 |
| ta | 1.0055 | 1.0044 | 1.0037 |
| gu | 1.0045 | 1.0039 | 1.0035 |
| bn | 1.0011 | 1.0005 | 1.0005 |

Missing stock ids on the held-out slice, counted against the Spark rows
of that keep-set:

| Domain | Method 1 distinct missing | Method 2 distinct missing | Method 2 + ancestors distinct missing |
| --- | ---: | ---: | ---: |
| english | 12298 | 2109 | 1811 |
| code | 850 | 77 | 74 |
| math | 2189 | 180 | 158 |
| hi | 109 | 55 | 54 |
| mr | 113 | 49 | 48 |
| kn | 125 | 40 | 36 |
| te | 51 | 27 | 25 |
| ta | 120 | 53 | 52 |
| gu | 111 | 52 | 47 |
| bn | 65 | 33 | 31 |

Share of held-out documents whose Spark piece strings match stock.
Indic text is routed, so this check is only on English, code, and math.

| Domain | Script cut | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic |
| --- | ---: | ---: | ---: | ---: |
| english | 0.9897 | 0.0142 | 0.0375 | 0.2674 |
| code | 0.9886 | 0.0000 | 0.0000 | 0.5720 |
| math | 0.9925 | 0.0000 | 0.0000 | 0.3433 |

## Step-0 loss, bits per byte

Mean document bits/byte, with the standard error of that document mean.
Sarvam 65k has no Spark embedding, so it has fertility and no loss.

| Domain | Stock Spark | Script cut + Indic | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic |
| --- | ---: | ---: | ---: | ---: | ---: |
| english | 0.8839 ± 0.0126 | 0.9401 ± 0.0156 | 1.0980 ± 0.0142 | 1.0294 ± 0.0131 | 0.9013 ± 0.0130 |
| code | 0.3337 ± 0.0036 | 0.4720 ± 0.0048 | 0.5618 ± 0.0058 | 0.4270 ± 0.0041 | 0.3359 ± 0.0036 |
| math | 0.2383 ± 0.0043 | 0.2810 ± 0.0043 | 0.4598 ± 0.0054 | 0.3799 ± 0.0064 | 0.2421 ± 0.0044 |
| hi | 0.4853 ± 0.0119 | 1.0024 ± 0.0153 | 0.9860 ± 0.0146 | 0.9850 ± 0.0145 | 0.9847 ± 0.0145 |
| mr | 0.5075 ± 0.0117 | 1.1068 ± 0.0145 | 1.0937 ± 0.0139 | 1.0929 ± 0.0137 | 1.0916 ± 0.0137 |
| kn | 0.5217 ± 0.0116 | 1.1385 ± 0.0168 | 1.1219 ± 0.0169 | 1.1197 ± 0.0170 | 1.1176 ± 0.0165 |
| te | 0.5655 ± 0.0181 | 1.2469 ± 0.0194 | 1.2332 ± 0.0191 | 1.2323 ± 0.0191 | 1.2316 ± 0.0191 |
| ta | 0.4196 ± 0.0135 | 1.0005 ± 0.0137 | 0.9897 ± 0.0132 | 0.9867 ± 0.0131 | 0.9863 ± 0.0131 |
| gu | 0.6593 ± 0.0198 | 1.2649 ± 0.0205 | 1.2506 ± 0.0202 | 1.2488 ± 0.0202 | 1.2484 ± 0.0201 |
| bn | 0.5171 ± 0.0130 | 1.1434 ± 0.0135 | 1.1241 ± 0.0128 | 1.1233 ± 0.0126 | 1.1232 ± 0.0127 |

## Step-0 loss, nats per token

| Domain | Stock Spark | Script cut + Indic | Method 1 + Indic | Method 2 + Indic | Method 2 + ancestors + Indic |
| --- | ---: | ---: | ---: | ---: | ---: |
| english | 2.6696 | 2.8390 | 2.9783 | 2.8940 | 2.6958 |
| code | 0.8057 | 1.1393 | 1.2231 | 0.9999 | 0.8095 |
| math | 0.5909 | 0.6969 | 1.0399 | 0.8906 | 0.5991 |
| hi | 2.1682 | 6.2511 | 6.1317 | 6.1320 | 6.1317 |
| mr | 2.1830 | 7.1924 | 7.0839 | 7.0853 | 7.0848 |
| kn | 1.9589 | 6.7458 | 6.6196 | 6.6129 | 6.6136 |
| te | 1.7769 | 7.5313 | 7.4357 | 7.4348 | 7.4345 |
| ta | 1.7638 | 6.6177 | 6.5201 | 6.5092 | 6.5116 |
| gu | 2.4207 | 6.6387 | 6.5491 | 6.5445 | 6.5451 |
| bn | 1.9470 | 6.5988 | 6.4798 | 6.4777 | 6.4777 |

## Recommendation

Among the three 32k-English plus Indic tokenizers, Method 2 + ancestors + Indic stays closest to stock English. Its fertility ratio is 1.0098 and its step-0 loss is 0.9013 bits/byte, against stock 0.8839. The script cut keeps every Spark piece outside the dropped scripts. Its English fertility ratio is 1.0001 and its step-0 loss is 0.9401 bits/byte, at 114276 rows. Indic step-0 loss is the loss of the copied Spark rows plus the untrained new rows, before any alignment step.

Lists:

- `reports/s3_keep_method1_plus_indic.json`
- `reports/s3_keep_method1_appended_pieces.json`
- `reports/s3_keep_method2_plus_indic.json`
- `reports/s3_keep_method2_appended_pieces.json`
- `reports/s3_keep_method2_ancestors_plus_indic.json`
- `reports/s3_keep_method2_ancestors_appended_pieces.json`

