# Short speed trial note

Before GPU: write the trial ID, source commit, one-sentence hypothesis, changed component, expected whole-update gain, and one correctness check.

After GPU: record reference/candidate valid tokens and elapsed seconds, memory, billed VM time, any failure, and the decision: keep exploring, reject, or rerun because the result is close. Keep generated logs and traces outside Git. Commit the interpretation and next hypothesis.

For a serious finalist in this pass, add dense numerical and matched-token quality evidence, sustained one-GPU full-update speed, peak memory and a physics-based explanation of the remaining exposed time. QAT and 32-GPU scaling are outside this five-hour run.
