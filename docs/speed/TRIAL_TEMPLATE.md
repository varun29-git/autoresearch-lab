# Short speed trial note

Before GPU: write the trial ID, source commit, one-sentence hypothesis, changed component, expected whole-update gain, and one correctness check.

After GPU: record reference/candidate valid tokens and elapsed seconds, memory, billed VM time, any failure, and the decision: keep exploring, reject, or rerun because the result is close. Keep generated logs and traces outside Git. Commit the interpretation and next hypothesis.

For a serious finalist in this pass, add dense numerical, quality and sustained-speed evidence plus a physics-based 32-GPU transfer analysis. QAT and measured 32-GPU scaling are later production gates, outside this five-hour dense run.
