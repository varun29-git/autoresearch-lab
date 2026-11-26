"""New-row alignment. Kept rows, including the seven target scripts, are restored.

English uses KL against the original model after the deleted rows are removed
and the softmax is recomputed. The teacher puts zero mass on every new id.
Indic uses next-token cross-entropy on this encoder's ids. Only the appended
rows are meant to receive that gradient; `restore_kept_rows` copies the
snapshot back because AdamW decay moves a row whose gradient is zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def english_kl(student_logits, teacher_kept_logits):
    """KL(teacher || student). Teacher support is the kept rows only.

    `student_logits` is [..., kept + new]. `teacher_kept_logits` is [..., kept].
    """
    kept = teacher_kept_logits.shape[-1]
    if student_logits.shape[-1] < kept:
        raise ValueError("student logits are shorter than the teacher support")
    teacher_log = F.log_softmax(teacher_kept_logits.float(), dim=-1)
    teacher_prob = teacher_log.exp()
    student_log = F.log_softmax(student_logits.float(), dim=-1)
    # Mass on the new ids is zero in the teacher, so those terms drop out.
    # Their presence in the student softmax is what the kept-term KL penalizes.
    return torch.sum(
        teacher_prob * (teacher_log - student_log[..., :kept]), dim=-1)


def indic_cross_entropy(student_logits, target_ids):
    """Next-token loss on this encoder's ids. Targets may be old or new ids."""
    return F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.shape[-1]),
        target_ids.reshape(-1),
    )


def restore_kept_rows(parameter, snapshot, n_kept):
    """Copy the kept prefix back, including when `parameter` is the fp32 master."""
    if snapshot.shape != parameter.shape:
        raise ValueError("snapshot shape does not match the parameter")
    with torch.no_grad():
        parameter[:n_kept].copy_(snapshot[:n_kept])
