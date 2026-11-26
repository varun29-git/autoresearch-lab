"""Dependency-free policy checks shared by the Lambda and CPU packers."""


def validate_policy(policy):
    if (policy.get('schema')!='ci1_fixed_translation_pairs_v1'
            or policy.get('phase')!='ordinary_dense_stable'
            or policy.get('program_ceiling_content_tokens')!=20_000_000_000
            or type(policy.get('phase_ceiling_content_tokens'))is not int
            or not 0<policy['phase_ceiling_content_tokens']<=20_000_000_000
            or type(policy.get('family_partition_count'))is not int
            or not 1<=policy['family_partition_count']<=64
            or type(policy.get('family_partition_index'))is not int
            or not 0<=policy['family_partition_index']<policy['family_partition_count']
            or not isinstance(policy.get('family_seed'),str) or not policy['family_seed']
            or policy.get('minimum_translation_score')!=95
            or type(policy.get('max_preparation_seconds'))is not int
            or not 1<=policy['max_preparation_seconds']<=1800):
        raise ValueError('invalid bounded dense translation-pair policy')


def validate_program(tasks):
    selected=[t for t in tasks if t.get('plan',{}).get('translation_pairs')]
    if not selected:return
    policies=[t['plan']['translation_pairs']for t in selected]
    for task,p in zip(selected,policies):
        validate_policy(p)
        if (task.get('kind')!='packing' or not task['id'].startswith('materialize-dense-stable-')
                or task.get('fixed_selection_from')!=task['id'].removeprefix('materialize-')
                or task.get('fixed_partition_sizes') or task['plan'].get('reservation_only')):
            raise ValueError('pair policy assigned outside unsplit ordinary dense materialization')
    if (sum(p['phase_ceiling_content_tokens']for p in policies)>20_000_000_000
            or len({p['family_seed']for p in policies})!=1
            or len({p['family_partition_count']for p in policies})!=1
            or policies[0]['family_partition_count']!=len(policies)
            or {p['family_partition_index']for p in policies}!=set(range(len(policies)))):
        raise ValueError('paired program exceeds its global cap or repeats family partitions')
