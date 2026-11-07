"""CPU-only index repair handler; deployment and campaign records stay external."""
from production.source_index_repair import repair_object, s3_path


def handler(event, context):
    import boto3
    from botocore.config import Config
    tasks = event['objects']
    if not 1 <= len(tasks) <= 80:
        raise ValueError('an invocation must contain between one and 80 objects')
    prefix = event['immutable_output_prefix']
    s3_path(prefix)
    client = boto3.client('s3', config=Config(connect_timeout=5, read_timeout=25,
        retries={'mode': 'standard', 'max_attempts': 3}))
    results = []
    for task in tasks:
        if context.get_remaining_time_in_millis() < 35000:
            results.append(dict(status='deferred', original_index_uri=task['index_uri']))
            continue
        try:
            result = repair_object(client, task, prefix)
            result['runtime'] = 'aws_lambda_cpu'
        except Exception as exc:
            result = dict(status='rejected', original_index_uri=task['index_uri'],
                          tokens_uri=task['tokens_uri'], source=task['source'],
                          error=type(exc).__name__ + ': ' + str(exc))
        results.append(result)
    return {'results': results}
