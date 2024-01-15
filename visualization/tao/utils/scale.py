import yaml

import scaleapi

from .paths import SCALE_KEY


def get_scale_key(live: bool):
    with open(SCALE_KEY, 'r') as f:
        return yaml.safe_load(f)['live' if live else 'test']


def create_client(live: bool):
    return scaleapi.ScaleClient(get_scale_key(live))


def list_tasks(client, *args, **kwargs):
    """Get all tasks. Handles pagination for client.tasks()."""
    tasks = []
    assert 'offset' not in kwargs
    offset = 0
    while True:
        kwargs['offset'] = offset
        task_page = client.tasks(*args, **kwargs)
        tasks.extend(task_page)
        if len(task_page) < 100:
            break
        else:
            offset += len(task_page)
        print(f'Fetched {len(tasks)} tasks.')
    return tasks
