import numpy as np

from env import BaseGridEnv
from tasks import task_generator


def main():
    env = BaseGridEnv(
        size=5,
        max_steps=20,
        object_types=5,
        object_colors=5,
        max_depth=2,
        prune_prob=0.1,
        num_distractor_rules=0,
        num_distractor_objects=0,
    )

    print("=== TEST 1: normal reset ===")
    obs, info = env.reset(seed=123)
    print("obs shape:", obs.shape)
    print("num objects:", len(env.objects))
    print("task keys:", env.task.keys())
    print("leaf objects:", env.task["leaf_task_nodes"])
    print("all task nodes:", env.task["all_task_nodes"])

    assert obs.shape == (5, 5, env.num_channels)
    assert len(env.objects) == len(env.task["leaf_task_nodes"])
    assert env.task is not None

    print("\n=== TEST 2: a few random steps ===")
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={t+1}, action={action}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"num_objects={len(env.objects)}"
        )
        if terminated or truncated:
            break

    print("\n=== TEST 3: fixed externally supplied task ===")
    fixed_task = task_generator(
        max_depth=2,
        prune_prob=0.1,
        num_distractor_rules=0,
        num_distractor_objects=0,
        object_types=list(range(5)),
        object_colors=list(range(5)),
        rng=None,
    )

    obs1, _ = env.reset(seed=999, task=fixed_task)
    task1 = env.task

    obs2, _ = env.reset(seed=555, task=fixed_task)
    task2 = env.task

    print("same task object reused:", task1 == task2)
    print("task after reset with override:", env.task)

    assert task1 == fixed_task
    assert task2 == fixed_task

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()