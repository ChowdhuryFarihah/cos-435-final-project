# only one type of rule NEAR(A,B) --> C
import random 
import numpy as np
from typing import Optional

def task_generator(max_depth : int, prune_prob : float, num_distractor_rules : int, 
                   num_distractor_objects : int, object_types : list[int], 
                   object_colors : list[int], rng: Optional[random.Random] = None) -> list:
    
    if max_depth < 0:
        raise ValueError("depth must be >= 0")
    if not (0.0 <= prune_prob <= 1.0):
        raise ValueError("prune_prob must be in [0, 1]")
    if num_distractor_rules < 0:
        raise ValueError("num_distractor_rules must be >= 0")
    if num_distractor_objects < 0:
        raise ValueError("num_distractor_objects must be >= 0")
    
    rng = rng or random.Random()

    used_objects: set[tuple[int, int]] = set()
    nodes: list[dict] = []

    goal_object_1 = sample_unique_object(object_types=object_types, object_colors=object_colors,
                                       used_objects=used_objects, rng = rng )
    
    goal_object_2 = sample_unique_object(object_types=object_types, object_colors=object_colors,
                                       used_objects=used_objects, rng = rng )
    
    goal_node =  {
        "id": 0,
        "kind": "goal",
        "goal_type": "near",
        "object_1": goal_object_1,
        "object_2": goal_object_2
    }
    nodes.append(goal_node)

    initial_objects  : list[dict] = []
    expand_task_tree(target_object=goal_object_1, parent_id=0, current_depth=0, max_depth=max_depth,
                     prune_prob=prune_prob, nodes=nodes, initial_objects=initial_objects,
                     used_objects=used_objects, object_types=object_types, object_colors=object_colors, 
                     rng=rng)
    expand_task_tree(target_object=goal_object_2, parent_id=0, current_depth=0, max_depth=max_depth,
                     prune_prob=prune_prob, nodes=nodes, initial_objects=initial_objects,
                     used_objects=used_objects, object_types=object_types, object_colors=object_colors, 
                     rng=rng)
    
    return {
        "all_task_nodes": nodes,
        "leaf_task_nodes" : initial_objects
    }


def sample_unique_object(object_types : list[int], object_colors : list[int], 
                         used_objects : set[tuple[int, int]],
                         rng: random.Random)->dict:
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        obj_type = rng.choice(object_types)
        obj_color = rng.choice(object_colors)
        key = (obj_type, obj_color)
        if key not in used_objects:
            used_objects.add(key)
            return {"type": obj_type, "color": obj_color}
        attempt += 1

    raise ValueError("Ran out of unique objects")

def expand_task_tree(target_object: dict, parent_id: int, current_depth: int,
                     max_depth: int, prune_prob: float, nodes: list,
                     initial_objects: list, used_objects: set, object_types: list[int],
                     object_colors: list[int], rng=None) -> None:
    rng = rng or random.Random()

    if current_depth >= max_depth or rng.random() < prune_prob:
        initial_objects.append(target_object)
        return

    input_object_1 = sample_unique_object(object_types, object_colors, used_objects, rng)
    input_object_2 = sample_unique_object(object_types, object_colors, used_objects, rng)

    rule_node = {
        "id": len(nodes),
        "kind": "rule",
        "rule_type": "near",
        "parent": parent_id,
        "output_object": target_object,
        "input_object_1": input_object_1,
        "input_object_2": input_object_2,
    }
    nodes.append(rule_node)

    expand_task_tree(
        target_object=input_object_1,
        parent_id=rule_node["id"],
        current_depth=current_depth + 1,
        max_depth=max_depth,
        prune_prob=prune_prob,
        nodes=nodes,
        initial_objects=initial_objects,
        used_objects=used_objects,
        object_types=object_types,
        object_colors=object_colors,
        rng=rng,
    )

    expand_task_tree(
        target_object=input_object_2,
        parent_id=rule_node["id"],
        current_depth=current_depth + 1,
        max_depth=max_depth,
        prune_prob=prune_prob,
        nodes=nodes,
        initial_objects=initial_objects,
        used_objects=used_objects,
        object_types=object_types,
        object_colors=object_colors,
        rng=rng,
    )