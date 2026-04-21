# only one type of rule NEAR(A,B) --> C

def task_generator(depth : int, prune_prob : float, num_distractor_rules : int, 
                   num_distractor_objects : int, sample_distractor_rules : bool,
                   object_types : list[int], object_colors : list[int]) -> list:
    
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if not (0.0 <= prune_prob <= 1.0):
        raise ValueError("prune_prob must be in [0, 1]")
    if num_distractor_rules < 0:
        raise ValueError("num_distractor_rules must be >= 0")
    if num_distractor_objects < 0:
        raise ValueError("num_distractor_objects must be >= 0")
    
def task_tree(depth : int, current_depth : int, prune_prob : float):
    if current_depth > depth:
        return None 
    
