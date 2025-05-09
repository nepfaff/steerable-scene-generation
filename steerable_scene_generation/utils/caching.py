from functools import lru_cache, wraps

import torch

from steerable_scene_generation.algorithms.common.dataclasses import (
    PlantSceneGraphCache,
    SceneVecDescription,
)


def have_objects_in_scene_changed(
    scene: torch.Tensor,
    cache: PlantSceneGraphCache,
    scene_vec_desc: SceneVecDescription,
) -> bool:
    """
    Returns True if the object types in `scene` are different from the object types in
    `cache` and False otherwise.
    """
    for i, obj in enumerate(scene):
        model_path = scene_vec_desc.get_model_path(
            i if scene_vec_desc.model_path_vec_len is None else obj
        )

        if model_path != cache.object_model_paths[i]:
            return True
    return False


def is_hashable(obj):
    """Check if an object is hashable."""
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def conditional_cache(argument_name: str):
    """
    Decorator to conditionally cache the result of a function based on the value of a
    specific argument.

    This decorator uses functools.lru_cache to cache the result of the function.
    Caching is controlled by the value of the specified argument. If the argument's
    value evaluates to True, the result is cached; otherwise, the result is not cached.

    Note that only hashable arguments are chached.

    Args:
        argument_name (str): The name of the argument to use for caching.
    """

    def decorator(func):
        @lru_cache()
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_it = kwargs.get(argument_name)
            if cache_it is None:
                # Try to get the argument from positional arguments if not found in
                # kwargs.
                arg_names = func.__code__.co_varnames
                arg_index = arg_names.index(argument_name)
                cache_it = args[arg_index] if len(args) > arg_index else False

            if cache_it:
                # Check if all arguments are hashable
                if all(is_hashable(arg) for arg in args) and all(
                    is_hashable(v) for v in kwargs.values()
                ):
                    return cached_func(*args, **kwargs)
                else:
                    print("Arguments are not hashable. Skipping cache.")
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
