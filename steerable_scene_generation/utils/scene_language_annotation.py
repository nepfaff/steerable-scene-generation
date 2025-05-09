import enum
import math
import random
import re

from collections import Counter
from typing import List, Tuple

import torch

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.language import (
    format_object_list_with_counts,
    format_object_with_article,
    label_duplicate_objects,
)


class LanguageMode(enum.Enum):
    """Describes the type of language annotation to create."""

    OBJECT_NUMBER = "object_number"
    """Specify the number of objects in the scene."""
    OBJECT_NAMES = "object_names"
    """Randomly sample a subset of objects in the scene and specify their names."""
    SPATIAL_RELATIONS = "spatial_relations"
    """Describe spatial relationships between objects in the scene."""
    ALL = "all"
    """Create one of each annotation type for each scene."""

    def __str__(self):
        return self.value


class SceneType(enum.Enum):
    """Describes the type of scene that is contained in a dataset."""

    TABLE_TOP = "table_top"
    """A scene with a table top."""
    SHELF = "shelf"
    """A scene with a shelf."""
    ROOM = "room"
    """A room-level scene."""
    VARIABLE = "variable"
    """A scene of variable type. The type will be determined by the welded objects."""

    def __str__(self):
        return self.value


POSSIBLE_OBJECT_STR_TO_NAME = {
    "floor": "floor",
    "table": "table",
    "shelf": "shelf",
    "shelves": "shelf",
    "chair": "chair",
    "plate": "plate",
    "bowl": "bowl",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "chopstick": "chopstick",
    "utensil_crock": "utensil crock",
    "teapot": "teapot",
    "cup": "cup",
    "mug_inomata": "mug",
    "mug_jeans": "water jug",
    "steamer_bottom": "steamer bottom",
    "steamer_top": "steamer lid",
    "apple": "apple",
    "pear": "pear",
    "avocado": "avocado",
    "cereal_box": "cereal box",
    "bread_slice": "bread slice",
    "lamp": "lamp",
    "book": "book",
    "stacking_ring": "stacking ring",
    "wooden_railway": "toy train",
    "nintendo_3ds_game": "nintendo game",
    "coke": "coke can",
    "banana_concentrate": "coke can",  # For backward compatibility
    "tea_bottle": "tea bottle",
    "speaker": "speaker",
    "game": "board game",
    "box": "box",
    "sphere": "sphere",
    "cylinder": "cylinder",
}

# Combine all object strings into a single regex for fast lookup.
OBJECT_NAME_PATTERN = re.compile(
    "("
    + "|".join(
        re.escape(object_str) for object_str in POSSIBLE_OBJECT_STR_TO_NAME.keys()
    )
    + ")",
    re.IGNORECASE,
)

# Objects that can be above other objects in spatial relationships.
ABOVE_OBJECTS = {
    "apple",
    "pear",
    "avocado",
    "bread_slice",
    "bowl",
    "steamer_top",
    "nintendo_3ds_game",
    "game",
}
# Objects that can be below other objects in spatial relationships.
BELOW_OBJECTS = {
    "plate",
    "bread_slice",
    "steamer_bottom",
    "nintendo_3ds_game",
    "game",
    "table",
}
# Objects that can contain other objects.
CONTAINER_OBJECTS = {"bowl"}
# Big objects for room-level scenes.
BIG_OBJECTS = {"table", "shelf", "shelves"}


def get_object_number(object_model_paths: List[str | None]) -> int:
    """
    Computes the number of objects in a scene by counting the number of non-empty
    objects in `object_model_paths`.

    Args:
        object_model_paths (List[str | None]): The model paths of all the objects in a
            scene.
    """
    num_objects = sum(1 for obj in object_model_paths if obj is not None)
    return num_objects


def get_object_number_annotation(object_model_paths: List[str | None]) -> str:
    """
    Get the object number annotation for a scene.
    """
    # Count the number of non-empty objects.
    num_objects = get_object_number(object_model_paths)

    # Create the language annotation.
    return f"A scene with {num_objects} objects."


def check_object_number_format(prompt: str) -> Tuple[bool, int | None]:
    """
    Check if the prompt matches the format: "A scene with {num_objects} objects."

    Args:
        prompt (str): The string to check.

    Returns:
        Tuple[bool, int|None]:
            bool: True if the string matches the format, False otherwise.
            int | None: The number of objects if the format matches, None otherwise.
    """
    pattern = r"^A scene with (\d+) objects\.$"
    match = re.match(pattern, prompt)
    if match:
        num_objects = int(match.group(1))
        return True, num_objects
    return False, None


def extract_object_names(object_model_paths: List[str | None]) -> List[str]:
    """
    Extract object names from the given model paths using a predefined pattern.
    Each non-None path should result in exactly one object name.

    Args:
        object_model_paths (List[str | None]): List of model paths, where each path
            may be None or a string containing an object name.

    Returns:
        List[str]: List of object names, preserving duplicates. Only None paths are
            excluded.
    """
    object_names = []
    for path in object_model_paths:
        if path is None:
            continue
        matches = list(OBJECT_NAME_PATTERN.finditer(path))
        if matches:
            # Use the last match.
            object_str = matches[-1].group(1).lower()
            if object_str in POSSIBLE_OBJECT_STR_TO_NAME:
                object_names.append(POSSIBLE_OBJECT_STR_TO_NAME[object_str])
            else:
                raise ValueError(
                    f"No corresponding name found for object string: {object_str}"
                )
        else:
            raise ValueError(f"No match found in path: {path}")
    return object_names


def determine_scene_type(
    model_paths: List[str],
    scene_vec_desc: SceneVecDescription,
) -> SceneType:
    """Determine the scene type based on the model paths. This uses a heuristic and
    thus might be incorrect.

    Args:
        model_paths (List[str]): List of model paths in the scene.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        SceneType: The scene type. Returns VARIABLE if the scene type cannot be
            determined.
    """
    # Count welded objects to determine scene type.
    welded_paths = [
        path
        for path in model_paths
        if path is not None and scene_vec_desc.is_welded_object(path)
    ]
    welded_path_to_count = Counter(welded_paths)

    if not welded_path_to_count:
        return SceneType.VARIABLE

    # A room-level scene has more than one welded object.
    num_welded_objects = sum(welded_path_to_count.values())
    if num_welded_objects > 1:
        return SceneType.ROOM

    # Determine scene type based on the welded object name.
    most_frequent_welded_path = max(welded_path_to_count, key=welded_path_to_count.get)
    if (
        "shelf" in most_frequent_welded_path.lower()
        or "shelves" in most_frequent_welded_path.lower()
    ):
        return SceneType.SHELF
    elif "table" in most_frequent_welded_path.lower():
        return SceneType.TABLE_TOP
    else:
        return SceneType.VARIABLE


def sample_object_names(
    object_names: List[str],
    min_number: int,
    max_number: int,
    prioritize_big_objects: bool = False,
) -> List[str]:
    """
    Randomly sample a subset of object names.

    Args:
        object_names (List[str]): The list of object names to sample from.
        min_number (int): The minimum number of object names to include in the
            annotation.
        max_number (int): The maximum number of object names to include in the
            annotation.
        prioritize_big_objects (bool): Whether to prioritize big objects in the
            sampling. This will only sample from non-big objects if there are fewer
            non-big objects than `min_number`.
    """
    num_objects_to_describe = random.randint(min_number, max_number)

    if prioritize_big_objects:
        # Split into big and non-big objects.
        big_objects = [name for name in object_names if name in BIG_OBJECTS]
        other_objects = [name for name in object_names if name not in BIG_OBJECTS]

        # Sample from big objects first, up to num_objects_to_describe.
        if big_objects:
            num_big_objects = min(len(big_objects), num_objects_to_describe)
            indices = torch.randperm(len(big_objects))[:num_big_objects]
            sampled_names = [big_objects[i] for i in indices]
        else:
            sampled_names = []

        # If we have remaining slots, sample from other objects.
        remaining_slots = num_objects_to_describe - len(sampled_names)
        if remaining_slots > 0 and other_objects:
            num_additional = min(remaining_slots, len(other_objects))
            indices = torch.randperm(len(other_objects))[:num_additional]
            sampled_names.extend(other_objects[i] for i in indices)

        return sampled_names
    else:
        object_indices = torch.randperm(len(object_names))[:num_objects_to_describe]
        return [object_names[i] for i in object_indices]


def get_object_name_annotation(
    object_model_paths: List[str | None],
    min_number: int = 1,
    subset_probability: float = 0.5,
    prioritize_big_objects: bool = False,
) -> str:
    """
    Generate a textual annotation that lists all or a subset of the objects in the
    scene. Subset annotations specify that there are additional objects in the scene
    that are not listed.

    Args:
        object_model_paths (List[str]): A list of paths corresponding to object models
            in the scene.
        min_number (int): The minimum number of object names to include in the
            annotation.
        subset_probability (float): The probability that the annotation will list a
            subset of the objects in the scene.
    Returns:
        str: A descriptive sentence listing the objects in the scene.
    """
    object_names = extract_object_names(object_model_paths)

    # Create the language annotation.
    if len(object_names) == 0:
        return "A scene with no objects."
    elif len(object_names) == 1:
        return f"A scene with {format_object_with_article(object_names[0])}."
    else:
        use_subset = random.random() < subset_probability and len(object_names) > 2

        if use_subset:
            # When using a subset, limit the maximum number of objects to describe.
            max_objects_to_describe = min(10, len(object_names) - 1)
            sampled_object_names = sample_object_names(
                object_names=object_names,
                min_number=min_number,
                max_number=max_objects_to_describe,
                prioritize_big_objects=prioritize_big_objects,
            )
            object_names_str = format_object_list_with_counts(sampled_object_names)
            return f"A scene with {object_names_str} and some other objects."
        else:
            # When describing all objects, use the complete list
            object_names_str = format_object_list_with_counts(object_names)
            return f"A scene with {object_names_str}."


def check_object_name_format(prompt: str) -> Tuple[bool, bool]:
    """
    Check if the prompt matches the object name annotation format without spatial
    relationships.

    Args:
        prompt (str): The prompt to check.

    Returns:
        Tuple[bool, bool]:
            bool: True if the prompt matches the format, False otherwise.
            bool: True if the prompt contains a subset of objects, False otherwise.
    """
    # Check for empty scene format.
    if prompt == "A scene with no objects.":
        return True, False

    # Check if there are multiple periods (indicating spatial relationships).
    if prompt.count(".") > 1:
        return False, False

    # Check basic format starts with "A scene with" and ends with period.
    if not prompt.startswith("A scene with ") or not prompt.endswith("."):
        return False, False

    # Check for subset format.
    is_subset = "and some other objects." in prompt

    # Remove the prefix and suffix.
    content = prompt[len("A scene with ") :].rstrip(".")
    if is_subset:
        content = content.replace(" and some other objects", "")

    # Handle the case of a single object.
    if re.match(r"^(?:a|an) [a-z ]+$", content.strip()):
        return True, is_subset

    # Split by commas and "and".
    parts = []
    for part in content.split(","):
        if " and " in part:
            parts.extend(part.split(" and "))
        else:
            parts.append(part)
    parts = [p.strip() for p in parts if p.strip()]

    # Check each part.
    for part in parts:
        if not re.match(
            r"^(?:a|an|two|three|four|five|six|seven|eight|nine|ten|\d+) [a-z ]+$", part
        ):
            return False, False

    return True, is_subset


def extract_object_counts_from_prompt(prompt: str) -> dict[str, int] | None:
    """
    Extract object counts from a prompt that lists objects.

    Args:
        prompt (str): The prompt to parse, e.g. "A scene with two apples and a bowl."

    Returns:
        dict[str, int] | None: Dictionary mapping display names to counts, or None if the
            prompt doesn't list objects.
    """
    # Skip if it's an object number prompt.
    is_obj_num_prompt, _ = check_object_number_format(prompt)
    if is_obj_num_prompt:
        return None

    # Look for content between "with" and period.
    match = re.search(r"with (.*?)\.", prompt)
    if not match:
        return None

    content = match.group(1)
    object_counts = {}

    # Number word mapping.
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    for display_name in set(POSSIBLE_OBJECT_STR_TO_NAME.values()):
        # Check for plural forms with numbers (numeric or word).
        number_alternatives = "|".join(list(number_words.keys()) + [r"\d+"])

        # Handle irregular plurals like "knives" for "knife".
        if display_name == "knife":
            plural_pattern = (
                rf"({number_alternatives}) (?:{re.escape(display_name)}s?|knives)"
            )
        elif display_name == "shelf":
            plural_pattern = (
                rf"({number_alternatives}) (?:{re.escape(display_name)}s?|shelves)"
            )
        else:
            plural_pattern = rf"({number_alternatives}) {re.escape(display_name)}s?"

        plural_matches = re.finditer(plural_pattern, content, re.IGNORECASE)

        for m in plural_matches:
            num_str = m.group(1).lower()
            # Convert word numbers to digits.
            if num_str in number_words:
                num = number_words[num_str]
            else:
                num = int(num_str)
            object_counts[display_name] = num

        # Check for singular forms with articles.
        singular_match = re.search(rf"(?:a|an) {re.escape(display_name)}", content)
        if singular_match and display_name not in object_counts:
            object_counts[display_name] = 1

    return object_counts if object_counts else None


def format_shelf_spatial_relationships(
    translations: List[torch.Tensor],
    sampled_names: List[str],
    sampled_indices: List[int],
    distance_threshold: float = 0.2,
    min_per_axis_threshold: float = 0.05,
    # The default tier heights are for # package://drake_models/manipulation_station/shelves.sdf
    shelf_tier_heights: List[float] = [-0.3995, -0.13115, 0.13115, 0.3995],
    max_shelf_relations_per_tier: int = 2,
) -> List[str]:
    relationships = []
    sampled_translations = [translations[i] for i in sampled_indices]

    # Skip shelf objects from spatial relationships.
    non_shelf_indices = [
        (i, name, trans)
        for i, (name, trans) in enumerate(zip(sampled_names, sampled_translations))
        if name != "shelf"
    ]
    if not non_shelf_indices:
        return relationships

    # Group objects by tier based on their z-coordinate.
    tier_objects: List[List[Tuple[int, str, torch.Tensor]]] = [
        [] for _ in range(len(shelf_tier_heights))
    ]

    # Map tier index to description.
    tier_descriptions = ["bottom", "lower middle", "upper middle", "top"]
    for idx, name, trans in non_shelf_indices:
        z = trans[2].item()
        # Find the shelf tier below the object
        tier = 0  # Default to bottom shelf.
        for i, height in enumerate(shelf_tier_heights):
            if z >= height:
                tier = i

        tier_objects[tier].append((idx, name, trans))

    # Add tier relationships.
    for tier, items in enumerate(tier_objects):
        if items:
            # Randomly sample up to max_shelf_relations_per_tier objects from this tier.
            sampled_items = random.sample(
                items, min(max_shelf_relations_per_tier, len(items))
            )
            for idx, name, _ in sampled_items:
                relationships.append(
                    f"The {name} is on the {tier_descriptions[tier]} shelf."
                )

    # Process spatial relationships within each tier.
    for tier_items in tier_objects:
        if len(tier_items) >= 2:
            # Extract the indices and names for this tier.
            tier_indices = [item[0] for item in tier_items]
            tier_names = [item[1] for item in tier_items]

            # Get spatial relationships for objects in this tier.
            tier_relationships = format_spatial_relationships(
                translations=translations,
                sampled_names=tier_names,
                sampled_indices=tier_indices,
                distance_threshold=distance_threshold,
                min_per_axis_threshold=min_per_axis_threshold,
            )

            relationships.extend(tier_relationships)

    return relationships


def format_room_spatial_relationships(
    translations: List[torch.Tensor],
    names: List[str],
    sampled_names: List[str],
    sampled_indices: List[int],
    distance_threshold: float = 0.2,
    big_object_distance_threshold: float = 4.0,
) -> List[str]:
    """
    Format spatial relationships for room-level scenes.
    It first gets the spatial relationships for big objects, then for other objects.
    """
    # Get big objects.
    big_objects = [
        (name, trans) for name, trans in zip(names, translations) if name in BIG_OBJECTS
    ]
    big_object_names = label_duplicate_objects([obj[0] for obj in big_objects])
    big_object_translations = [obj[1] for obj in big_objects]

    # Get spatial relationships for big objects.
    big_object_relationships = format_spatial_relationships(
        translations=big_object_translations,
        sampled_names=big_object_names,
        sampled_indices=list(range(len(big_object_names))),
        distance_threshold=big_object_distance_threshold,
    )

    # Get spatial relationships for other objects.
    other_object_relationships = format_spatial_relationships(
        translations=translations,
        sampled_names=sampled_names,
        sampled_indices=sampled_indices,
        distance_threshold=distance_threshold,
    )

    relationships = [
        *big_object_relationships,
        *other_object_relationships,
    ]
    return relationships


def format_spatial_relationships(
    translations: List[torch.Tensor],
    sampled_names: List[str],
    sampled_indices: List[int],
    distance_threshold: float = 0.2,
    min_per_axis_threshold: float = 0.05,
) -> List[str]:
    """
    Generate spatial relationship descriptions based on world coordinates.

    This function computes spatial relationships between objects in a scene using their
    3D positions in a global/world coordinate system. The relationships (e.g.,
    "in front of", "to the left of") are determined by relative positions along the x,
    y, and z axes.

    World Coordinate Axes:
        - x-axis:
            - Positive: "in front of"
            - Negative: "behind"
        - y-axis:
            - Positive: "to the right of"
            - Negative: "to the left of"
        - z-axis:
            - Positive: "on top of"
            - Negative: "below"

    Args:
        translations (List[torch.Tensor]): Translation vectors for each object
            representing their positions in the world coordinate system.
        sampled_names (List[str]): Names for sampled objects with unique identifiers for
            duplicates.
        sampled_indices (List[int]): Indices of sampled objects in the original lists.
        distance_threshold (float): Maximum distance to consider for spatial
            relationships.
        min_per_axis_threshold (float): Minimum distance along each axis to consider
            for spatial relationships.

    Returns:
        List[str]: A list of sentences describing spatial relationships between the
        sampled objects.
    """
    relationships = []
    sampled_translations = [translations[i] for i in sampled_indices]

    for i, (name_i, trans_i) in enumerate(zip(sampled_names, sampled_translations)):
        for j, (name_j, trans_j) in enumerate(zip(sampled_names, sampled_translations)):
            if i >= j:
                continue

            # Compute relative position and relationship.
            relative_vector = trans_j - trans_i
            if torch.norm(relative_vector).item() > distance_threshold:
                continue
            dx, dy, dz = relative_vector.tolist()
            angle = math.degrees(math.atan2(dy, dx))

            new_relationships = []

            # Vertical relationships: Above/Inside and Below/Contains.
            if dz > 1e-3:  # Above/Inside
                if name_j in ABOVE_OBJECTS and name_i in BELOW_OBJECTS:
                    new_relationships.append(f"The {name_j} is above the {name_i}.")
                elif name_j in ABOVE_OBJECTS and name_i in CONTAINER_OBJECTS:
                    new_relationships.append(f"The {name_j} is inside the {name_i}.")
            if dz < -1e-3:  # Below/Contains
                if name_j in BELOW_OBJECTS and name_i in ABOVE_OBJECTS:
                    new_relationships.append(f"The {name_j} is below the {name_i}.")
                elif name_j in CONTAINER_OBJECTS and name_i in ABOVE_OBJECTS:
                    new_relationships.append(f"The {name_j} contains the {name_i}.")

            # Horizontal relationships: In front/Behind.
            is_x_dominant = abs(dx) > abs(dy) and abs(dx) > abs(dz)
            if dx > min_per_axis_threshold and is_x_dominant and abs(angle) < 20:
                new_relationships.append(f"The {name_j} is in front of the {name_i}.")
            if dx < -min_per_axis_threshold and is_x_dominant and abs(angle - 180) < 20:
                new_relationships.append(f"The {name_j} is behind the {name_i}.")

            # Horizontal relationships: To the right/To the left.
            is_y_dominant = abs(dy) > abs(dx) and abs(dy) > abs(dz)
            if dy > min_per_axis_threshold and is_y_dominant and abs(angle - 90) < 20:
                new_relationships.append(
                    f"The {name_j} is to the right of the {name_i}."
                )
            if dy < -min_per_axis_threshold and is_y_dominant and abs(angle + 90) < 20:
                new_relationships.append(
                    f"The {name_j} is to the left of the {name_i}."
                )

            # Next to relationship.
            if (
                len(new_relationships) == 0
                and torch.norm(relative_vector).item() < distance_threshold / 2.0
            ):
                new_relationships.append(f"The {name_j} is next to the {name_i}.")

            relationships.extend(new_relationships)

    return relationships


def get_spatial_relation_annotation(
    object_model_paths: List[str],
    translations: List[torch.Tensor],
    scene_type: SceneType,
    subset_probability: float = 0.5,
    max_spatial_relationships: int = 5,
    distance_threshold: float = 0.2,
) -> str:
    """
    This function first creates an object name annotation by listing all or a subset of
    objects from the scene. It then describes spatial relationships between some of
    these listed objects using relative positions (e.g., "on top of", "to the left of").
    The final annotation consists of an object name sentence followed by up to three
    spatial relationship sentences.

    Args:
        object_model_paths (List[str]): A list of paths corresponding to object models
            in the scene.
        translations (List[torch.Tensor]): A list of translation vectors for each
            object, representing their positions in the scene.
        scene_type (SceneType): The type of scene that is contained in the dataset.
        subset_probability (float): The probability that the object name annotation will
            list a subset of the objects in the scene.
        max_spatial_relationships (int): The maximum number of spatial relationships to
            include in the annotation.
        distance_threshold (float): The maximum distance between objects to consider for
            spatial relationships.

    Returns:
        str: A descriptive annotation that includes object names and spatial
            relationships.
    """
    object_names = extract_object_names(object_model_paths)
    if not object_names:
        return "A scene with no objects."

    # Get the object name annotation first.
    object_name_annotation = get_object_name_annotation(
        object_model_paths,
        min_number=2,
        subset_probability=subset_probability,
        prioritize_big_objects=scene_type == SceneType.ROOM,
    )

    # Find which objects were mentioned in the annotation.
    mentioned_objects = []
    mentioned_indices = []
    for i, name in enumerate(object_names):
        if name in object_name_annotation.lower():
            mentioned_objects.append(name)
            mentioned_indices.append(i)

    if len(mentioned_objects) < 2:
        return object_name_annotation

    # Sample a subset of the mentioned objects for spatial relationships.
    num_objects_to_describe = random.randint(2, min(10, len(mentioned_objects)))
    relationship_indices = torch.randperm(len(mentioned_objects))[
        :num_objects_to_describe
    ].tolist()
    sampled_names = [mentioned_objects[i] for i in relationship_indices]
    sampled_indices = [mentioned_indices[i] for i in relationship_indices]

    # Label duplicate objects for relationships.
    labeled_sampled_names = label_duplicate_objects(sampled_names)

    # Generate spatial relationships using only the sampled objects.
    if scene_type == SceneType.SHELF:
        spatial_relationships = format_shelf_spatial_relationships(
            translations=translations,
            sampled_names=labeled_sampled_names,
            sampled_indices=sampled_indices,
            distance_threshold=distance_threshold,
        )
    elif scene_type == SceneType.ROOM:
        spatial_relationships = format_room_spatial_relationships(
            translations=translations,
            names=sampled_names,
            sampled_names=labeled_sampled_names,
            sampled_indices=sampled_indices,
            distance_threshold=distance_threshold,
        )
    else:
        spatial_relationships = format_spatial_relationships(
            translations=translations,
            sampled_names=labeled_sampled_names,
            sampled_indices=sampled_indices,
            distance_threshold=distance_threshold,
        )

    # Optionally include spatial relationships.
    if spatial_relationships:
        # Ensure we don't try to sample more relationships than are available.
        num_relationships_to_include = min(
            random.randint(1, max_spatial_relationships), len(spatial_relationships)
        )
        sampled_relationships = random.sample(
            spatial_relationships, num_relationships_to_include
        )
        return f"{object_name_annotation} {' '.join(sampled_relationships)}"
    return object_name_annotation


def check_spatial_relation_format(prompt: str) -> Tuple[bool, str | None, str | None]:
    """
    Check if the prompt matches the spatial relation annotation format and extract
    components.

    Args:
        prompt (str): The prompt to check.

    Returns:
        Tuple[bool, str | None, str | None]:
            bool: True if the prompt matches the format, False otherwise.
            str | None: The object name portion if format matches, None otherwise.
            str | None: The spatial relations portion if format matches, None otherwise.
    """
    # Split into sentences.
    sentences = prompt.split(". ")
    if len(sentences) == 1:
        return False, None, None

    # Extract object name part (first sentence).
    object_name_str = sentences[0] + "."

    # Check if the first sentence is a valid object name format.
    is_valid_object_format, _ = check_object_name_format(object_name_str)
    if not is_valid_object_format:
        return False, None, None

    # Define spatial relationship keywords
    spatial_keywords = (
        "above|below|inside|contains|in front of|behind|"
        "to the right of|to the left of|next to"
    )

    # Get all sentences except the first one, and handle the last empty string.
    relationship_sentences = [s.strip() for s in sentences[1:] if s.strip()]
    if not relationship_sentences:
        return False, None, None

    # Check each relationship sentence contains at least one spatial keyword.
    for sentence in relationship_sentences:
        if not re.search(spatial_keywords, sentence, re.IGNORECASE):
            return False, None, None

    # Combine spatial relations into a single string if valid.
    # Make sure each sentence ends with exactly one period.
    spatial_relation_str = (
        ". ".join(s.rstrip(".") for s in relationship_sentences) + "."
    )

    return True, object_name_str, spatial_relation_str


def get_language_annotation(
    scene_vec_desc: SceneVecDescription,
    scene: torch.Tensor,
    scene_type: SceneType,
    language_mode: LanguageMode,
    num_spatial_relation_annotations: int = 1,
    subset_probability: float = 0.5,
    max_spatial_relationships: int = 5,
    spatial_relation_distance_threshold: float = 0.2,
) -> List[str]:
    """
    Get the language annotations for a scene based on the language mode.

    Args:
        scene_vec_desc (SceneVecDescription): The scene vector description.
        scene (torch.Tensor): The scene tensor.
        scene_type (SceneType): The type of scene that is contained in the dataset.
        language_mode (LanguageMode): The language mode to use for annotations.
        num_spatial_relation_annotations (int): The number of spatial relation
            annotations to include per scene if the language mode is
            'spatial_relations'.
        subset_probability (float): The probability that the object name annotation will
            list a subset of the objects in the scene.
        max_spatial_relationships (int): The maximum number of spatial relationships to
            include in the annotation.
        spatial_relation_distance_threshold (float): The maximum distance between
            objects to consider for spatial relationships.

    Returns:
        List[str]: The language annotations for the scene.
    """
    object_model_paths = [scene_vec_desc.get_model_path(obj) for obj in scene]

    annotations: List[str] = []
    if language_mode in [LanguageMode.OBJECT_NUMBER, LanguageMode.ALL]:
        annotations.append(get_object_number_annotation(object_model_paths))

    if scene_type == SceneType.VARIABLE:
        # Attempt to determine the scene type.
        scene_type = determine_scene_type(object_model_paths, scene_vec_desc)
        if scene_type == SceneType.VARIABLE:
            # If still can't determine the scene type, return the current annotations.
            return annotations

    if language_mode in [LanguageMode.OBJECT_NAMES, LanguageMode.ALL]:
        annotations.append(
            get_object_name_annotation(
                object_model_paths,
                subset_probability=subset_probability,
                prioritize_big_objects=scene_type == SceneType.ROOM,
            )
        )

    if (
        language_mode in [LanguageMode.SPATIAL_RELATIONS, LanguageMode.ALL]
        and num_spatial_relation_annotations > 0
    ):
        # Get the absolute translations of the objects.
        translations = [scene_vec_desc.get_translation_vec(obj) for obj in scene]

        annotation_candiates = set()
        for _ in range(num_spatial_relation_annotations):
            annotation_candiates.add(
                get_spatial_relation_annotation(
                    object_model_paths=object_model_paths,
                    translations=translations,
                    scene_type=scene_type,
                    subset_probability=subset_probability,
                    max_spatial_relationships=max_spatial_relationships,
                    distance_threshold=spatial_relation_distance_threshold,
                )
            )

        # Add the unique annotations to the list.
        annotations.extend(list(annotation_candiates))

    return annotations
