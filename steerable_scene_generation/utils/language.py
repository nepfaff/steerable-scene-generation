from collections import Counter, defaultdict
from typing import List


def number_to_words(n: int) -> str:
    """Convert a number to its word representation (up to 10)."""
    number_words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return number_words.get(n, str(n))


def starts_with_vowel_sound(word: str) -> bool:
    """Determine if a word starts with a vowel sound."""
    vowels = {"a", "e", "i", "o", "u"}
    word = word.lower()

    # Special cases for "an"
    special_an_words = {
        "hour",
        "honor",
        "honest",
        "heir",
        "herb",
    }  # 'herb' for American English
    if word in special_an_words:
        return True

    # Special cases for "a"
    special_a_words = {"unicorn", "university", "unique", "unit", "one", "once"}
    if word in special_a_words:
        return False

    # Handle acronyms/initialisms
    if len(word) == 1 and word[0].isalpha():
        return word[0] in vowels  # Single-letter acronyms (e.g., F -> 'an', U -> 'a')

    # General rule: check first letter
    return word[0] in vowels


def get_article(word: str) -> str:
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article


def format_object_with_article(object_name: str) -> str:
    """
    Format an object name with the appropriate article ('a' or 'an').
    """
    article = get_article(object_name)
    return f"{article} {object_name}"


def format_object_list_with_counts(object_names: List[str]) -> str:
    """
    Format a list of object names into a human-readable string with quantities.
    Repeated objects are grouped and their counts are included.
    """
    # Dictionary for irregular plurals.
    irregular_plurals = {
        "box": "boxes",
        "cereal box": "cereal boxes",
        "knife": "knives",
        "shelf": "shelves",
        "leaf": "leaves",
        "wolf": "wolves",
        "life": "lives",
        "person": "people",
        "child": "children",
        "foot": "feet",
        "tooth": "teeth",
        "mouse": "mice",
    }

    # Count occurrences of each object
    object_counts = Counter(object_names)

    # Build the formatted string
    formatted_objects = []
    for obj, count in object_counts.items():
        if count == 1:
            article = get_article(obj)
            formatted_objects.append(f"{article} {obj}")
        else:
            count_word = number_to_words(count)
            # Use irregular plural if it exists, otherwise add 's'
            plural = irregular_plurals.get(obj, f"{obj}s")
            formatted_objects.append(f"{count_word} {plural}")

    # Combine objects into a sentence with appropriate punctuation.
    if len(formatted_objects) == 1:
        return formatted_objects[0]
    elif len(formatted_objects) == 2:
        return f"{formatted_objects[0]} and {formatted_objects[1]}"
    else:
        return f"{', '.join(formatted_objects[:-1])}, and {formatted_objects[-1]}"


def label_duplicate_objects(object_names: List[str]) -> List[str]:
    """
    Add unique identifiers to duplicate objects in a list when needed.

    Args:
        object_names (List[str]): A list of object names, potentially with duplicates.

    Returns:
        List[str]: A list of object names with unique identifiers where needed.
    """
    # Context-sensitive labeling for disambiguation.
    counts = defaultdict(int)
    labeled_objects = []
    number_words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }

    for name in object_names:
        counts[name] += 1
        if counts[name] > 1:
            # Add ordinal only if there is ambiguity (duplicates exist).
            ordinal = number_words.get(counts[name], f"{counts[name]}th")
            labeled_objects.append(f"{ordinal} {name}")
        else:
            labeled_objects.append(name)

    return labeled_objects
