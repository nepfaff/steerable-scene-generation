import unittest

from steerable_scene_generation.utils.scene_language_annotation import (
    check_object_name_format,
    check_object_number_format,
    check_spatial_relation_format,
    extract_object_counts_from_prompt,
)


class TestSceneLanguageAnnotation(unittest.TestCase):
    def test_check_object_number_format(self):
        # Test valid formats.
        self.assertEqual(
            check_object_number_format("A scene with 5 objects."), (True, 5)
        )
        self.assertEqual(
            check_object_number_format("A scene with 0 objects."), (True, 0)
        )
        self.assertEqual(
            check_object_number_format("A scene with 10 objects."), (True, 10)
        )

        # Test invalid formats.
        self.assertEqual(
            check_object_number_format("A scene with five objects."), (False, None)
        )
        self.assertEqual(
            check_object_number_format("A scene with objects."), (False, None)
        )
        self.assertEqual(
            check_object_number_format("Scene with 5 objects."), (False, None)
        )
        self.assertEqual(
            check_object_number_format("A scene with 5 objects"), (False, None)
        )
        self.assertEqual(check_object_number_format(""), (False, None))

    def test_check_object_name_format(self):
        # Test valid formats.
        self.assertEqual(
            check_object_name_format("A scene with a bowl."), (True, False)
        )
        self.assertEqual(
            check_object_name_format("A scene with an apple."), (True, False)
        )
        self.assertEqual(
            check_object_name_format("A scene with a plate and a bowl."), (True, False)
        )
        self.assertEqual(
            check_object_name_format("A scene with two spoons, a fork, and a knife."),
            (True, False),
        )
        self.assertEqual(
            check_object_name_format(
                "A scene with a bowl, two apples and some other objects."
            ),
            (True, True),
        )

        # Test invalid formats.
        self.assertEqual(check_object_name_format(""), (False, False))
        self.assertEqual(
            check_object_name_format("A scene with a bowl. The bowl is empty."),
            (False, False),
        )
        self.assertEqual(
            check_object_name_format("Scene with a bowl and a plate."), (False, False)
        )
        self.assertEqual(check_object_name_format("A scene with."), (False, False))

    def test_check_spatial_relation_format(self):
        # Test valid formats.
        valid_prompt = "A scene with a bowl and a plate. The plate is next to the bowl."
        self.assertEqual(
            check_spatial_relation_format(valid_prompt),
            (
                True,
                "A scene with a bowl and a plate.",
                "The plate is next to the bowl.",
            ),
        )

        valid_prompt_multiple = (
            "A scene with two spoons and a plate. "
            "The first spoon is to the left of the plate. "
            "The second spoon is to the right of the plate."
        )
        self.assertEqual(
            check_spatial_relation_format(valid_prompt_multiple),
            (
                True,
                "A scene with two spoons and a plate.",
                "The first spoon is to the left of the plate. "
                "The second spoon is to the right of the plate.",
            ),
        )

        valid_prompt_multiple_2 = (
            "A scene with a bowl, a plate, two spoons, a pear, and a knife. "
            "The pear is next to the spoon. "
            "The knife is to the right of the plate. "
            "The spoon is behind the plate. "
            "The knife is to the right of the pear. "
            "The knife is next to the spoon."
        )
        self.assertEqual(
            check_spatial_relation_format(valid_prompt_multiple_2),
            (
                True,
                "A scene with a bowl, a plate, two spoons, a pear, and a knife.",
                "The pear is next to the spoon. "
                "The knife is to the right of the plate. "
                "The spoon is behind the plate. "
                "The knife is to the right of the pear. "
                "The knife is next to the spoon.",
            ),
        )

        # Test invalid formats.
        invalid_prompts = [
            "",  # Empty string
            "A scene with a bowl.",  # No spatial relations
            "The bowl is next to the plate.",  # No object list
            "A scene with. The bowl is next to the plate.",  # Invalid object list
            "A scene with a bowl and a plate. The bowl is floating.",  # Invalid spatial relation
        ]

        for prompt in invalid_prompts:
            self.assertEqual(
                check_spatial_relation_format(prompt),
                (False, None, None),
                f"Failed for prompt: {prompt}",
            )

    def test_extract_object_counts_from_prompt(self):
        # Test valid formats.
        self.assertEqual(
            extract_object_counts_from_prompt("A scene with three cereal boxes."),
            {"cereal box": 3},
        )
        self.assertEqual(
            extract_object_counts_from_prompt("A scene with a bowl and two apples."),
            {"bowl": 1, "apple": 2},
        )
        self.assertEqual(
            extract_object_counts_from_prompt("A scene with 5 plates and a fork."),
            {"plate": 5, "fork": 1},
        )
        self.assertEqual(
            extract_object_counts_from_prompt("A scene with one mug and three spoons."),
            {"mug": 1, "spoon": 3},
        )

        # Test invalid formats.
        invalid_prompts = [
            "A scene with 5 objects.",  # Object number format
            "A scene with some objects.",  # No specific counts
            "A scene.",  # No objects
            "",  # Empty string
        ]
        for prompt in invalid_prompts:
            self.assertIsNone(
                extract_object_counts_from_prompt(prompt),
                f"Failed for prompt: {prompt}",
            )


if __name__ == "__main__":
    unittest.main()
