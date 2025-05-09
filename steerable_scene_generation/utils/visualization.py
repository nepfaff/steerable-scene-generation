import copy
import logging
import multiprocessing
import os
import sys

from functools import partial
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from pydrake.all import (
    ApplyCameraConfig,
    ApplyVisualizationConfig,
    CameraConfig,
    DiagramBuilder,
    ImageLabel16I,
    MeshcatVisualizer,
    RenderEngineGltfClientParams,
    RenderEngineVtkParams,
    RenderLabel,
    Rgba,
    RigidTransform,
    Role,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    Transform,
    VisualizationConfig,
)
from tqdm import tqdm

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.utils.drake_utils import (
    create_plant_and_scene_graph_from_scene,
)

console_logger = logging.getLogger(__name__)


def setup_virtual_display_if_needed() -> None:
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        print("Setting up virtual display for rendering.")
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()


def get_visualized_scene_htmls(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    background_color: List[float] = [1.0, 1.0, 1.0],
    simulation_time: float = 1e-6,
    weld_objects: bool = True,
    visualize_proximity: bool = False,
) -> List[str]:
    """
    Visualize a batch of scenes in meshcat and returns the corresponding HTMLs.

    Args:
        scenes (torch.Tensor): A batch of scenes to visualize. Shape (B, N, T+R+M) where
            T is the translation vector length, R is the rotation vector length, and M
            is the model path vector length. The scenes must be inverse normalized.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        background_color (List[float]): The background color of the meshcat window.
        simulation_time (float): The simulation time in seconds.
        weld_objects (bool): Whether to weld the objects to the world frame.
        visualize_proximity (bool): Whether to visualize proximity. This increases the
            HTML file size.
    """
    assert len(scenes.shape) == 3

    if isinstance(scenes, torch.Tensor):
        scenes = scenes.cpu().detach().numpy()

    htmls = []
    for scene in scenes:
        builder = DiagramBuilder()
        result = create_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            scene_vec_desc=scene_vec_desc,
            weld_objects=weld_objects,
            time_step=0.0,
        )

        # Create meshcat.
        meshcat = StartMeshcat()
        meshcat.SetProperty("/Background", "top_color", background_color)
        meshcat.SetProperty("/Background", "bottom_color", background_color)
        meshcat.SetProperty("/Grid", "visible", False)

        if not weld_objects or not visualize_proximity:
            # Add a visualizer to publish the recording.
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder, result.scene_graph, meshcat
            )

        if visualize_proximity:
            config = VisualizationConfig(publish_proximity=True, publish_inertia=False)
            ApplyVisualizationConfig(
                config=config,
                builder=builder,
                meshcat=meshcat,
                plant=result.plant,
                scene_graph=result.scene_graph,
            )

        diagram = builder.Build()

        # Simulate.
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)

        if not weld_objects or not visualize_proximity:
            visualizer.StartRecording()

        simulator.AdvanceTo(simulation_time)

        if not weld_objects or not visualize_proximity:
            visualizer.StopRecording()
            visualizer.PublishRecording()

        html = meshcat.StaticHtml()
        htmls.append(html)

    return htmls


def _determine_camera_pose(
    model_paths: List[str],
    scene_vec_desc: SceneVecDescription,
    camera_poses: dict[str, dict[str, list[float]]] | None = None,
    object_transforms: List[RigidTransform | None] | None = None,
) -> RigidTransform:
    """
    Determine the camera pose based on the scene type.
    The scene type is determined based on the welded scene objects and heuristics.
    Hence, the determined scene type might be incorrect.

    Args:
        model_paths (List[str]): List of model paths in the scene.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        camera_poses (dict[str, dict[str, list[float]]] | None): The mapping from
            scene type to camera poses. Expected format: {
                "scene_type": {
                    "xyz": [x, y, z],
                    "rpy": [r, p, y]
                }
            }
        object_transforms (List[RigidTransform | None] | None): The transforms of the
            objects in the scene. None for empty objects.

    Returns:
        RigidTransform: The camera pose transform.
    """
    default_pose = RigidTransform(
        RollPitchYaw([-3.14, 0.0, 1.57]),
        [0.0, 0.0, 1.5],
    )

    # Count welded objects to determine scene type.
    welded_path_to_count: dict[str, int] = {}
    welded_path_to_transforms: dict[str, list[RigidTransform]] = {}
    for model_path, transform in zip(model_paths, object_transforms):
        if not scene_vec_desc.is_welded_object(model_path):
            continue
        welded_path_to_count[model_path] = welded_path_to_count.get(model_path, 0) + 1
        # Keep track of all transforms for welded objects.
        welded_path_to_transforms[model_path] = welded_path_to_transforms.get(
            model_path, []
        ) + [transform]

    if not welded_path_to_count:
        return default_pose

    # Determine scene type based on most frequent welded object.
    most_frequent_welded_path = max(welded_path_to_count, key=welded_path_to_count.get)

    # Use the transforms to determine whether this is a room-level scene.
    dist_from_origin_for_room_level = 0.5
    scene_type = ""
    for transform in welded_path_to_transforms[most_frequent_welded_path]:
        dist_from_origin = np.linalg.norm(transform.translation())
        if dist_from_origin > dist_from_origin_for_room_level:
            scene_type = "room"
            break

    # TODO: Make this more general.
    if scene_type == "room":
        pass
    elif "cafe_table" in most_frequent_welded_path:
        scene_type = "dimsum_table"
    elif "table" in most_frequent_welded_path:
        scene_type = "tri_table"
    elif "shelves" in most_frequent_welded_path:
        scene_type = "shelf"
    elif "cylinder" in most_frequent_welded_path:
        # Used for integration tests.
        scene_type = "test"
    else:
        console_logger.warning(
            f"Unknown scene type: {most_frequent_welded_path}. Using room."
        )
        scene_type = "room"

    # Get camera pose for scene type.
    if camera_poses is not None and scene_type in camera_poses:
        pose_config = camera_poses[scene_type]
        return RigidTransform(RollPitchYaw(pose_config["rpy"]), pose_config["xyz"])

    return default_pose


def get_render(
    scene: np.ndarray,
    scene_vec_desc: SceneVecDescription,
    camera_width: int = 640,
    camera_height: int = 480,
    background_color: List[float] = [1.0, 1.0, 1.0],
    camera_poses: Optional[dict[str, dict[str, list[float]]]] = None,
    use_blender_server: bool = False,
    blender_server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Visualize a scene and return the corresponding RGB image.

    Args:
        scene (np.ndarray): The unormalized scene to visualize.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        background_color (List[float]): The background color of the rendered image.
        camera_poses (dict[str, dict[str, list[float]]]): The mapping from scene
            type to camera poses. Expected format: {
                "scene_type": {
                    "xyz": [x, y, z],
                    "rpy": [r, p, y]
                }
            }
        use_blender_server (bool): Whether to send render requests to a Blender server.
        blender_server_url (str): The URL of the Blender server.

    Returns:
        np.ndarray: The rendered RGBA image. Shape (H, W, 4) where H is the image height
        and W is the image width.
    """
    builder = DiagramBuilder()
    result = create_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        scene_vec_desc=scene_vec_desc,
        weld_objects=True,
        time_step=0.0,
    )

    # Get camera pose based on scene type.
    X_WC = _determine_camera_pose(
        model_paths=result.object_model_paths,
        object_transforms=result.object_transforms,
        scene_vec_desc=scene_vec_desc,
        camera_poses=camera_poses,
    )

    # Add camera.
    camera_config = CameraConfig(
        X_PB=Transform(X_WC),
        width=camera_width,
        height=camera_height,
        background=Rgba(
            background_color[0], background_color[1], background_color[2], 1.0
        ),
        renderer_class=(
            RenderEngineGltfClientParams(base_url=blender_server_url)
            if use_blender_server
            else RenderEngineVtkParams(backend="GLX")
        ),
    )
    ApplyCameraConfig(
        config=camera_config,
        builder=builder,
        plant=result.plant,
        scene_graph=result.scene_graph,
    )
    builder.ExportOutput(
        builder.GetSubsystemByName(
            f"rgbd_sensor_{camera_config.name}"
        ).color_image_output_port(),
        "rgba_image",
    )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    rgba_image = copy.deepcopy(diagram.GetOutputPort("rgba_image").Eval(context).data)
    return rgba_image


def get_scene_renders(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    camera_width: int = 640,
    camera_height: int = 480,
    background_color: List[float] = [1.0, 1.0, 1.0],
    num_workers: int = 1,
    camera_poses: Optional[dict[str, dict[str, list[float]]]] = None,
    use_blender_server: bool = False,
    blender_server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Visualize a batch of scenes and return the corresponding RGB images.

    Args:
        scenes (torch.Tensor): A batch of unormalized scenes to visualize.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        background_color (List[float]): The background color of the rendered image.
        num_workers (int): The number of workers to use for rendering.
        camera_poses (dict[str, dict[str, list[float]]]): The mapping from scene
            type to camera poses. Expected format: {
                "scene_type": {
                    "xyz": [x, y, z],
                    "rpy": [r, p, y]
                }
            }
        use_blender_server (bool): Whether to send render requests to a Blender server.
        blender_server_url (str): The URL of the Blender server.

    Returns:
        np.ndarray: The rendered RGBA images. Shape (B, H, W, 4) where H is the image
        height and W is the image width.
    """
    assert len(scenes.shape) == 3
    assert num_workers >= 1
    assert (
        "DISPLAY" in os.environ or use_blender_server
    ), "Require a display to render images."

    if isinstance(scenes, torch.Tensor):
        scenes = scenes.cpu().detach().numpy()

    get_render_func = partial(
        get_render,
        scene_vec_desc=scene_vec_desc,
        camera_width=camera_width,
        camera_height=camera_height,
        background_color=background_color,
        camera_poses=camera_poses,
        use_blender_server=use_blender_server,
        blender_server_url=blender_server_url,
    )
    if num_workers == 1 or len(scenes) == 1:
        images = []
        for scene in tqdm(scenes, desc="Rendering scenes", leave=False):
            image = get_render_func(scene)
            images.append(image)
    else:
        num_workers = np.min([num_workers, len(scenes), multiprocessing.cpu_count()])
        with multiprocessing.Pool(num_workers) as pool:
            images = pool.map(get_render_func, scenes)

    return np.stack(images, axis=0)


def get_colors() -> np.ndarray:
    """
    Returns 60 unique but potentially perceptually similar RGB colors of shape (60, 3).
    """
    # Get colors from tab20, tab20b, and tab20c.
    colors_tab20 = np.array(plt.cm.tab20.colors)
    colors_tab20b = np.array(plt.cm.tab20b.colors)
    colors_tab20c = np.array(plt.cm.tab20c.colors)

    # Combine all colors into one array.
    colors = np.vstack((colors_tab20, colors_tab20b, colors_tab20c))
    return colors


def select_most_dissimilar_colors(colors: np.ndarray, num_colors: int) -> np.ndarray:
    """
    Select the specified number of most dissimilar colors from a list of colors.

    This function iteratively selects colors that maximize the minimum distance
    from already selected colors, ensuring a diverse set of dissimilar colors.

    Args:
        colors (np.ndarray): An array of RGB colors with shape (M, 3), where M is the
            number of colors.
        num_colors (int): The number of dissimilar colors to select.

    Returns:
        np.ndarray: An array of the most dissimilar colors with shape (num_colors, 3).
    """
    assert num_colors <= len(colors)

    # Start with the first color.
    selected_colors = [colors[0]]

    while len(selected_colors) < num_colors:
        # Find the color with the maximum minimum distance to the selected colors.
        maximum_minimum_distance = -1
        next_color_to_add = None

        for candidate_color in colors:
            if any(
                np.allclose(candidate_color, selected_color)
                for selected_color in selected_colors
            ):
                continue

            # Calculate distances.
            distances_to_selected_colors = np.array(
                [
                    np.linalg.norm(candidate_color - selected_color)
                    for selected_color in selected_colors
                ]
            )

            # Find the minimum distance to any selected color.
            minimum_distance_to_selected = distances_to_selected_colors.min()

            # Update if this color has a greater minimum distance.
            if minimum_distance_to_selected > maximum_minimum_distance:
                maximum_minimum_distance = minimum_distance_to_selected
                next_color_to_add = candidate_color

        if next_color_to_add is not None:
            selected_colors.append(next_color_to_add)

    return np.array(selected_colors)


def convert_label_image_to_rgb_image(
    image: ImageLabel16I,
    label_to_color_mapping: Dict[int, np.ndarray],
    background_color: List[float] = [1.0, 1.0, 1.0],
    backup_color: List[float] = [0.0, 0.0, 0.0],
) -> np.ndarray:
    """
    Convert a label image into an RGB image by converting all reserved labels to
    background and all other labels to their corresponding color in
    `label_to_color_mapping`.
    """
    reserved_labels = [
        RenderLabel.kDoNotRender,
        RenderLabel.kDontCare,
        RenderLabel.kEmpty,
        RenderLabel.kUnspecified,
    ]
    reserved_labels_int = [int(label) for label in reserved_labels]

    image = np.squeeze(image)
    color_image = np.zeros((*image.shape[:2], 3))

    # Assign colors.
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            label = image[i, j]
            if label in reserved_labels_int:
                # Use the background color for all reserved labels.
                color_image[i, j] = background_color
            elif label in label_to_color_mapping:
                color_image[i, j] = label_to_color_mapping[label]
            else:
                color_image[i, j] = backup_color

    return color_image


def get_label_image_render(
    scene: np.ndarray,
    model_colors: np.ndarray,
    scene_vec_desc: SceneVecDescription,
    camera_poses: dict[str, dict[str, list[float]]],
    camera_width: int = 640,
    camera_height: int = 480,
) -> np.ndarray:
    """
    Visualize a scene as a semantic label image (each object type has a different
    color).

    Args:
        scene (np.ndarray): The unormalized scene to visualize. Shape (N, T+R+M) where
            N is the number of objects, T is the translation vector length, R is the
            rotation vector length, and M is the model path vector length.
        model_colors (np.ndarray): The model colors for the label image, of the same
            length as `model_paths`.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        camera_poses (dict[str, dict[str, list[float]]]): The mapping from scene type
            to camera poses. Expected format: {
                "scene_type": {
                    "xyz": [x, y, z],
                    "rpy": [r, p, y]
                }
            }
        camera_width (int): The camera width.
        camera_height (int): The camera height.

    Returns:
        np.ndarray: The rendered image. Shape (H, W, 3) where H is the image height and
            W is the image width.
    """
    assert len(scene_vec_desc.model_paths) == len(model_colors)

    builder = DiagramBuilder()
    result = create_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        scene_vec_desc=scene_vec_desc,
        weld_objects=True,
        time_step=0.0,
    )

    # Create a map from model index to color.
    model_idx_to_color = {}
    for model_path, model_idx in zip(result.object_model_paths, result.model_indices):
        if model_path is None:
            continue
        color_idx = scene_vec_desc.model_paths.index(model_path)
        model_idx_to_color[int(model_idx)] = model_colors[color_idx]

    # Get camera pose based on scene type
    X_WC = _determine_camera_pose(
        model_paths=result.object_model_paths,
        object_transforms=result.object_transforms,
        scene_vec_desc=scene_vec_desc,
        camera_poses=camera_poses,
    )

    # Add camera.
    camera_config = CameraConfig(
        X_PB=Transform(X_WC),
        width=camera_width,
        height=camera_height,
        rgb=False,
        label=True,
        renderer_class=RenderEngineVtkParams(backend="GLX"),
    )
    ApplyCameraConfig(
        config=camera_config,
        builder=builder,
        plant=result.plant,
        scene_graph=result.scene_graph,
    )
    sensor = builder.GetSubsystemByName(f"rgbd_sensor_{camera_config.name}")
    builder.ExportOutput(
        sensor.label_image_output_port(),
        "label_image",
    )

    # Build diagram.
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Make labels correspond to model indices.
    plant = result.plant
    scene_graph = result.scene_graph
    source_id = plant.get_source_id()
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        properties = copy.deepcopy(inspector.GetPerceptionProperties(geometry_id))
        if properties is None:
            continue
        frame_id = inspector.GetFrameId(geometry_id)
        body = plant.GetBodyFromFrameId(frame_id)
        new_label = int(body.model_instance())
        properties.UpdateProperty("label", "id", RenderLabel(new_label))
        scene_graph.RemoveRole(
            scene_graph_context, source_id, geometry_id, Role.kPerception
        )
        scene_graph.AssignRole(scene_graph_context, source_id, geometry_id, properties)

    # Get label image and convert to RGB.
    label_image = copy.deepcopy(diagram.GetOutputPort("label_image").Eval(context).data)
    rgb_image = convert_label_image_to_rgb_image(
        image=label_image, label_to_color_mapping=model_idx_to_color
    )

    return rgb_image


def get_scene_label_image_renders(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    camera_poses: dict[str, dict[str, list[float]]],
    camera_width: int = 640,
    camera_height: int = 480,
    num_workers: int = 1,
) -> np.ndarray:
    """
    Visualize a batch of scenes as semantic label images.

    Args:
        scenes (torch.Tensor): A batch of unormalized scenes to visualize. Shape (B, N,
            T+R+M) where T is the translation vector length, R is the rotation vector
            length, and M is the model path vector length.
        scene_vec_desc (SceneVecDescription): The scene vector description.
        camera_poses (dict[str, dict[str, list[float]]]): The mapping from scene type
            to camera poses. Expected format: {
                "scene_type": {
                    "xyz": [x, y, z],
                    "rpy": [r, p, y]
                }
            }
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        num_workers (int): The number of workers to use for rendering. If 1, rendering
            is done sequentially.

    Returns:
        np.ndarray: The rendered images. Shape (B, H, W, 3) where H is the image height
            and W is the image width.
    """
    assert len(scenes.shape) == 3
    assert num_workers >= 1
    assert "DISPLAY" in os.environ, "Require a display to render images."

    # Associate a color to each model.
    colors = get_colors()
    model_colors = select_most_dissimilar_colors(
        colors=colors, num_colors=len(scene_vec_desc.model_paths)
    )

    if isinstance(scenes, torch.Tensor):
        scenes = scenes.cpu().detach().numpy()

    get_render_func = partial(
        get_label_image_render,
        model_colors=model_colors,
        scene_vec_desc=scene_vec_desc,
        camera_poses=camera_poses,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    if num_workers == 1 or len(scenes) == 1:
        images = []
        for scene in scenes:
            image = get_render_func(scene)
            images.append(image)
    else:
        num_workers = np.min([num_workers, len(scenes), multiprocessing.cpu_count()])
        with multiprocessing.Pool(num_workers) as pool:
            images = pool.map(get_render_func, scenes)

    return np.stack(images, axis=0)
