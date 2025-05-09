import argparse
import copy

from PIL import Image
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    ModelVisualizer,
    PackageMap,
    Parser,
    RenderEngineGltfClientParams,
    RigidTransform,
    Transform,
)


def add_package_maps(
    package_map: PackageMap, package_names: list[str], package_file_paths: list[str]
) -> None:
    for package_name, package_path in zip(package_names, package_file_paths):
        try:
            package_map.Add(package_name, package_path)
        except Exception as e:
            print(f"Failed to add package {package_name} with path {package_path}.")
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "drake_directive_path", type=str, help="Path to the Drake directive file."
    )
    parser.add_argument(
        "--package_names",
        nargs="+",
        type=str,
        default=["tri", "gazebo", "greg"],
        help="An optional list of package names for resolving the model paths.",
    )
    parser.add_argument(
        "--package_file_paths",
        nargs="+",
        type=str,
        default=["data/tri", "data/gazebo", "data/greg"],
        help="An optional list of package file paths to `package.xml` for resolving the model "
        "paths.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the scene. Default is visualization with Meshcat.",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=1920,
        help="The width of the camera for rendering.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=1080,
        help="The height of the camera for rendering.",
    )
    parser.add_argument(
        "--use_blender_server",
        action="store_true",
        help="Whether to use the Blender server for rendering.",
    )
    parser.add_argument(
        "--blender_server_url",
        type=str,
        default="http://127.0.0.1:8000",
        help="The URL of the Blender server for rendering.",
    )
    parser.add_argument(
        "--img_save_path",
        type=str,
        required=None,
        help="The path to save the rendered image to.",
    )
    args = parser.parse_args()
    drake_directive_path = args.drake_directive_path
    package_names = args.package_names
    package_file_paths = args.package_file_paths
    render = args.render
    camera_width = args.camera_width
    camera_height = args.camera_height
    use_blender_server = args.use_blender_server
    blender_server_url = args.blender_server_url
    img_save_path = args.img_save_path

    if render:
        if not use_blender_server:
            raise NotImplementedError(
                "Visualization without Blender server is not implemented yet."
            )
        if img_save_path is None:
            raise ValueError("img_save_path must be provided if render is True.")

        # Build the diagram.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(plant)
        add_package_maps(parser.package_map(), package_names, package_file_paths)
        parser.AddModels(drake_directive_path)
        plant.Finalize()

        # Add camera.
        # Assumes that we are using the camera from the blender server scene and thus
        # don't need a camera pose here.
        camera_config = CameraConfig(
            X_PB=Transform(RigidTransform()),
            width=camera_width,
            height=camera_height,
            renderer_class=(RenderEngineGltfClientParams(base_url=blender_server_url)),
        )
        ApplyCameraConfig(
            config=camera_config, builder=builder, plant=plant, scene_graph=scene_graph
        )
        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            ).color_image_output_port(),
            "rgba_image",
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        print("Rendering image...")
        rgba_image = copy.deepcopy(
            diagram.GetOutputPort("rgba_image").Eval(context).data
        )
        Image.fromarray(rgba_image).save(img_save_path)
        print(f"Saved rendered image to {img_save_path}.")
    else:
        visualizer = ModelVisualizer(publish_contacts=False)
        add_package_maps(visualizer.package_map(), package_names, package_file_paths)
        visualizer.AddModels(drake_directive_path)
        visualizer.Run()


if __name__ == "__main__":
    main()
