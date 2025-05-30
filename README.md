# Steerable Scene Generation with Post Training and Inference-Time Search

Code for "Steerable Scene Generation with Post Training and Inference-Time Search".

#### [[Project Website]](https://steerable-scene-generation.github.io/) | [[Paper]](https://arxiv.org/abs/2505.04831) | [[Overview Video]](https://youtu.be/Ur5I1lZJfAQ)

[Nicholas Pfaff](https://nepfaff.github.io/)<sup>1</sup>,
[Hongkai Dai](https://hongkai-dai.github.io/)<sup>2</sup>,
[Sergey Zakharov](https://zakharos.github.io/)<sup>2</sup>,
[Shun Iwase](https://sh8.io/)<sup>2,3</sup>,
[Russ Tedrake](https://locomotion.csail.mit.edu/russt.html)<sup>1,2</sup> <br/>
<sup>1</sup>Massachusetts Institute of Technology,
<sup>2</sup>Toyota Research Institute, <sup>3</sup>Carnegie Mellon University

![Teaser](assets/teaser.webp)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{pfaff2025_steerable_scene_generation,
    author        = {Pfaff, Nicholas and Dai, Hongkai and Zakharov, Sergey and Iwase, Shun and Tedrake, Russ},
    title         = {Steerable Scene Generation with Post Training and Inference-Time Search},
    year          = {2025},
    eprint        = {2505.04831},
    archivePrefix = {arXiv},
    primaryClass  = {cs.RO},
    url           = {https://arxiv.org/abs/2505.04831}, 
}
```

## Setup

### Installation

This repo uses Poetry for dependency management. First, install
[Poetry](https://python-poetry.org/docs/#installation), and make sure to have Python3.10
installed on your system.

Run the following to ensure that poetry created a virtual environment in `./.venv` for
easy access:
```bash
poetry config virtualenvs.in-project true
```

Next, install all the required dependencies to the virtual environment with the
following command:
```bash
poetry install
```

Activate the environment:
```bash
source .venv/bin/activate
```

#### Wandb

Create a [wandb](https://wandb.ai/site) account for cloud logging and checkpointing.
Run `wandb login` to login.

Then modify the wandb entity (account) in `configurations/config.yaml`.

#### VScode/ Cursor

If using VScode, please modify `.vscode/settings.json` so that the python interpreter is
set correctly.

### Download Model Checkpoints and Object Models

Both model checkpoints and object models are required for inference.

Please download the model checkpoints from
[here](https://mitprod-my.sharepoint.com/:f:/g/personal/nepfaff_mit_edu/ErtxEn8LBjtAl3ur_Q0XISwBMIj5IL0CCPkzrRqrxpbxZQ?e=s5r70P)
and the object models from
[here](https://mitprod-my.sharepoint.com/:f:/g/personal/nepfaff_mit_edu/Es7wAEHXwrFOi1KXUW8pPPsB3ToO68rylBRaQs73IMFOPw?e=yu3WMq).

Move the object models under `data`, creating `data/tri`, `data/greg`, and
`data/gazebo`.

## Dataset Format

We use the [Huggingface dataset](https://huggingface.co/docs/datasets/index) format with
a custom metadata `.json` file.

Please see [scripts/convert_processed_pkl_to_hf.py](scripts/convert_processed_pkl_to_hf.py)
and
[steerable_scene_generation/datasets/scene/scene.py](steerable_scene_generation/datasets/scene/scene.py)
for the dataset format.

Note that DDPM training works best when the entire object vector is normalized while
mixed discrete-continuous diffusion works best when only the continuous part is
normalized. See
[scripts/create_unnormalized_one_hot_vec_dataset_version.py](scripts/create_unnormalized_one_hot_vec_dataset_version.py).

### Procedural Datasets

We generated all our datasets using the procedural pipeline from Greg Izatt. See
[here](https://github.com/nepfaff/spatial_scene_grammars/tree/main/spatial_scene_grammars_examples/dimsum_restaurant)
for our Restaurant (High-Clutter) grammar. Our other grammars are in the same repository.

The generation scripts in that repository will generate pickle files for legacy reasons.
These pickle files can then be converted into our dataset format by first preprocessing
([scripts/preprocess_greg_scene_data.py](scripts/preprocess_greg_scene_data.py)) and
converting to a Huggingface dataset
([scripts/convert_processed_pkl_to_hf.py](scripts/convert_processed_pkl_to_hf.py)).

### Our Scene Dataset

Our datasets are available on the [Hugging Face Hub](https://huggingface.co/nepfaff).

We carefully balanced preprocessing to ensure the datasets are ready to use while
keeping storage requirements reasonable. Specifically:
- All datasets have only the continuous features normalized (see
[Dataset Format](#dataset-format)).
- We provide both individual scene-type datasets and a combined dataset.
- Each dataset includes [language annotations](scripts/create_language_annotations.py),
duplicating every scene three times with different annotation types, except for the
combined dataset, where each scene is assigned a single random annotation type.
- The datasets are directly compatible with the
[released checkpoints](#download-model-checkpoints-and-object-models) of the same name.

## Inference

How to use a trained model. This also applies to running inference with our released
model checkpoints.

### Unconditional Sampling

```bash
python scripts/sample_and_render.py load=checkpoint_path \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene
```
`checkpoint_path` is the local path to the checkpoint.
`dataset_path` is the path to the Huggingface scene dataset directory or the Huggingface
hub dataset ID.
`model_path_vec_len` and `max_num_objects_per_scene` are dataset-specific attributes
and can be found in the dataset metadata.
See the script docstring for additional details.

Note that sampling doesn't require dataset access. If you don't have dataset access,
please specify the path to the metadata `.json` file instead of `dataset_path`. We
provide the metadata files that belong to our checkpoints in `data/metadatas`.

Example of sampling without dataset access:
```bash
python scripts/sample_and_render.py load=data/checkpoints/living_room_shelf.ckpt \
 dataset.processed_scene_data_path=data/metadatas/living_room_shelf.json \
 dataset.model_path_vec_len=19 \
 dataset.max_num_objects_per_scene=23
```
This assumes that you placed the downloaded `living_room_shelf.ckpt` inside
`data/checkpoints`.

### Text-Conditioned Sampling

Same as unconditional sampling but set `algorithm.classifier_free_guidance.weight` to
a value >= 0 and set `algorithm.classifier_free_guidance.sampling.labels` to the
desired language prompt. We recommend CFG weights in [0,1] for the best average results.

### Scene Rearrangement and Completion

```bash
python scripts/rearrange_complete_scenes.py load=checkpoint_path \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene
```
for rearrangement. Specify `+completion=True` for completion.

Note that scene rearrangement and completion requires access to the dataset as the
initial samples are loaded from the dataset. It would be possible to also sample the
initial samples to avoid needing dataset access but this isn't currently implemented.

### Inference-Time Search

```bash
python scripts/inference_time_search.py load=checkpoint_path \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene
```

See `configurations/algorithm/scene_diffuser_base.yaml`/`predict.inference_time_search`
for inference-time search config options. In particular, you might want to reduce
`algorithm.predict.inference_time_search.max_steps` for a faster search.

Note that sampling doesn't require dataset access. If you don't have dataset access,
please specify the path to the metadata `.json` file instead of `dataset_path`. We
provide the metadata files that belong to our checkpoints in `data/metadatas`.

### Post Processing

All of the above example commands don't use post processing. To enable post processing,
specify `algorithm.postprocessing.apply_forward_simulation=True` and
`algorithm.postprocessing.apply_non_penetration_projection=True`. This ensures that the
resulting scenes are physically feasible and ready for robotic simulations.

### Recommended Generation to Physics Simulation Workflow

We currently don't have a good workflow for exporting scenes into various robot
description formats.

We support exporting [Drake](https://drake.mit.edu/) directives but only in test mode
using the `algorithm.test.num_directives_to_generate` option. For example, one might
generate Drake directive outputs as follows:
```bash
python main.py +name=some_test_run_name load=checkpoint_path \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene \
 experiment.tasks=[test] \
 algorithm.test.num_directives_to_generate=10
```
where `experiment.tasks=[test]` specifies to run testing and
`algorithm.test.num_directives_to_generate=10` specifies to sample and export 10 scenes
as Drake directives. Once generated, you can visualize the resulting directives using
[scripts/visualize_drake_directive.py](scripts/visualize_drake_directive.py).
Note that we observed occasional loss in rotation precision when converting to Drake
directives. This might result in some initially physical feasible scenes being no longer
physical feasible after export.

For the best results, we recommend exporting the scenes using our pickle format with
the `algorithm.test.num_samples_to_save_as_pickle` option. This generates a pickle file
with the following dictionary format:
```python
{
    "scenes": sampled_scenes_np,
    "scenes_normalized": sampled_scenes_normalized_np,
    "scene_vec_desc": scene_vec_desc,
}
```
Note that such a pickle file can also be generated with
[scripts/sample_and_render.py](scripts/sample_and_render.py), using the
`+save_pickle=True` flag.
Once exported, you can then use our
[SceneVecDescription](steerable_scene_generation/algorithms/common/dataclasses.py)
interface to access the scene attributes such as its model path, its xyz translation,
or its rotation. For example, the rotation can be obtained in quaternion format using
`scene_vec_desc.get_quaternion(scene)`.
[create_plant_and_scene_graph_from_scene(...)](steerable_scene_generation/utils/drake_utils.py)
shows how one would use SceneVecDescription to create a
[Drake MultibodyPlant](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html)
from a scene. This plant could then be simulated using Drake.

## Training your own models

```bash
python main.py +name=some_training_run_name \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene \
 algorithm.classifier_free_guidance.max_length=max_length
```
where `some_training_run_name` is the required identifying name that will be displayed
on wandb. `dataset_path` is the path to the Huggingface scene dataset directory.
`model_path_vec_len` and `max_num_objects_per_scene` are dataset-specific attributes
and can be found in the dataset metadata.
`max_length` is the maximum number of BERT tokens. Language annotations will be cut off
if they produce more tokens than this. We recommend using a small number for development
runs and a larger one for hero runs. See
[scripts/compute_max_dataset_language_tokens.py](scripts/compute_max_dataset_language_tokens.py)
for determining the token number required to fit the longest dataset annotation.

All our default training configs are optimized for 8 NVIDIA A100 GPUs and you might need
to tweak the parameters slightly when training on a single GPU.

## RL Post Training

First, train a DDPM base model. For more efficient RL optimization, you might decide to
train an unconditional model:
```bash
python main.py +name=some_training_run_name \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene \
 algorithm=scene_diffuser_flux_transformer algorithm.trainer=ddpm \
 algorithm.classifier_free_guidance.use=False \
 algorithm.ema.use=False
```

Then further optimize with reinforcement learning. This might require trying a
few combinations of number of DDIM steps, timestep sampling, and batch size until you
find one that optimally uses your vram capacity.
```bash
python main.py +name=some_training_run_name \
 load=base_model_wandb_id \
 dataset.processed_scene_data_path=model_path_vec_len \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene \
 algorithm=scene_diffuser_flux_transformer \
 algorithm.classifier_free_guidance.use=False \
 algorithm.ema.use=False algorithm.trainer=rl_score \
 algorithm.ddpo.use_object_number_reward=True \
 algorithm.noise_schedule.scheduler=ddim \
 algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
 experiment.training.max_steps=230001 \
 experiment.validation.limit_batch=1 \
 experiment.validation.val_every_n_step=50 \
 algorithm.ddpo.ddpm_reg_weight=200.0 \
 experiment.reset_lr_scheduler=True \
 experiment.training.lr=1e-6 \
 experiment.lr_scheduler.num_warmup_steps=250 \
 algorithm.ddpo.batch_size=32 \
 experiment.training.checkpointing.every_n_train_steps=500 \
 algorithm.num_additional_tokens_for_sampling=20 \
 algorithm.ddpo.n_timesteps_to_sample=100 \
 experiment.find_unused_parameters=True
```
The above is a setup with 150 DDIM steps for sampling of which 100 are sampled at every
RL optimization step. Also notice, that `algorithm.num_additional_tokens_for_sampling=20`
increased the maximum number of objects allowed by the scene representation by 20
compared to what has been seen during pretraining.

We performed post training on 8 NVIDIA A100 80GB GPUs, giving us an effective batch size
of 32*8=256. For single-GPU training, you might want to set
`experiment.training.optim.accumulate_grad_batches` to an integer greater than one for
more stable training with a higher effective batch size.

`some_training_run_name`, `dataset_path`, `model_path_vec_len`,
`max_num_objects_per_scene`, and `max_length` are as [above](#training-your-own-models).

### Baselines

#### DiffuScene Training Example

```bash
python main.py +name=some_training_run_name \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene \
 algorithm.classifier_free_guidance.max_length=max_length \
 algorithm=scene_diffuser_diffuscene algorithm.trainer=ddpm \
 experiment.find_unused_parameters=True
```
`some_training_run_name`, `dataset_path`, `model_path_vec_len`,
`max_num_objects_per_scene`, and `max_length` are as [above](#training-your-own-models).

#### MiDiffusion Training Example

```bash
python main.py +name=some_training_run_name \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene \
 algorithm.classifier_free_guidance.max_length=max_length \
 algorithm=scene_diffuser_mixed_midiffusion
```
`some_training_run_name`, `dataset_path`, `model_path_vec_len`,
`max_num_objects_per_scene`, and `max_length` are as [above](#training-your-own-models).

## Evaluation

We provide the following script for evaluating a single or multiple checkpoints.

```bash
python scripts/evaluate_checkpoints.py load=09j16rpt \
 dataset.processed_scene_data_path=dataset_path \
 dataset.model_path_vec_len=model_path_vec_len \
 dataset.max_num_objects_per_scene=max_num_objects_per_scene
```
where `09j16rpt` is the wandb run ID, whose checkpoints we want to evaluate. Note that
you can also pass a path to a local checkpoint instead of a wandb ID.

By default, this script will evaluate all checkpoints. You can also specify to only
evaluate a single or a list of checkpoints using `checkpoint_version`. Example:
`checkpoint_version=19` or `checkpoint_version=[4,7,9]`. All our paper results were
produced with the last checkpoint only and we observed performance to monotonically
increase if the training dataset is big enough.
Evaluation is done with unconditional generation by default but conditional can be
enabled with `+conditional=True`.

See the script docstring for more details and arguments. The
[integration tests](tests/integration/test_evaluate_checkpoints_script.py) for this
script might also be useful for additional documentation.

#### Note on APF metric

We note that our prompt-following (APF) metrics use the model names for validating
whether the desired prompt models appear in the actual scene. Hence, renaming the models
without regenerating the language annotations will lead to a drop in APF performance.
Please add/ modify entries in `POSSIBLE_OBJECT_STR_TO_NAME` in
[scene_language_annotation.py](steerable_scene_generation/utils/scene_language_annotation.py)
when adding or changing models.

## Codebase Structure

This repo is adopted from [Boyuan Chen](https://boyuan.space/)'s
[research template](https://github.com/buoyancy99/research-template) repo. Please see
that repository for detailed organizational instructions.

## Testing

Run unit tests:
```bash
python -m unittest discover -s ./tests/unit  -p 'test_*.py'
```

Run integration tests:
```bash
python -m unittest discover -s ./tests/integration  -p 'test_*.py'
```

Use the `--failfast` flag to stop on the first failure.

## Figures

The website videos were created with the help of
[drake-blender-recorder](https://github.com/nepfaff/drake-blender-recorder).

See `configurations/algorithm/scene_diffuser_base.yaml`/`visualization` for how to send
render requests to a Blender server instead of rendering with the default VTK renderer.

Note that the brick floor texture that we used for visualizing our restaurant samples
and robot demos is from
[Poliigon](https://www.poliigon.com/texture/reclaimed-dutch-bond-brick-texture-dull-brown/8320).
This texture is completely free but we aren't allowed to redistribute it. Please
download it from there and add it to the `data/greg/models/misc/floor/assets/floor.gltf`
model for the best-looking renders.
