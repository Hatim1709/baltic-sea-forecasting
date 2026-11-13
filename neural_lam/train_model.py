# Standard library
import inspect
import json
import random
import time
from argparse import ArgumentParser

# Third-party
# for logging the model:
import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .config import load_config_and_datastores
from .models import (
    EDM,
    FM,
    SI,
    FlowMap,
    GraphCast,
    GraphCRPS,
    GraphDET,
    GraphEFM,
    GraphFM,
)
from .models.flow_map import add_flow_map_args
from .weather_dataset import WeatherDataModule

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
    "graph_crps": GraphCRPS,
    "graph_det": GraphDET,
    "EDM": EDM,
    "FM": FM,
    "SI": SI,
    "FlowMap": FlowMap,
}

# Flow map training is split in parts, each launched as its own run (with
# --load pointing to the previous part), see the flow_map_* schedulers below:
# part name -> (objective, unroll steps (None = --ar_steps_train),
#               default warmup epochs)
FLOW_MAP_SCHEDULERS = {
    "flow_map_pretrain": ("map", 1, 1),
    "flow_map_finetune": ("map", 1, 5),
    "flow_map_crps": (None, None, 2),
}


# Checkpoints store args (argparse.Namespace) and config objects, which
# torch>=2.6 refuses to unpickle by default. Lightning >= 2.6 takes
# weights_only in fit/test; older versions (e.g. 2.5 in the Berzelius bsf
# env) do not accept the argument and load with weights_only=False.
_CKPT_LOAD_KWARGS = (
    {"weights_only": False}
    if "weights_only" in inspect.signature(pl.Trainer.fit).parameters
    else {}
)


def _limit_batches(value):
    """--limit_*_batches for pl.Trainer: None (all), a fraction or a count"""
    if value is None:
        return None
    return int(value) if value > 1 else float(value)


def build_parser():
    """Build the shared training parser without loading data or a model."""
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graphcast",
        help="Model architecture to train/evaluate (default: graphcast)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP (default: 1)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=str,
        default=["auto"],
        help="Devices to use for training. Can be the string 'auto' or a list "
        "of integer id's corresponding to the desired devices, e.g. "
        "'--devices 0 1'. Note that this cannot be used with SLURM, instead "
        "set 'ntasks-per-node' in the slurm setup (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_dim_grid",
        type=int,
        help=(
            "Dimensionality of hidden representations related "
            "to grid nodes (default: None, use same as hidden_dim)."
        ),
    )
    parser.add_argument(
        "--hidden_dim_edge",
        type=int,
        help=(
            "Dimensionality of hidden representations related to edge nodes "
            "(edge encodings and in edge-level MLPs)"
            "(default: None, use same as hidden_dim_grid)"
        ),
    )
    parser.add_argument(
        "--hidden_dim_mesh_nodes",
        type=int,
        help=(
            "Dimensionality of hidden representations related to mesh nodes "
            "(mesh encodings and in mesh-level MLPs)"
            "(default: None, use same as hidden_dim)"
        ),
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=3,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 3)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in encoder GNN (default: 1)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in prior GNN (default: 1)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--learn_prior",
        type=int,
        default=1,
        help="If the prior should be learned as a mapping from previous state "
        "and forcing, otherwise static with mean 0 (default: 1 (yes))",
    )

    # EDM options
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum value for sigma in EDM (default: 0.002)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=88,
        help="Maximum value for sigma in EDM (default: 88)",
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=1,
        help="Expected standard deviation of the data in EDM (default: 1.0)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7,
        help="Rho parameter in EDM (default: 7)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        help="Sampler to use in EDM (heun/edm/ddpm) (default: heun)",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=20,
        help="Number of steps to use in sampler (default: 20)",
    )
    parser.add_argument(
        "--backbone_model",
        type=str,
        default="graph_diff",
        help="Backbone model to use in EDM (graph_diff) (default: graph_diff)",
    )
    parser.add_argument(
        "--pred_residual",
        action="store_true",
        help="If the EDM model should predict residuals instead of the "
        "next state. NOTE: has no effect for graph_det or graph_crps, which "
        "always use the residual parametrization to match graph_fm",
    )
    parser.add_argument(
        "--channel_mult_noise",
        type=int,
        default=2,
        help="Channel multiplier for noise MLP (default: 2)",
    )

    # Graph-CRPS options
    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="linear",
        help="Type of encoder to use (default: 'linear')",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=32,
        help="Dimension of the noise vector z, 32 in FGN (default: 32)",
    )
    parser.add_argument(
        "--crps_alpha",
        type=float,
        default=0.95,
        help="Alpha parameter for the Almost Fair CRPS (default: 0.95)",
    )
    parser.add_argument(
        "--train_distr_crps",
        action="store_true",
        help="Run training using CRPS-loss distributed across multiple GPUs"
        "(default: False)",
    )
    parser.add_argument(
        "--channel_mult_emb",
        type=int,
        default=2,
        help="Channel multiplier for noise embedding MLP",
    )
    parser.add_argument(
        "--deterministic_noise_value",
        type=float,
        default=0.,
        help="Value for the noise parameter in the deterministisc version."
    )

    # Graph-CRPS sources of stochasticity, see GraphCRPS
    parser.add_argument(
        "--stochasticity",
        type=str,
        nargs="*",
        default=["fgn"],
        choices=["fgn", "mc", "sdl"],
        help="graph_crps sources of ensemble spread, any combination: fgn "
        "(global noise vector in all conditional layer norms), mc (MC "
        "dropout), sdl (noise on every mesh level, stochastic decomposition "
        "layers) (default: fgn)",
    )
    parser.add_argument(
        "--mc_dropout_p",
        type=float,
        default=0.1,
        help="MC dropout rate of --stochasticity mc (default: 0.1)",
    )
    parser.add_argument(
        "--mc_dropout_scope",
        type=str,
        default="processor",
        choices=["processor", "mesh", "all"],
        help="MLPs with MC dropout: processor (the hierarchical processor "
        "GNNs), mesh (also mesh init and read-out GNNs), all (the whole "
        "backbone) (default: processor)",
    )
    parser.add_argument(
        "--sdl_alpha",
        type=float,
        default=0.235,
        help="Fixed amplitude alpha of the SDL noise (default: 0.235)",
    )
    parser.add_argument(
        "--sdl_noise_dim",
        type=int,
        default=32,
        help="Dimension of the latent vector of each SDL (default: 32)",
    )
    parser.add_argument(
        "--sdl_m_init",
        type=float,
        default=1.0,
        help="Initial value of the learned per-channel SDL modulation m "
        "(default: 1.0)",
    )
    parser.add_argument(
        "--sdl_points",
        type=str,
        default="down",
        choices=["down", "init"],
        help="Where the SDL noise is injected: down (on every level in each "
        "processor down sweep), init (once per level after mesh init) "
        "(default: down)",
    )
    parser.add_argument(
        "--sdl_levels",
        type=int,
        nargs="+",
        default=None,
        help="Mesh levels (0 = finest) with SDL noise (default: all)",
    )
    parser.add_argument(
        "--stoch_warmup_epochs",
        type=float,
        default=2,
        help="graph_crps: epochs over which the noise of all sources is "
        "ramped up from 0 (the deterministic model) to full strength, "
        "counted from the start of the crps_finetune schedule (default: 2)",
    )
    parser.add_argument(
        "--crps_epochs",
        type=int,
        default=25,
        help="crps_finetune: epochs of single-step (ar=1) CRPS training, "
        "including the noise ramp (default: 25)",
    )
    parser.add_argument(
        "--crps_ar2_epochs",
        type=int,
        default=5,
        help="crps_finetune: epochs of 2-step rollout CRPS training after "
        "the ar=1 epochs, 0 to skip (default: 5)",
    )
    parser.add_argument(
        "--crps_lr",
        type=float,
        default=1e-4,
        help="crps_finetune: peak learning rate, warmed up from crps_lr/10 "
        "over --warmup_epochs and cosine-annealed back to crps_lr/10 "
        "(default: 1e-4)",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="crps_finetune: first train only the SDL parameters for this "
        "many epochs (needs --stochasticity sdl) (default: 0)",
    )

    # Training options
    parser.add_argument(
        "--ar_steps_train",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss function "
        "(default: 1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-3,
        help="Minimum learning rate for cosine annealing (default: 1e-4)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon", "muon_flat", "soap", "soap_muon"],
        help="Optimizer to use. 'muon' applies the Muon optimizer to 2D "
        "hidden weights (AdamW for embeddings/output/1D params); 'muon_flat' "
        "additionally flattens 4D conv weights so Muon optimizes them too; "
        "'soap' runs Adam in the eigenbasis of Shampoo's preconditioner for "
        "all parameters; 'soap_muon' adds Muon-style orthogonalization on "
        "top of SOAP for the same 2D hidden weights 'muon' would take "
        "(default: adamw)",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="Momentum for the Muon optimizer (default: 0.95)",
    )
    parser.add_argument(
        "--muon_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the muon optimizer"
        "(default: 0.0)",
    )
    parser.add_argument(
        "--muon_exclude_patterns",
        nargs="+",
        default=[
            "interior_embedder",
            "boundary_embedder",
            "g2m_embedder",
            "m2g_embedder",
            "output_map",
            "mesh_embedders",
            "mesh_same_embedders",
            "mesh_up_embedders",
            "mesh_down_embedders",
        ],
        help="Parameter-name substrings excluded from the Muon (and "
        "SOAP-Muon) parameter group, i.e. input embeddings and the final "
        "output layer",
    )
    parser.add_argument(
        "--soap_betas",
        type=float,
        nargs=2,
        default=[0.95, 0.95],
        help="Adam betas for the soap optimizer (default: 0.95 0.95)",
    )
    parser.add_argument(
        "--soap_muon_betas",
        type=float,
        nargs=2,
        default=[0.95, 0.98],
        help="Adam betas for the soap_muon optimizer (default: 0.95 0.98)",
    )
    parser.add_argument(
        "--soap_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the soap and soap_muon optimizers. Matches "
        "the muon default here; upstream SOAP uses 0.01 (default: 0.0)",
    )
    parser.add_argument(
        "--soap_precondition_frequency",
        type=int,
        default=10,
        help="How often (in steps) soap/soap_muon refresh the preconditioner "
        "eigenbasis. soap_muon tolerates larger values (~40) since the "
        "orthogonalization runs every step (default: 10)",
    )
    parser.add_argument(
        "--soap_muon_disable_sqrt",
        action="store_true",
        help="Disable soap_muon's square-root correction after "
        "orthogonalization (singular-value power 0 instead of 0.5). Changes "
        "the update rule, not the cost: unlike the reference implementation "
        "this one always orthogonalizes with Newton-Schulz, never a full SVD. "
        "Reported to be unstable on some datasets (default: False)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--sample_obs_noise",
        type=int,
        default=0,
        help="If observation noise should be sampled during rollouts (both "
        "training and eval), or just mean prediction used "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=4,
        help="Number of sanity check steps to run before training (default: 4)",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Force gradient checkpointing at each unroll step, also for "
        "single-step phases. It is already enabled automatically whenever "
        "more than one step is unrolled (default: False)",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=None,
        help="Limit the training batches per epoch: a fraction (<= 1.0) or a "
        "number of batches (> 1), e.g. for smoke tests (default: all)",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=None,
        help="Limit the validation batches, as --limit_train_batches "
        "(default: all)",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=float,
        default=None,
        help="Limit the test batches (--eval), as --limit_train_batches, "
        "e.g. 1 to only plot an example (default: all)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Clip gradients to this global norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Linear LR warmup epochs of the flow_map_* and crps_finetune "
        "schedulers (default: 1 for flow_map_pretrain, 5 for "
        "flow_map_finetune, 2 for flow_map_crps, --stoch_warmup_epochs "
        "rounded up for crps_finetune)",
    )
    parser.add_argument(
        "--adaln_zero",
        action="store_true",
        help="GraphDiff backbone: use LN(x) * (1 + scale(emb)) + shift(emb) "
        "with zero-initialised scale/shift in all conditional layer norms, "
        "instead of LN(x) * scale(emb) + shift(emb) (default: False)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=10,
        help="Number of steps to unroll prediction for during evaluation "
        "(default: 10)",
    )
    parser.add_argument(
        "--input_steps",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of state steps as input (default: 1)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--num_latents_plot",
        type=int,
        default=4,
        help="Number of samples of latent variable to plot (default: 4)",
    )

    # Logger Settings
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Logger to use for training (wandb/mlflow) (default: wandb)",
    )
    parser.add_argument(
        "--logger-project",
        type=str,
        default="baltic-sea-forecasting",
        help="Logger project name, for eg. Wandb "
        "(default: baltic-sea-forecasting)",
    )
    parser.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Steps to log val loss for (default: 1 2 3 5 10 15 19)",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--var_leads_val_plot",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to plot during
            validation step (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--current_forcing_step",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include current time t in forcing window (default: 1).",
    )
    parser.add_argument(
        "--num_past_forcing_steps",
        type=int,
        default=2,
        help="Number of past time steps to use as input for forcing data "
        "(default: 2).",
    )
    parser.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=0,
        help="Number of future time steps to use as input for forcing data",
    )
    parser.add_argument(
        "--current_boundary_step",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include current time t in boundary window (default: 1).",
    )
    parser.add_argument(
        "--num_past_boundary_steps",
        type=int,
        default=2,
        help="Number of past time steps to use as boundary input (default: 2).",
    )
    parser.add_argument(
        "--num_future_boundary_steps",
        type=int,
        default=0,
        help="Number of future time steps to use as atmosphere input "
        "(default: 0)",
    )
    parser.add_argument(
        "--current_atmosphere_step",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include current time t in atmosphere window (default: 1).",
    )
    parser.add_argument(
        "--num_past_atmosphere_steps",
        type=int,
        default=2,
        help="Number of past time steps to use as atmosphere input "
        "(default: 2).",
    )
    parser.add_argument(
        "--num_future_atmosphere_steps",
        type=int,
        default=0,
        help="Number of future time steps to use as boundary input "
        "(default: 0)",
    )
    parser.add_argument(
        "--use_atmosphere_g2m",
        action="store_true",
        help="Use atmosphere as separate grid nodes in g2m encoding "
        "(experimental, default: False).",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    parser.add_argument(
        "--save_ensemble_zarr",
        action="store_true",
        help="Save full ensemble forecasts to zarr during test "
        "(default: False)",
    )
    parser.add_argument(
        "--save_forecasts",
        action="store_true",
        help="Save the full test set (all init_times) ensemble forecasts "
        "and targets to a single zarr store, appended along init_time "
        "one batch at a time (default: False)",
    )
    parser.add_argument(
        "--forecasts_zarr_path",
        type=str,
        default=None,
        help="Path to the zarr store used by --save_forecasts. "
        "If not provided, defaults to "
        "{logger.save_dir}/ensemble_forecasts.zarr",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=[
            "pretrain",
            "finetune",
            "probabilistic",
            "deterministic",
            "finetune_deterministic",
            "finetune_ar1",
            "crps_finetune",
            *FLOW_MAP_SCHEDULERS,
        ],
        help="Multi-phase training scheduler. flow_map_pretrain (1 deg, "
        "ar=1, flow-map loss), flow_map_finetune (0.25 deg, ar=1, flow-map "
        "loss) and flow_map_crps (ar=--ar_steps_train, CRPS of 1-jump "
        "rollouts, --fmap_objective crps or map+crps) are the three parts "
        "of the FlowMap schedule, each its own run with --load set to the "
        "checkpoint of the previous part. --epochs is the length of the "
        "part. finetune_ar1: the LR schedule of finetune_deterministic "
        "(60 epochs after a 175-epoch pretraining) but all 60 epochs at "
        "ar=1, for models whose AR training is too expensive (FM). "
        "Re-running a part with --load set to its own checkpoint "
        "resumes or extends it (to --epochs in total). crps_finetune: "
        "probabilistic fine-tuning of graph_crps from a (graph_det) "
        "checkpoint given with --load, --crps_epochs at ar=1 (noise ramped "
        "up over --stoch_warmup_epochs) then --crps_ar2_epochs at ar=2, see "
        "training_scripts/graph_crps/README.md.",
    )
    add_flow_map_args(parser)

    return parser


@logger.catch
def main(input_args=None):
    """Main function for training and evaluating models."""
    parser = build_parser()
    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }
    args.var_leads_val_plot = {
        int(k): v for k, v in json.loads(args.var_leads_val_plot).items()
    }

    # Asserts for arguments
    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load neural-lam configuration and datastore to use
    (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    ) = load_config_and_datastores(config_path=args.config_path)

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        input_steps=args.input_steps,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        current_forcing_step=bool(args.current_forcing_step),
        current_boundary_step=bool(args.current_boundary_step),
        current_atmosphere_step=bool(args.current_atmosphere_step),
        num_past_boundary_steps=args.num_past_boundary_steps,
        num_future_boundary_steps=args.num_future_boundary_steps,
        num_past_atmosphere_steps=args.num_past_atmosphere_steps,
        num_future_atmosphere_steps=args.num_future_atmosphere_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_atmosphere_g2m=args.use_atmosphere_g2m,
        density_channel=getattr(config.training, "density_channel", None),
        train_distr_crps=args.train_distr_crps,
        seed=args.seed,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Set devices to use
    if args.devices == ["auto"]:
        devices = "auto"
    else:
        try:
            devices = [int(i) for i in args.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")

    # Load model parameters Use new args for model
    ModelClass = MODELS[args.model]
    model = ModelClass(
        args,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )

    if args.eval:
        prefix = f"eval-{args.eval}-"
    else:
        prefix = "train-"
    # graph_crps runs are named by their noise sources, e.g. graph_crps_fgn
    model_name = args.model
    if args.model == "graph_crps" and args.stochasticity:
        model_name += "_" + "_".join(args.stochasticity)
    run_name = (
        f"{prefix}{model_name}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    training_logger = utils.setup_training_logger(
        datastore=datastore, args=args, run_name=run_name
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
        )
    )
    if args.model == "graph_crps":
        # val_mean_loss is the loss of a single ensemble member, which
        # favours low spread. Also keep the best ensemble CRPS.
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename="min_val_crps",
                monitor="val_mean_ens_crps",
                mode="min",
            )
        )
    # Separate callback that keeps "last.ckpt" up to date after *every* epoch.
    # A monitored ModelCheckpoint only saves when validation runs, so with
    # --val_interval > 1 (check_val_every_n_epoch != 1, which also makes
    # Lightning save at validation end rather than at train epoch end)
    # last.ckpt would lag up to val_interval epochs behind, and restarting a
    # stage would silently redo that training.
    last_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    callbacks.append(last_ckpt_callback)
    # Log the LR to the training logger, so that a mis-restored schedule after
    # a restart is visible instead of only showing up as a worse loss.
    callbacks.append(
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    )

    # To enable no validation during training, set val_interval to None
    if args.val_interval == 0:
        args.val_interval = None

    def _make_trainer(max_epochs, strategy, extra_callbacks=None):
        """Build a pl.Trainer with the given max_epochs and strategy."""
        cbs = list(callbacks)
        if extra_callbacks:
            cbs.extend(extra_callbacks)
        return pl.Trainer(
            max_epochs=max_epochs,
            deterministic=True,
            strategy=strategy,
            accelerator=device_name,
            num_nodes=args.num_nodes,
            devices=devices,
            logger=training_logger,
            log_every_n_steps=1,
            callbacks=cbs,
            check_val_every_n_epoch=args.val_interval,
            precision=args.precision,
            gradient_clip_val=args.grad_clip,
            gradient_clip_algorithm="norm",
            limit_train_batches=_limit_batches(args.limit_train_batches),
            limit_val_batches=_limit_batches(args.limit_val_batches),
            limit_test_batches=_limit_batches(args.limit_test_batches),
            num_sanity_val_steps=args.num_sanity_val_steps,
            use_distributed_sampler=not args.train_distr_crps,
        )

    def _run_phase(
        model,
        data_module,
        max_epochs,
        strategy,
        ckpt_path=None,
        extra_callbacks=None,
    ):
        """Run one training phase, return path to last checkpoint."""
        t = _make_trainer(max_epochs, strategy,
                          extra_callbacks=extra_callbacks)
        if t.global_rank == 0:
            utils.init_training_logger_metrics(
                training_logger, val_steps=args.val_steps_to_log
            )
        # Checkpoints store args (argparse.Namespace) and config objects,
        # which torch>=2.6 refuses to unpickle by default
        t.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            **_CKPT_LOAD_KWARGS,
        )
        return last_ckpt_callback.last_model_path

    def _completed_epochs(ckpt_path):
        """Number of epochs already completed in the given checkpoint."""
        if not ckpt_path:
            return 0
        try:
            ckpt = torch.load(
                ckpt_path, map_location="cpu", mmap=True, weights_only=False
            )
        except (RuntimeError, TypeError, ValueError):
            # mmap requires a zipfile-serialized checkpoint; fall back to a
            # full read (slow, but only done once at startup).
            ckpt = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
        epoch = ckpt.get("epoch")
        if epoch is None:
            print(
                f"[warning] {ckpt_path} has no 'epoch' entry; assuming no "
                "epochs completed, so the schedule starts at phase 1"
            )
            return 0
        return int(epoch) + 1

    def _run_phases(phases):
        """Run a multi-phase schedule, resuming into the middle of it.

        Parameters
        ----------
        phases : list of (label, max_epochs, setup, strategy)
            ``max_epochs`` is cumulative over the whole schedule (as Lightning
            counts epochs). ``setup`` configures model/data for the phase and
            is called for *every* phase, including phases that the checkpoint
            passed with --load has already completed, so that resuming
            mid-schedule reproduces the configuration of the phase that is
            resumed. Phases already completed are not re-run.
        """
        ckpt_path = args.load
        done = _completed_epochs(ckpt_path)
        if done:
            print(
                f"[{args.scheduler}] Resuming from {ckpt_path} with {done} "
                "epoch(s) already completed"
            )
        for label, max_epochs, setup, strategy in phases:
            setup()
            if done >= max_epochs:
                print(f"[{args.scheduler}] {label} -- already done, skipping")
                continue
            print(f"[{args.scheduler}] {label}")
            new_ckpt = _run_phase(
                model, data_module, max_epochs, strategy, ckpt_path=ckpt_path
            )
            # Empty if the phase saved no checkpoint; keep resuming from the
            # checkpoint we came in with rather than restarting from scratch.
            ckpt_path = new_ckpt or ckpt_path
            done = max(done, max_epochs)

    def _set_ar_steps(n_steps):
        """Set the training unroll length for a phase.

        Gradient checkpointing at each unroll step is enabled whenever more
        than one step is unrolled, since the activations kept for backward
        otherwise scale with the unroll length (a 1-step phase that fits in
        memory can OOM at 2 steps). ``--grad_checkpointing`` forces it on for
        every phase, including 1-step ones.
        """
        data_module.ar_steps_train = n_steps
        model.grad_checkpointing = args.grad_checkpointing or n_steps > 1

    def _run_crps_finetune():
        """Probabilistic (CRPS) fine-tuning of graph_crps, typically from a
        graph_det checkpoint given with --load.

        Phases, counted from the epoch after the --load checkpoint:
        (0. --freeze_backbone_epochs at ar=1, only the SDL parameters
            trained)
        1. --crps_epochs at ar=1. The noise of all sources is ramped up from 0
           (= the deterministic model) over the first --stoch_warmup_epochs,
           while the LR warms up from crps_lr/10 to crps_lr.
        2. --crps_ar2_epochs at ar=2 (gradient checkpointing on).
        The LR is cosine-annealed from crps_lr to crps_lr/10 over all phases.

        --load from a checkpoint of another scheduler (e.g. graph_det from
        finetune_deterministic) starts the fine-tuning after its last epoch,
        with a fresh optimizer. --load from a crps_finetune checkpoint
        resumes it (optimizer and LR schedule restored), also within the
        noise ramp; a larger --crps_epochs then extends phase 1, as long as
        phase 2 has not started.
        """
        assert args.model == "graph_crps", (
            "--scheduler crps_finetune is for --model graph_crps"
        )
        if args.freeze_backbone_epochs > 0:
            assert "sdl" in model.stochasticity, (
                "--freeze_backbone_epochs needs --stochasticity sdl, it only "
                "trains the SDL parameters"
            )

        start_epoch, same_part = 0, False
        if args.load:
            ckpt = torch.load(
                args.load, map_location="cpu", mmap=True, weights_only=False
            )
            loaded_args = ckpt.get("hyper_parameters", {}).get("args")
            same_part = (
                getattr(loaded_args, "scheduler", None) == "crps_finetune"
            )
            done = _completed_epochs(args.load)
            if same_part:
                start_epoch = ckpt.get("crps_stage_start_epoch", 0)
            else:
                start_epoch = done
            del ckpt
            print(
                f"[crps_finetune] --load {args.load}: "
                + (
                    f"resuming (started at epoch {start_epoch}, "
                    f"{done - start_epoch} epoch(s) done)"
                    if same_part
                    else f"starting CRPS fine-tuning after epoch {done}, "
                    "with a fresh optimizer"
                )
            )
        # Keep the optimizer and LR schedule only within the fine-tuning
        model.restore_opt = same_part
        model.crps_stage_start_epoch = start_epoch

        warmup_epochs = (
            args.warmup_epochs
            if args.warmup_epochs is not None
            else int(np.ceil(args.stoch_warmup_epochs))
        )
        warmup_epochs = max(warmup_epochs, 1)
        total_epochs = (
            args.freeze_backbone_epochs
            + args.crps_epochs
            + args.crps_ar2_epochs
        )
        # Per-epoch LR factor (epochs since the start of the fine-tuning,
        # restored with the scheduler state on resume): linear warmup from
        # 1/10 to 1, then cosine back to 1/10 at the end. A LambdaLR built
        # from the current flags (rather than a SequentialLR, which stores
        # T_max in its state), so that extending the run re-stretches the
        # cosine instead of letting the LR rise again after the old end.
        lr_floor = 0.1

        def _lr_factor(epoch):
            if epoch < warmup_epochs:
                return lr_floor + (1 - lr_floor) * epoch / warmup_epochs
            progress = min(
                (epoch - warmup_epochs)
                / max(total_epochs - warmup_epochs, 1),
                1.0,
            )
            return lr_floor + (1 - lr_floor) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )

        _orig_configure = model.configure_optimizers

        def configure_optimizers_crps(self_ref=model):
            opt = _orig_configure()
            optimizer = opt["optimizer"] if isinstance(opt, dict) else opt
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.crps_lr
                param_group["initial_lr"] = args.crps_lr
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }

        model.configure_optimizers = configure_optimizers_crps

        def _freeze_phase():
            model.freeze_backbone = True
            _set_ar_steps(1)

        def _ar1_phase():
            model.freeze_backbone = False
            _set_ar_steps(1)

        def _ar2_phase():
            model.freeze_backbone = False
            _set_ar_steps(2)

        phases = []
        end = start_epoch
        if args.freeze_backbone_epochs > 0:
            end += args.freeze_backbone_epochs
            phases.append(
                (
                    f"Phase 0: {args.freeze_backbone_epochs} ep, ar=1, only "
                    "SDL parameters",
                    end,
                    _freeze_phase,
                )
            )
        end += args.crps_epochs
        phases.append(
            (
                f"Phase 1: {args.crps_epochs} ep, ar=1 (noise ramp over "
                f"{args.stoch_warmup_epochs} ep)",
                end,
                _ar1_phase,
            )
        )
        if args.crps_ar2_epochs > 0:
            end += args.crps_ar2_epochs
            phases.append(
                (f"Phase 2: {args.crps_ar2_epochs} ep, ar=2", end, _ar2_phase)
            )
        print(
            f"[crps_finetune] epochs {start_epoch} -> {end}, stochasticity "
            f"{sorted(model.stochasticity)}, lr {args.crps_lr / 10} -> "
            f"{args.crps_lr} -> {args.crps_lr / 10} (warmup {warmup_epochs} "
            f"ep), distributed CRPS: {args.train_distr_crps}"
        )

        ckpt_path = args.load
        done = _completed_epochs(ckpt_path)
        for label, max_epochs, setup in phases:
            setup()
            if done >= max_epochs:
                print(f"[crps_finetune] {label} -- already done, skipping")
                continue
            print(f"[crps_finetune] {label}")
            new_ckpt = _run_phase(
                model, data_module, max_epochs, "ddp", ckpt_path=ckpt_path
            )
            ckpt_path = new_ckpt or ckpt_path
            done = max(done, max_epochs)
            # Later phases continue from this run's checkpoint: keep its
            # optimizer state and LR schedule
            model.restore_opt = True

    def _add_cosine_lr(model, max_epochs):
        """Attach a CosineAnnealingLR scheduler to the model."""
        _orig_configure = model.configure_optimizers

        def configure_optimizers_with_cosine(self_ref=model):
            opt = _orig_configure()
            if isinstance(opt, dict):
                optimizer = opt["optimizer"]
            else:
                optimizer = opt
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-5
            )
            return {"optimizer": optimizer, "lr_scheduler": sched}

        model.configure_optimizers = configure_optimizers_with_cosine

    def _add_finetune_sequential_lr(
        model,
        warmup_epochs,
        finetune_epochs,
        initial_lr,
        lr,
        eta_min,
    ):
        """Linear LR warmup then cosine decay over finetune_epochs."""
        _orig_configure = model.configure_optimizers

        def configure_optimizers_finetune(self_ref=model):
            opt = _orig_configure()
            if isinstance(opt, dict):
                optimizer = opt["optimizer"]
            else:
                optimizer = opt
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            linear_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=initial_lr / lr,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=finetune_epochs - warmup_epochs,
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[linear_warmup, cosine_annealing],
                milestones=[warmup_epochs],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        model.configure_optimizers = configure_optimizers_finetune

    if args.eval:
        strategy = (
            "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"
        )
        trainer = _make_trainer(args.epochs, strategy)
        if trainer.global_rank == 0:
            utils.init_training_logger_metrics(
                training_logger, val_steps=args.val_steps_to_log
            )
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=args.load,
            **_CKPT_LOAD_KWARGS,
        )

    elif args.scheduler == "pretrain":
        total_epochs = 100 + 200 + 25
        _add_cosine_lr(model, total_epochs)

        def _pretrain_p1():
            model.kl_beta = 0.0
            _set_ar_steps(1)

        def _pretrain_p2():
            model.kl_beta = 0.1

        def _pretrain_p3():
            _set_ar_steps(2)

        _run_phases(
            [
                (
                    "Phase 1/3: 100 epochs, ar=1, kl_beta=0",
                    100,
                    _pretrain_p1,
                    "ddp_find_unused_parameters_true",
                ),
                (
                    "Phase 2/3: 200 epochs, ar=1, kl_beta=0.1",
                    300,
                    _pretrain_p2,
                    "ddp",
                ),
                (
                    "Phase 3/3: 25 epochs, ar=2, kl_beta=0.1",
                    325,
                    _pretrain_p3,
                    "ddp",
                ),
            ]
        )

    elif args.scheduler == "finetune":
        prior_done = 325
        warmup_epochs = 5
        finetune_epochs = 165
        _add_finetune_sequential_lr(
            model,
            warmup_epochs=warmup_epochs,
            finetune_epochs=finetune_epochs,
            initial_lr=1e-5,
            lr=1e-4,
            eta_min=1e-5,
        )

        def _finetune_p1():
            model.kl_beta = 0.0
            model.crps_weight = 0.0
            _set_ar_steps(1)

        def _finetune_p2():
            model.kl_beta = 0.1
            model.crps_weight = 0.0
            _set_ar_steps(1)

        def _finetune_p3():
            model.kl_beta = 0.1
            model.crps_weight = 0.0
            _set_ar_steps(2)

        def _finetune_p4():
            model.kl_beta = 0.1
            model.crps_weight = 1e6
            _set_ar_steps(2)

        _run_phases(
            [
                (
                    "Phase 1/4: 5 ep warmup, ar=1, kl_beta=0, "
                    "strategy=ddp_find_unused_parameters_true",
                    prior_done + warmup_epochs,
                    _finetune_p1,
                    "ddp_find_unused_parameters_true",
                ),
                (
                    "Phase 2/4: 150 ep, ar=1, kl_beta=0.1, strategy=ddp",
                    prior_done + 155,
                    _finetune_p2,
                    "ddp",
                ),
                (
                    "Phase 3/4: 5 ep, ar=2, kl_beta=0.1, strategy=ddp",
                    prior_done + 160,
                    _finetune_p3,
                    "ddp",
                ),
                (
                    "Phase 4/4: 5 ep, ar=2, kl_beta=0.1, crps_weight=1e6, "
                    "strategy=ddp",
                    prior_done + finetune_epochs,
                    _finetune_p4,
                    "ddp",
                ),
            ]
        )

    elif args.scheduler == "probabilistic":
        total_epochs = 100 + 200 + 25 + 25
        _add_cosine_lr(model, total_epochs)

        def _prob_p1():
            model.crps_weight = 0.0
            model.kl_beta = 0.0
            _set_ar_steps(1)

        def _prob_p2():
            model.kl_beta = 0.1

        def _prob_p3():
            _set_ar_steps(2)

        def _prob_p4():
            model.crps_weight = 1e4

        _run_phases(
            [
                (
                    "Phase 1/4: 100 epochs, kl_beta=0",
                    100,
                    _prob_p1,
                    "ddp_find_unused_parameters_true",
                ),
                (
                    "Phase 2/4: 200 epochs, kl_beta=0.1",
                    300,
                    _prob_p2,
                    "ddp",
                ),
                (
                    "Phase 3/4: 25 epochs, ar=2, kl_beta=0.1",
                    325,
                    _prob_p3,
                    "ddp",
                ),
                (
                    "Phase 4/4: 25 epochs, ar=2, kl_beta=0.1, "
                    "crps_weight=1e4",
                    350,
                    _prob_p4,
                    "ddp",
                ),
            ]
        )

    elif args.scheduler == "deterministic":
        total_epochs = 150 + 25
        _add_cosine_lr(model, total_epochs)

        def _det_p1():
            _set_ar_steps(1)

        def _det_p2():
            _set_ar_steps(2)

        _run_phases(
            [
                ("Phase 1/2: 150 epochs, ar=1", 150, _det_p1, "ddp"),
                ("Phase 2/2: 25 epochs, ar=2", 175, _det_p2, "ddp"),
            ]
        )

    elif args.scheduler in ("finetune_deterministic", "finetune_ar1"):
        prior_done = 175
        _add_finetune_sequential_lr(
            model,
            warmup_epochs=5,
            finetune_epochs=60,
            initial_lr=1e-5,
            lr=1e-4,
            eta_min=1e-5,
        )

        def _ft_det_p1():
            _set_ar_steps(1)

        def _ft_det_p2():
            _set_ar_steps(2)

        if args.scheduler == "finetune_ar1":
            # Same length and LR schedule, but no ar=2 phase. Same DDP
            # strategy as the default single-phase training of FM.
            _run_phases(
                [
                    (
                        "Phase 1/1: 60 ep, ar=1",
                        prior_done + 60,
                        _ft_det_p1,
                        "ddp_find_unused_parameters_true",
                    )
                ]
            )
        else:
            _run_phases(
                [
                    (
                        "Phase 1/2: 50 ep, ar=1",
                        prior_done + 50,
                        _ft_det_p1,
                        "ddp",
                    ),
                    (
                        "Phase 2/2: 10 ep, ar=2",
                        prior_done + 60,
                        _ft_det_p2,
                        "ddp",
                    ),
                ]
            )

    elif args.scheduler == "crps_finetune":
        _run_crps_finetune()

    elif args.scheduler in FLOW_MAP_SCHEDULERS:
        assert args.model == "FlowMap", (
            f"--scheduler {args.scheduler} is for --model FlowMap"
        )
        objective, ar_steps, default_warmup = FLOW_MAP_SCHEDULERS[
            args.scheduler
        ]
        if objective is None:
            # The CRPS part: crps or map+crps (map+crps unless given)
            objective = (
                args.fmap_objective
                if args.fmap_objective in ("crps", "map+crps")
                else "map+crps"
            )
        if ar_steps is None:
            ar_steps = args.ar_steps_train if args.ar_steps_train > 1 else 2
        warmup_epochs = (
            default_warmup if args.warmup_epochs is None else args.warmup_epochs
        )

        # Where this part starts. --load is a checkpoint either of the
        # previous part (start a new part after its last epoch) or of this
        # part itself (resume/extend it, keeping the optimizer state).
        start_epoch, same_part = 0, False
        if args.load:
            ckpt = torch.load(
                args.load, map_location="cpu", mmap=True, weights_only=False
            )
            loaded_args = ckpt.get("hyper_parameters", {}).get("args")
            same_part = (
                getattr(loaded_args, "scheduler", None) == args.scheduler
            )
            done = _completed_epochs(args.load)
            if same_part:
                start_epoch = ckpt.get("fmap_stage_start_epoch", 0)
            else:
                start_epoch = done
            del ckpt
            print(
                f"[{args.scheduler}] --load {args.load}: "
                + (
                    f"resuming this part (started at epoch {start_epoch}, "
                    f"{done - start_epoch} epoch(s) done)"
                    if same_part
                    else f"starting a new part after epoch {done}"
                )
            )
            # Keep the optimizer and LR schedule only within a part
            model.restore_opt = same_part
            if same_part and not args.restore_opt:
                print(
                    f"[{args.scheduler}] restoring the optimizer state "
                    "(same part)"
                )
        max_epochs = start_epoch + args.epochs

        model.fmap_objective = objective
        model.fmap_stage_start_epoch = start_epoch
        # A new part restarts the EMA (new resolution or objective)
        model.fmap_reset_ema_on_train_start = not same_part
        _set_ar_steps(ar_steps)

        # Per-epoch LR: linear warmup, then cosine from --lr to --min_lr
        # over the --epochs of this part. The lambda is rebuilt from the
        # current --epochs, so extending a part re-stretches the cosine (the
        # epoch count is restored with the scheduler state).
        part_epochs = args.epochs
        lr_floor = min(args.min_lr / args.lr, 1.0)

        def _lr_factor(epoch):
            warm = min(1.0, (epoch + 1) / warmup_epochs) if warmup_epochs else 1
            progress = min(epoch / max(part_epochs, 1), 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return warm * (lr_floor + (1 - lr_floor) * cosine)

        _orig_configure = model.configure_optimizers

        def configure_optimizers_flow_map(self_ref=model):
            opt = _orig_configure()
            optimizer = opt["optimizer"] if isinstance(opt, dict) else opt
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
                param_group["initial_lr"] = args.lr
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }

        model.configure_optimizers = configure_optimizers_flow_map
        if same_part:
            print(
                f"[{args.scheduler}] NOTE: the cosine LR runs over the "
                f"current --epochs {part_epochs}; if this is larger than in "
                "the earlier run of the part, the remaining schedule is "
                "re-stretched"
            )

        # The learned loss weight is unused with the pure CRPS objective
        strategy = (
            "ddp_find_unused_parameters_true"
            if objective == "crps"
            else "ddp"
        )
        print(
            f"[{args.scheduler}] epochs {start_epoch} -> {max_epochs}, "
            f"objective {objective}, ar_steps_train {ar_steps}, "
            f"lr {args.lr} -> {args.min_lr}, warmup {warmup_epochs} epochs"
        )
        _run_phase(
            model, data_module, max_epochs, strategy, ckpt_path=args.load
        )

    else:
        # Default: single-phase training
        strategy = (
            "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"
        )
        _add_cosine_lr(model, args.epochs)
        trainer = _make_trainer(args.epochs, strategy)
        if trainer.global_rank == 0:
            utils.init_training_logger_metrics(
                training_logger, val_steps=args.val_steps_to_log
            )
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=args.load,
            **_CKPT_LOAD_KWARGS,
        )


if __name__ == "__main__":
    main()
