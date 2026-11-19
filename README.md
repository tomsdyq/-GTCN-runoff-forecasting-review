# G-TCN for multi-day runoff forecasting

Source release for *A variable-as-node Graph-Temporal Convolutional Network for multi-day runoff forecasting*, by Tao Zhou, Zhongyi Li and Weidong Shi, Environmental Modelling & Software, article 107242.

## Installation

Use Python 3.11 in a fresh environment. From the extracted directory run `python -m pip install -r requirements.txt`. For GPU training, install the matching PyTorch CUDA build for your hardware. Use `--device cpu` to run on CPU.

## Data

Obtain the CAMELS-US basin time series v1.2 from NCAR (https://doi.org/10.5065/D6MW2F4D). Extract the Daymet basin-mean forcing and USGS streamflow. In the examples, replace CAMELS_ROOT with the directory that contains both `basin_mean_forcing/daymet/` and `usgs_streamflow/`. Preserve the subdirectories and filenames supplied by the provider. The two text files in `data/` list the 673 study gauges and the 241-gauge subset. Keep leading zeros in gauge IDs.

Daily precipitation, solar radiation, maximum temperature, minimum temperature and vapour pressure provide the meteorological inputs. Each sample has 15 historical days and seven lead days. Observed lead-period meteorology represents perfect forcing. The runoff channel contains observed history followed by seven zeros. Only the last seven outputs are scored.

## Pretrain, fine-tune and test

Run these commands from the directory containing `run.py`. Each output directory must be new. The first command trains a regional G-TCN. The second initializes one basin from that regional checkpoint. Repeat the last two commands for each target basin, changing the gauge ID and output directories.

```sh
python run.py pretrain --camels CAMELS_ROOT --basins data/673basins_list.txt --out runs/regional
python run.py finetune --camels CAMELS_ROOT --basin 01013500 --checkpoint runs/regional/best.pt --out runs/ft_01013500
python run.py test --camels CAMELS_ROOT --basin 01013500 --checkpoint runs/ft_01013500/best.pt --out runs/test_01013500
```

For a regional checkpoint, test using the same basin list used for regional training. For a fine-tuned checkpoint, test with the same single gauge used for fine-tuning. The runner rebuilds the corresponding normalization statistics from the training period. Testing uses all test windows, including the last partial batch.

| Split | First day | Last day |
|---|---|---|
| Training | 1980-10-01 | 1995-09-30 |
| Validation | 1995-10-01 | 2000-09-30 |
| Test | 2000-10-01 | 2014-09-30 |

The full model uses 64 hidden channels, four temporal convolution blocks, Adam (0.001), batches of 512 and a 200-epoch schedule. Regional training uses the basin-weighted NSE loss. Fine-tuning uses MSE and basin-specific normalization. The warm-up lasts 50 epochs. Set `--epochs 1` for a short functional check. The default seed is 1. Results depend on initialization and the numerical environment.

Training writes `training.csv` and the selected state dictionary `best.pt`. The archived training code selects the largest batch-averaged validation score. The original score uses prediction variance in the denominator. This release preserves that checkpoint selection behaviour. The test CSV calculates NSE with observation variance and reports each gauge and lead separately. RMSE is in mm/day. Bias and ATPE2 are fractions (multiply by 100 for percentages).

## Revision experiments

`gtcn.py` also contains the component ablations. Use `python run.py --help` to list variants. For example, run the no-decay variant on the 241-basin subset for 120 epochs using the original 200-epoch scheduler:

```sh
python run.py pretrain --camels CAMELS_ROOT --basins data/241basins_list.txt --variant abl1_no_decay --epochs 120 --schedule-epochs 200 --seed 1 --out runs/no_decay
```

Pass the same variant when loading its checkpoint. The package includes the G-TCN training and evaluation workflow and ablation models. Obtain CAMELS separately. Historical weights and predictions, third-party benchmark pipelines and manuscript figure inputs are outside this source distribution.

The data handling and training procedures follow the study workflow originally developed using the RR-Former framework (Yin et al., 2022, https://doi.org/10.1016/j.jhydrol.2022.127781). This release implements the CAMELS reader and training interface specifically for G-TCN.

## Licence

The project uses the MIT licence. The temporal convolution network from CMU Locus Lab retains its original licence in `TCNmaster/LICENSE`.

