## Usage

### Quick Start

Run a single example (approximately 20 minutes):
```bash
python training_torch.py --config-name=ant eval_population=False
```
### Reproducing Plots

To get phenotypic diversity plot (figure 4 in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.08.07.606541v1.abstract)):

```bash
python training_torch.py --config-name ant
```

To sweep through different compressions and environments run:

```bash
qsub UGE_job.sh
```

You should get results like [this](https://wandb.ai/self-supervisor/brax-cshl/reports/Genomic-Bottleneck-v2-Nov-1st--Vmlldzo1ODQyMzA3?accessToken=wh7ltbcurtd2xd8nekl1udteia84p7xjjw8ytq5vj1t6bj8hwqjmth8ux1fpwwbh)

Then edit the code in plot.py for your path so that is goes through the generated CSVs and plots the scatter plots (figure 3 in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.08.07.606541v1.abstract)).
