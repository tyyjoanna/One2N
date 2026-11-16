# Reproducibility and release scope

[← README](../README.md) · [Data and artifacts](data.md) · [Architecture](architecture.md)

This inference source release includes executable synthetic examples. It preserves the supplied model and numerical conventions while adding configuration, packaging, and streaming. Synthetic validation and full study reproduction have different scopes.

## Included materials

| Material | Included implementation |
| --- | --- |
| Source/version | Installable package, root scripts, dependency versions, MIT license |
| Run settings | Complete reference and synthetic YAML configurations |
| Installation | Python/CPU commands, Conda specification, GPU guidance, resource notes |
| Demonstration | Root `demo.py` generates random data and exercises both schedules |
| Expected products | Forecast NetCDF, resolved run settings, comparison summary |
| Test inputs | Generated fields and an original-model synthetic reference |
| Method | Module responsibilities, original-source mapping, numerical invariants |
| Data interface | Names, dimensions, units, grids, pressure levels, artifact roles |
| Automation | CPU workflow configured for Ubuntu and Windows |

The local dependency environment uses Python 3.12.7 and Windows CPU execution. Preparing the pinned CPU dependencies took approximately eight minutes including downloads. This records the dependency setup used for validation, not a hardware-independent benchmark. Local installation used a Python virtual environment; the alternative Conda recipe is syntax-checked rather than reported as an executed Conda environment creation.

All 26 automated tests passed against the installed wheel, without skipped tests; the full check took approximately 63 seconds on this host. Explicit initialization-time selection is covered. The documented root demo command completed in approximately seven seconds after initial imports on this host. Its `summary.json` recorded three residual updates, three output files per schedule, equal batch/streaming products, and the expected execution order. The executable prints elapsed time; timings depend on hardware, configuration, compilation caches, and network for installation.

## What random-data validation establishes

The demonstration exercises configuration, the configured small Swin-UNet, normalization, real conservative remapping, residual feedback, schedules, and NetCDF output. A controlled synthetic baseline replaces the research NeuralGCM checkpoint. Batch and streaming products are compared using configured tolerances. This checks architecture and execution.

Tests also cover original-model regression, strict parameter loading, file schemas, latent-state continuity, process transfer, output frequency, times, units, and output-only humidity clipping. Full model geometry can be checked without allocating a complete forecast tensor.

The frozen `tests/fixtures/original_model_reference.npz` is a compact regression fixture: random small-model state, synthetic input, and the corresponding original-model output. Neither the real forecast path nor `demo.py` reads it. It is optional for running inference but needed to execute the full test suite unchanged. Its provenance is documented in [`tests/fixtures/README.md`](../tests/fixtures/README.md); it is not a research checkpoint or weather dataset.

The real-data YAML has no built-in initialization date. The caller supplies one through `--start-time` or an explicit YAML setting. The demo YAML retains one synthetic epoch, from which fixtures and expected times are derived. Calendar strings in file naming conventions identify required input/output formats, rather than fixed forecast dates.

These checks do not establish atmospheric skill, real NeuralGCM agreement across schedules, or manuscript metrics. Random demonstration weights are not a trained research model.

## Materials external to this archive

| Material | Current availability |
| --- | --- |
| Weather initialization | Obtain from documented providers; prepare the required schema |
| NeuralGCM checkpoint | Obtain from the official project under its terms |
| Trained One2N checkpoint | Not redistributed; no public download location supplied |
| Matching statistics | Not redistributed; must accompany the trained weights |
| Training/manuscript evaluation scripts | Outside this inference release |
| Independent real-data reference outputs | Not included or validated here |
| Code archive DOI | Not assigned to this local archive |
| Hosted CI result | Available after upload and workflow execution |

## Publication use

Nature's [code and software guidance](https://media.nature.com/full/nature-cms/documents/GuidelinesCodePublication.pdf) requests versioned source, installation details, an example with expected behavior and timing, and licensing. Publication also requires access to supporting materials and maintenance of the paper-associated version. Editors and reviewers determine the final acceptable release.

This archive supplies the code-side materials above. Full paper reproduction additionally requires accessible matching artifacts, a real-data example with expected outputs and measured runtime, and relevant training/evaluation code when central to manuscript claims. After repository and archival identifiers exist, the manuscript's Code Availability statement should identify that concrete version and explain artifact access/restrictions. No repository URL, code DOI, artifact route, or successful real forecast is invented here.
