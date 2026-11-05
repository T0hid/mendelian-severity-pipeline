# Software submission checklist answers

Nature Medicine code-review materials, version 1.3.0. This file supplies answers and evidence locations; it is not a signed journal form or editorial approval.

| Item | Answer and evidence |
|---|---|
| Software/version | Mendelian severity pipeline 1.3.0; `CITATION.cff` |
| Source | https://github.com/T0hid/mendelian-severity-pipeline |
| Language | Python 3.11; Bash capsule launcher |
| Systems tested | Ubuntu 24.04 through GitHub; Windows local checks. Exact environments accompany runs |
| Hardware | CPU; no GPU needed for review. Resource estimates in `INSTALL.md` |
| Dependencies | `requirements.txt`; `src/requirements-full.txt` |
| Installation | `docs/INSTALL.md` |
| Example data | Two invented HPO profiles, frozen model, public inputs and reference outputs bundled |
| Demonstration | `python run.py`; open `results/index.html` |
| Expected outputs | `docs/INPUT_OUTPUT.md` |
| Runtime | Recorded in `results/execution.json`; estimates distinguished in installation guide |
| Reproduction | `docs/REPRODUCE.md`; three workflows plus saved-data plotting |
| Parameters | Original manifest/configuration and random seeds retained |
| Validation | Regression tests, source fingerprints, saved-result replay and checksums |
| Figures/source data | `docs/OUTPUT_MAP.csv`; workbook source index sheets |
| Restrictions | Individual prenatal data excluded; authorised cohort execution supported |
| Licence | Rights holder has not selected project-wide reuse terms; `docs/LICENSING.md` |
| DOI | Not assigned; deposit steps in `docs/CODE_AVAILABILITY.md` |
| Colleague testing | Not claimed; automated checks recorded separately |
| Code Ocean | Run-file instructions provided; hosted execution not verified |

Policy references checked on 5 November 2026:

- https://www.nature.com/nm/editorial-policies
- https://www.nature.com/nm/editorial-policies/reporting-standards
- https://media.nature.com/full/nature-cms/documents/GuidelinesCodePublication.pdf

The package provides source, installation/use instructions, example inputs, expected outputs, parameters, dependencies, plotting code and availability information. The rights holder must complete licence/deposit metadata and transfer these answers to any official form requested by the editor.
