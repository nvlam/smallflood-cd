# BIT upstream research-use notice

Source: https://github.com/justchenhao/BIT_CD
Commit: adcd7aea6f234586ffffdd4e9959404f96271711
Authors: Hao Chen, Zipeng Qi, Zhenwei Shi.
Paper: Remote Sensing Image Change Detection with Transformers,
IEEE TGRS, DOI 10.1109/TGRS.2021.3095166.

The upstream README's License section states:
"Code is released for non-commercial and research purposes only. For commercial
purposes, please contact the authors."
Retain this restriction and attribution. This is not an MIT/Apache license grant.

Vendored files: models/networks.py, models/help_funcs.py, models/resnet.py.
Compatibility changes only: package-relative imports; torch.hub loader import;
string equality in place of identity; no implicit pretrained download (False).
The bit_sar_v2 wrapper declares input/output/initialization and unused-module adaptations.
Original files are separately retained under tests/reference/bit_upstream for parity tests.
No official pretrained checkpoint is included or fetched.
