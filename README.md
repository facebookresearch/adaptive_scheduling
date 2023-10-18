#  Adaptive Learning Rate Scheduling by Refinement
Scripts for replicating the visualization plots in the paper ["When, Why and How Much? Adaptive Learning Rate Scheduling by Refinement".](https://arxiv.org/abs/2310.07831)

Please cite as:
```
@misc{defazio2023when,
title={When, Why and How Much? Adaptive Learning Rate Scheduling by Refinement},
author={Aaron Defazio and Ashok Cutkosky and Harsh Mehta and Konstantin Mishchenko},
year={2023},
eprint={2310.07831},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
```

# Refinement method
The `find_closed_form_schedule` function in the `find_schedule.py` implements the gradient norm to weight mapping.

The filtering with a median filter can be done using scipy's `scipy.ndimage.median_filter` implementation. Some care is needed to handle the padding:
```python
    filter_width = 2*(int(sigma2*nvals)//2) + 1
    pad = 2*filter_width
    gnorms_filtered2 = median_filter(np.pad(gnorms, (0, pad), mode='reflect'), size=filter_width, mode='nearest')[:-pad]
```


## License
Adaptive Scheduling is CC-BY-NC licensed, as found in the LICENSE file.