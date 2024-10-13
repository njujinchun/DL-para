## [Deep learning-based parameterization of heterogeneous geological parameter fields](https://doi.org/10.1016/j.advwatres.2024.104833)
Li Feng, [Shaoxing Mo](https://scholar.google.com/citations?user=b5m_q4sAAAAJ&hl=en&oi=ao), [Alexander Y. Sun](https://scholar.google.com/citations?hl=en&user=NfjnpFYAAAAJ), ..., Jichun Wu, 
[Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra)

This is a PyTorch implementation of probabilistic Convolutional Neural Network (pCNN) for uncertainty quantification of $CO_2$ plume migration in highly channelized permeability fields of aquifers for geological carbon storage. pCNN can provide reliable predictions and predictive uncertainties of $CO_2$ saturation fields and then 
be used for repeated model runs in uncertainty quantification (UQ) of $CO_2$ plume migration in non-Gaussian permeability fields. It can also be applied to other multiphase flow problems 
concerning complex image-to-image regressions. 

## Dependencies
* python 3
* PyTorch
* h5py
* matplotlib
* scipy

### pCNN network architecture
![](https://github.com/njujinchun/pCNN4GCS/blob/main/images/pCNN_arch.jpg)

### Reference $CO_2$ saturation fields (b-d), pCNN's predictions (e-g), the differences between references and predictions (h-j) and predictive uncertainties (k-m)
![](https://github.com/njujinchun/pCNN4GCS/blob/main/images/pCNN_predictions.JPG)

# Datasets
The datasets used in pCNN have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1mi9Cmgnufi3kSMCeedP7G_K-4aEcd3_A?usp=drive_link) and can be downloaded using this link.

# Network Training
```
python3 train_svgd.py
```

# Citation
See [Feng et al. (2024)](https://doi.org/10.1016/j.advwatres.2023.104607) for more information. If you find this repo useful for your research, please consider to cite:
```
@article{feng2024104607,
  title={Uncertainty quantification of CO$_2$ plume migration in highly channelized aquifers using probabilistic convolutional neural networks},
  author={Feng, Li and Mo, Shaoxing and Sun, Alexander Y and Wu, Jichun and Shi, Xiaoqing},
  journal={Advances in Water Resources},
  volume = {183},
  pages={104607},
  year={2024},
  publisher={Elsevier},
  doi = {https://doi.org/10.1016/j.advwatres.2023.104607}
}
```
or:
```
Feng, L., Mo, S., Sun, A. Y., Wu, J., & Shi, X. (2024). Uncertainty quantification of $CO_2$ plume migration in highly channelized aquifers using probabilistic 
convolutional neural networks. Advances in Water Resources, 104607. https://doi.org/10.1016/j.advwatres.2023.104607
```
Related article: [Zhu, Y., & Zabaras, N. (2018). Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification. J. Comput. Phys., 366, 415-447.](https://www.sciencedirect.com/science/article/pii/S0021999118302341)

## Questions
Contact Li Feng (dz1929010@smail.nju.edu.cn) or Shaoxing Mo (smo@nju.edu.cn) with questions or comments.
