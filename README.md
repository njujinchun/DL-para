## [Deep learning-based parameterization of heterogeneous geological parameter fields](https://doi.org/10.1016/j.advwatres.2024.104833)
Li Feng, [Shaoxing Mo](https://scholar.google.com/citations?user=b5m_q4sAAAAJ&hl=en&oi=ao), [Alexander Y. Sun](https://scholar.google.com/citations?hl=en&user=NfjnpFYAAAAJ), ..., Jichun Wu, 
[Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra)

This is a PyTorch implementation of deep learning (DL)-based parameterization model for effectively representing the heterogeneous geological parameter fields using low-dimensional latent vectors. The DL-based parameterization method generates a low-dimensional representation of the non-Gaussian permeability fields with multi- and intra-facies heterogeneity in a geological carbon storage (GCS) problem. 

Once trained, the DL model
88 can produce a non-Gaussian permeability field given an arbitrary low-dimensional latent
89 vector as input.
Results show that the DL model is able to represent the highly complex
and high-dimensional permeability fields using low-dimensional latent vectors.


pCNN can provide reliable predictions and predictive uncertainties of $CO_2$ saturation fields and then 
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
python3 train_DL.py
```

# Citation
See [Feng et al. (2024)](https://doi.org/10.1016/j.advwatres.2024.104833) for more information. If you find this repo useful for your research, please consider to cite:
```
@article{FENG2024104833,
	author = {Li Feng and Shaoxing Mo and Alexander Y. Sun and Dexi Wang and Zhengmao Yang and Yuhan Chen and Haiou Wang and Jichun Wu and Xiaoqing Shi},
	title = {Deep learning-based geological parameterization for history matching {CO}$_2$ plume migration in complex aquifers},
	journal = {Advances in Water Resources},
	pages = {104833},
	year = {2024},
	issn = {0309-1708},
	doi = {https://doi.org/10.1016/j.advwatres.2024.104833}
}
```
or:
```
Feng, L., Mo, S., Sun, A. Y., Wang, D., Yang, Z., Chen, Y., Wang, H., Wu, J., & Shi, X. (2024). Deep learning-based geological parameterization for history matching CO$_2$ plume migration in complex aquifers. Advances in Water Resources, 104833. https://doi.org/10.1016/j.advwatres.2024.104833
```
Related article: [Mo, S., Zabaras, N., Shi, X., Wu, J., 2020. Integration of adversarial autoencoders with residual dense convolutional networks for estimation of non-Gaussian hydraulic conductivities. Water Resources Research 56, e2019WR026082. doi:https://doi.org/10.1029/2019WR026082.]

## Questions
Contact Li Feng (fengli@smail.nju.edu.cn) or Shaoxing Mo (smo@nju.edu.cn) with questions or comments.
