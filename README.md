# ood_detection_methods
Master's Thesis source code. OOD detection methods on urine image patches. (dataset is not included)

See [Thesis report](https://github.com/erdemunal35/ood_detection_methods/blob/main/THESIS.pdf)

## Abstract
Deep Learning models are widely used in microscopic imaging. Unaware models can be easily fooled by unseen anomalous inputs and yield unexpected results. This thesis analyzes unsupervised out-of-distribution (OOD) detection methods to solve this problem. Baseline generative and classifier model-based OOD detection methods are elaborated and applied to a specific microscopic dataset. Results showed that Mahalanobis distancebased detector for pre-trained classifier provides an effective and applicable solution in terms of performance, eficiency and robustness. Generative models are concluded to be unfeasible to obtain distinguishable low dimensional feature representations. Reconstruction
error-based anomaly detection method with generative models is shown to be ineffective regardless of used model and error metric. Statistical two-sample tests have a considerable
potential to detect the shift between two image representations which can be obtained by various dimensionality reduction methods.

## Best Anomaly Detection Method for Microscopic Data (in terms of both accuracy and efficiency)
Taking the Mahalanobis distance based detector on penultime layer outputs of a trained (only with inliers) supervised classiifer obtained the best anomaly score.
See [the source code](https://github.com/erdemunal35/ood_detection_methods/blob/main/discriminative_mahalanobis.ipynb) for plots
