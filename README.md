# CDFSL MiniImageNet to EuroSAT and CUB

## Pretraining notebooks

- [VGG](notebooks/model_pre_training_vgg.ipynb)
- [ViT](notebooks/model_pre_training_Vit.ipynb)


## Accuracy of models

### Pre-train on `miniImageNet` dataset


| Model             | Validation Accuracy |
|-------------------|---------------------|
| VGG19             | 82.84%              |
| VisionTransformer | 93.83%              |

### Fine tuned on EuroSAT_RGB dataset 

- Episodes : 20

| Model             | Average Test Accuracy |
|-------------------|-----------------------|
| VGG19             | 78.94%                |
| VisionTransformer | 81.34%                |

### Fine tuned on CUB dataset

- Episodes : 20

| Model             | Average Test Accuracy |
|-------------------|-----------------------|
| VGG19             | 77.067%               |
| VisionTransformer | 81.34%                |


### Fine tuned on CUB dataset

- Episodes : 50

| Model             | Average Test Accuracy |
|-------------------|-----------------------|
| VGG19             | 73.41%                |
| VisionTransformer | 93.34%                |

## Pretrained - models

RestNet152d - best_model_ModelResnet152dTimm_FixedTransforms.pth
VGG19 - best_model_VGG19_fixedTransformers.pth
VisionTransformer - best_model_VistionTransformerTimm_pc18_fixedTransformers.pth

You can download pretrained models from this location [https://drive.google.com/drive/folders/1oD5kTRVJPSjhVz2C6cJeI1EXc2tmHMWq](https://drive.google.com/drive/folders/1oD5kTRVJPSjhVz2C6cJeI1EXc2tmHMWq)