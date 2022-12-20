# Thesis

## Topic
Detection of periprosthetic joint infection and abrasion in histology of patients with total hip and knee arthroplasty.

## Approach

### Infection detection

- **Approach 1:** Neutrophil counting
    1. Train a neutrophil detector on H&E slides using the [Lizard dataset](https://arxiv.org/abs/2108.11195)
    2. Run inference on WSI tiles, get the histogram for number of neutrophils for each tile
    3. Analyze histograms and engineer a histogram feature that successfully tells apart infection from inflammation

- **Approach 2:** Multiple instance learning
    1. Get CNN embeddings for each tile
    2. Apply [attention-based deep multiple instance learning](https://arxiv.org/pdf/1802.04712.pdf) for classification

### Abrasion detection
1. Get CNN embeddings for each tile
2. Apply [attention-based deep multiple instance learning](https://arxiv.org/pdf/1802.04712.pdf) for classification

## Built upon

1. [This repository](https://github.com/adamshephard/TIAger) from TIGER challenge by Shephard et al.
2. [PyHIST](https://github.com/manuel-munoz-aguirre/PyHIST) by Manuel Muñoz Aguirre
3. [YOLOv5](https://github.com/ultralytics/yolov5/) by Ultralytics 