# Dataset Directory

- Save dataset as `./${PROJECT_NAME}`  
- Annotation json format: COCO (Refer to [coco-format](http://cocodataset.org/#format-data))

> `${PROJECT_NAME}` is from [`../configs/initial_settings.json`](../configs/initial_settings.json)

```
# ./
${PROJECT_NAME}
│
│
├── labels
│   ├── train.json
│   └── val.json  --> Not necessary for training.
│
│
└── imgs
    │
    ├── train
    │      │
    │      ├── 0001.png
    │      ├── 0002.png
    │      ├── 0003.png
    │      ├── . . .
    │
    │
    └──── val  --> Not necessary for training.
           │
           ├── 0001.png
           ├── 0002.png
           ├── 0003.png
           ├── . . .
```
