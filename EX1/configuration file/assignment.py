_base_=[
    '/pytorchlearning/pytorch/mmpretrain/configs/_base_/models/resnet50.py',
    '/pytorchlearning/pytorch/mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
    '/pytorchlearning/pytorch/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    '/pytorchlearning/pytorch/mmpretrain/configs/_base_/default_runtime.py',
]

model=dict(head=dict(num_classes=5,topk=(1,)))
load_from='/pytorchlearning/pytorch/classdata/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

data_preprocessor=dict(
    mean=[123.675,116.28,103.53],
    std=[58.395,57.12,57.375],
    to_rgb=True,
    num_classes=5,
)

dataset_type='ImageNet'
data_root ='D:/pytorchlearning/pytorch/classdata/flower/'
classes=[c.strip() for c in open(f'{data_root}/classes.txt')]

train_dataloader=dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=f'{data_root}/train.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop',scale=224),
            dict(type='RandomFlip',prob=0.5,direction='horizontal'),
            dict(type='PackInputs')
        ]
    )
)

val_dataloader=dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=data_root,
        ann_file=f'{data_root}/val.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge',scale=256,edge='short'),
            dict(type='CenterCrop',crop_size=224),
            dict(type='PackInputs')
        ]
    )
)

val_cfg=dict()
val_evaluator=dict(type='Accuracy',topk=(1,))

#optim_wrapper=dict(
#   optimizer=dict(type='SGD',lr=0.005,momentum=0.9,weight_decay=1e-4)
#)
optim_wrapper=dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    ),
    clip_grad=dict(max_norm=1.0)
)

auto_scale_lr=dict(base_batch_size=256)

train_cfg=dict(
    by_epoch=True,
    max_epochs=20,
    val_interval=1,
)