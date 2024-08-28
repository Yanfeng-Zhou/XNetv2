import numpy as np
import torchio as tio

def dataset_cfg(dataet_name):

    config = {
        'CREMI':
            {
                'PATH_DATASET': '.../XNetv2/dataset/CREMI',
                'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
                'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.503902],
                'STD': [0.110739],
                'INPUT_SIZE': (128, 128),
                'PALETTE': list(np.array([
                    [255, 255, 255],
                    [0, 0, 0],
                ]).flatten())

            },
        'GlaS':
            {
                'PATH_DATASET': '.../XNetv2/dataset/GlaS',
                'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
                'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.787803, 0.512017, 0.784938],
                'STD': [0.428206, 0.507778, 0.426366],
                'INPUT_SIZE': (128, 128),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'ISIC-2017':
            {
                'PATH_DATASET': '.../XNetv2/dataset/ISIC-2017',
                'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
                'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.699002, 0.556046, 0.512134],
                'STD': [0.365650, 0.317347, 0.339400],
                'INPUT_SIZE': (128, 128),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'P-CT':
            {
                'PATH_DATASET': '.../XNetv2/dataset/P-CT',
                'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
                'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (96, 96, 96),
                'PATCH_OVERLAP': (80, 80, 80),
                'NUM_SAMPLE_TRAIN': 4,
                'NUM_SAMPLE_VAL': 8,
                'QUEUE_LENGTH': 48
            },
        'LiTS':
            {
                'PATH_DATASET': '.../XNetv2/dataset/LiTS',
                'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
                'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 3,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (112, 112, 32),
                'PATCH_OVERLAP': (56, 56, 16),
                'NUM_SAMPLE_TRAIN': 8,
                'NUM_SAMPLE_VAL': 12,
                'QUEUE_LENGTH': 48
            },
    }

    return config[dataet_name]
