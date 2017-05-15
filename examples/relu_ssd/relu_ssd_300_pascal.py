import os
import relu_ssd_module as module


if __name__ == "__main__":
    
    param = {
        'input_dim': 300,
        'pretrain_model': "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel",
        'data_set': "VOC0712Plus",
        'gpus': "0",
        'batch_size': 16,
        'accum_batch_size': 32,
        'use_batchnorm': False,
        'use_conv3_3': True,
        'use_unified_prediction': True,
        'num_test_image': 10991,
        'test_batch_size': 8,
        'residual_feature_depth': 256,

        'solver_param' : {
            # Train parameters            
            'weight_decay': 0.0005,
            'lr_policy': "multistep",
            'stepvalue': [80000, 130000, 160000],
            'gamma': 0.1,
            'momentum': 0.9,            
            'max_iter': 160000,
            'snapshot': 10000,
            'display': 10,
            'average_loss': 10,
            'type': "SGD",            
            
            # Test parameters
            'test_interval': 160000,
            },
        'det_out_param' : {
            'nms_param': {'nms_threshold': 0.45, 'top_k': 400},            
            'keep_top_k': 200,
            'confidence_threshold': 0.01,            
            },  
    }

    module.Train(param, os.path.abspath(__file__))


