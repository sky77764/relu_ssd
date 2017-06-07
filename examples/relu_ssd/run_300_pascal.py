import os
import relu_ssd_module as module

def LinearStep(stepvalue, step):
    length = len(stepvalue)

    values = []
    for i in range(1, length):
        gap = (stepvalue[i] - stepvalue[i-1]) / step
        l = range(stepvalue[i-1], stepvalue[i], gap)
        values = values + l

    return values

if __name__ == "__main__":
    lr_step = 8
    gamma = pow(0.1, 1.0/lr_step)

    stepvalue = [40000, 80000, 100000]

    stepvalue = LinearStep(stepvalue, lr_step)
    
    param = {
        'net_name': "RUN_3WAY",
        'input_dim': 300,
        'pretrain_model': "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel",
        'data_set': "VOC0712",
        'gpus': "0",
        'batch_size': 16,
        'accum_batch_size': 32,
        'use_batchnorm': False,
        'use_conv3_3': False,
        'use_unified_prediction': True,
        'num_test_image': 4952,
        'test_batch_size': 8,
        'residual_feature_depth': 256,
        'use_res_branch2' : True,
        'use_res_deconv' : True,


        'solver_param' : {
            # Train parameters            
            'weight_decay': 0.0005,
            'lr_policy': "multistep",
            'stepvalue': stepvalue,
            'gamma': gamma,
            'momentum': 0.9,            
            'max_iter': 120000,
            'snapshot': 10000,
            'display': 10,
            'average_loss': 10,
            'type': "SGD",            
            
            # Test parameters
            'test_interval': 5000,
            },
        'det_out_param' : {
            'nms_param': {'nms_threshold': 0.45, 'top_k': 400},            
            'keep_top_k': 200,
            'confidence_threshold': 0.01,            
            },  
    }

    module.Train(param, os.path.abspath(__file__))


