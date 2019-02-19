########################
#### CONFIG #####
global config1, config2, config3

config1={
    'FILTER1':16,#24
    'FILTER2':32,#24
    'FILTER3':64,#32
    'FILTER4':128,#64
    'FILTER5':256,#128
    'FILTER6':512,
    'FILTER7':1024,
    'BATCHSIZE':8,
    'NUMEPOCHS':50,
    'NUMCLASSES':5,
    'L2PENALTY':0.0001,
    'LEARNRATE':0.0001, #0.0001
    'TRAINSIZE':6396,
    'VALSIZE':1024,
    'H':384,
    'W':256,
    'C':1,
    'H0':512,
    'W0':512,
    'C0':1,
    #using default model with loss categorical-loss function, no data stratifying
    #modelName:"1x256x384_Base_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

    #training with weighted loss function
    #modelName:"1x256x384_WeightedLoss_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

    #training by augmenting data
    #modelName:"1x256x384_Augmented_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

    #same model as before but multiclass loss function
    #modelName:"1x256x384_MultiClass_3D16_3D32_3D64_3D128_3D256_3D512_1C1024"

    #same model as before but multiclass loss function
    'modelName':"1x256x384_MultiClassWeighetd_3D16_3D32_3D64_3D128_3D256_3D512_1C1024",
}

config2 = {
    'STEPPEREPOCHS':int(config1['TRAINSIZE']/config1['BATCHSIZE']), 
    'VALSTEPS':int(config1['VALSIZE']/config1['BATCHSIZE']),
}

config3 = {
    'DECAYRATE':1/(config2['STEPPEREPOCHS']*config1['NUMEPOCHS']),
    #CLASSWEIGHTS:{0: 1.0, 1: 2148, 2: 128, 3: 2864, 4: 525} #calculated globally (total_for_all_categories/total_for_category)
    'CLASSWEIGHTS':[ 1., 7.67229246,  4.85203026,  7.95997453,  6.26339826], #logarithm of above numbers
}

