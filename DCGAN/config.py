########################
#### CONFIG #####

config1={
    'FILTER1':16,#24
    'FILTER2':32,#24
    'FILTER3':64,#32
    'FILTER4':128,#64
    'FILTER5':256,#128
    'FILTER6':512,
    'FILTER7':1024,
    'LATENTDIM': 100, #arbitrary
    'BATCHSIZE': 4,
    'NUMEPOCHS':500,
    #'NUMCLASSES': 6, 
    'NUMCLASSES': 5,
    'L2PENALTY':0.00,
    'LEAKYALPHA':0.03,
    'LEARNRATE':0.0001, 
    #'TRAINSIZE':6396,
    #'VALSIZE':1024,
    'TRAINSIZE':8,
    'VALSIZE':60,
    'H':384,
    'W':256,
    'C':1,
    'H0':512,
    'W0':512,
    'C0':1,
    
    #I changed model architecture in the following run
    'modelName':'Discriminator_3D16_3D32_3D64_3D128_3D256_3D512_1C1024'
}

config2 = {
    'STEPPEREPOCHS':int(config1['TRAINSIZE']/config1['BATCHSIZE']), 
    'VALSTEPS':int(config1['VALSIZE']/config1['BATCHSIZE']),
}

config3 = {
    'DECAYRATE':1/(config2['STEPPEREPOCHS']*config1['NUMEPOCHS']),
    #TODO: FIX WEIGHTS
    #'CLASSWEIGHTS':{0: 1.0, 1: 173, 2: 935, 3: 165, 4: 842, 5: 2414} #calculated globally (total_for_all_categories/total_for_category)
    'CLASSWEIGHTS':[0.005, 0.025, 0.6, 3.2, 0.6, 3.0] #logarithm of above numbers
}
