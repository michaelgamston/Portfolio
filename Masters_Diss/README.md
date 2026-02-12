## Masters Disseration

In this project for my masters disseration I build and train a variational autoencoder on gin recipes with the aim of producing new samples that can be used by distilleries. 

This project involes: 

Data exploration and cleaning. Most of this exploration can be found within: 
    - data_processing
        - data_correlations.ipynb 
        - data_exploration.ipynb 
        - data_preprocess.ipynb 

Experiments using different types of traditional ML on the orignal data set to see which model is best at predicting the flavours present within each recipe, the best model is then used to assess the quality of generated samples. This file can be found in: 
    - ML 
        - original_labels_classification.ipynb 

Clustering using the botanicals and compounds to find better labels, along with some visualisation and inspection of the created clusters. These files can be found in: 
    -ML 
        - feature_clustering 
            - *

Clustering using vectorised versions of the written descriptions to find better labels, along with further inspection and visualisation. The vectorisation files can be found under: 
    - data_processing
        - vectorise_descriptions.ipynb
The clustering can be found under: 
    - ML
        - embedded_vector_clustering 
            - *

Experimenting with three different types of variational autoencoder and then assessing and visualising the generated samples. Along with an experiment to use bayesian optimisation to optimise the structure of the variational autoencoder. The bayesian tests can be found under: 
    - ML 
        - bayesian 
            - *
The variational autoencoder code and visualisations can be found under: 
    - ML 
        - auto_encoder
            - *