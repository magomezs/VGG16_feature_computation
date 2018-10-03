# VGG16_feature_computation
C++ class to get the output of a pre-trained VGG16 network

The class presents two public functions: 
- The class constructor. This has two inputs: model_file and trained_file.
    - model_file is a prototxt file defining the caffe layers of the VGG network. 
    - trained_file is the caffemodel file with the pre-trained weights for the VGG network. 
- The feature computation funtion. This has one input, a row image, and an output the feature array compute by the VGG net.


#ACKNOWLEDGMENTS
This work was supported by the Spanish Government through the CICYT projects (TRA2015-63708-R and TRA2016-78886-C3-1-R), and Ministerio de Educación, Cultura y Deporte para la Formación de Profesorado Universitario (FPU14/02143), and Comunidad de Madrid through SEGVAUTO-TRIES (S2013/MIT- 2713). We gratefully acknowledge the support of NVIDIA Corporation with the donation of the GPUs used for this research.
