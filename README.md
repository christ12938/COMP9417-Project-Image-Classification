# comp9417
modules included in this project:
1. image_sorter.py - takes the provided .png training images, and combine with their class labels in .npy file, to produce a directory containing the images where each class is in their own subdirectory named accordingly
2. nn_tensorflow.py - entry point of program for tensorflow neural networks pipeline, train on provided images using specified model, and run prediction on a test directory. see -h or --help for detailed list of arguments
3. nn_tensorflow_models.py - factory methods for creating models of various structure
4. nn_tensorflow_train.py - methods for different ways a model may be trained and fitted
5. nn_tensorflow_dataset.py - dataclass used in the other nn_tensorflow* modules
6. augmentation.py - program to split and augment X train, y train into augmented X_train_split, y_train_split, X_test_split, y_test_split for training and validation
7. nn_augmentation_model.py - seperate program to train the appened augmented data in 6. with class weights
