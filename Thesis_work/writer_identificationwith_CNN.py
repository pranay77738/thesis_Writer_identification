import os
import glob
import argparse
from time import time
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Case of fragment manually generated
# train_data_dir = "/home/mario/Scrivania/keras_transferlearning/data_avila/train"
# validation_data_dir = "/home/mario/Scrivania/keras_transferlearning/data_avila/test"
from keras_applications.nasnet import NASNetLarge
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# IMPORTANT: example of transfer learning with last layer with new classifier
# https://gogul09.github.io/software/flower-recognition-deep-learning

recoveryTraining = False

enableTransferlearning = True
enableFinetuning = True

typeOfGroundTruth = "_Auto"  # Generated with row detector automatically
# typeOfGroundTruth = "_Manual" # Generated with manually generated XML

# test_data_dir = "/home/mario/Scrivania/TensorFlow_ObjectDetection/Avila/ObjectLabeled/Fragment_Manual/test"

base_dir = "/home/mario/Scrivania/TensorFlow_ObjectDetection/Avila/ObjectLabeled/"

modelToUse = "InceptionResNetV2"  # Alternatives are "ResNet50", "InceptionResNetV2"

train_data_dir = base_dir + "/Train"
validation_data_dir = base_dir + "/Val"
test_data_dir = base_dir + "/Test"

# From args
batch_size = 16
nb_epochs = 200
patience = 20
fc_size = 1024

# Automatically evaluated
nb_train_samples = 1924
nb_validation_samples = 238
img_width, img_height = 256, 256

frozenLayers = 5

VGG19_MODEL = "VGG19"
ResNet50_MODEL = "ResNet50"
InceptionResNetV2_MODEL = "InceptionResNetV2"
InceptionV3_MODEL = "InceptionV3"
NASNetLarge_MODEL = "NASNetLarge"


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def add_new_last_layer(base_model, nb_classes, fc_size):
    """Add last layer to the convnet

    Args:
      base_model: keras model excluding top
      nb_classes: # of classes

    Returns:
      new keras model with last layer
    """

    """
    # ORIGINAL VERSION
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    """

    # REVISED VERSION
    x = base_model.output
    # if modelToUse == "InceptionResNetV2_MODEL":
    #    x = Flatten()(x)
    # else:
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(fc_size, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation="softmax")(x)

    model = Model(input=base_model.input, output=predictions)

    return model


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers in base_model and compile the model"""
    # for layer in model.layers:
    #    print(layer)

    for layer in base_model.layers:
        layer.trainable = False
    # ORIGINAL VERSION
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # PROPOSED VERSION
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
                  metrics=["accuracy"])
    # model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(), metrics=["accuracy"])


def setup_to_finetune(model, layerNumToUnfreeze):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:layerNumToUnfreeze]:
        layer.trainable = False
    for layer in model.layers[layerNumToUnfreeze:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(), metrics=["accuracy"])


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def evalAccuracy(confusionMatrix):
    accuracy = 0
    accuracyByRow = []
    tot = 0
    for i in range(len(confusionMatrix)):
        totRow = 0
        for j in range(len(confusionMatrix)):
            tot = tot + confusionMatrix[i][j]
            totRow = totRow + confusionMatrix[i][j]
        accuracy = accuracy + confusionMatrix[i][i]
        accuracyByRow.append((confusionMatrix[i][i] * 100.0) / totRow)

    accuracy = (100.0 * accuracy) / tot

    return accuracy, accuracyByRow


def train(args):
    # Initiate the train and test generators with data Augumentation
    # PARAMETERS FROM COMMAND LINE
    base_dir = args.base_dir
    nb_epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    patience = int(args.patience)
    fc_size = int(args.fc_size)
    modelToUse = args.base_model
    enableTransferlearning = args.enable_transferlearning == "y"
    enableFinetuning = args.enable_finetuning == "y"
    frozenLayers = int(args.frozen_layers)

    print("INPUT PARAMETERS: ")
    print("base_dir: " + base_dir)
    print("nb_epochs: " + str(nb_epochs))
    print("batch_size: " + str(batch_size))
    print("patience: " + str(patience))
    print("fc_size: " + str(fc_size))
    print("modelToUse: " + modelToUse)
    print("frozen_layers: " + str(frozenLayers))
    print("enableTransferLearning: " + str(enableTransferlearning))
    print("enableFineTuning: " + str(enableFinetuning))
    print("******************************************************")

    # exit()

    # print(args)

    if modelToUse == VGG19_MODEL:
        img_width, img_height = 256, 256
        if frozenLayers == -1:
            frozenLayers = 5
    elif modelToUse == ResNet50_MODEL:
        img_width, img_height = 224, 224
        if frozenLayers == -1:
            frozenLayers = 160
    elif modelToUse == InceptionResNetV2_MODEL:
        img_width, img_height = 299, 299
        if frozenLayers == -1:
            frozenLayers = 600

    elif modelToUse == InceptionV3_MODEL:
        img_width, img_height = 299, 299
        if frozenLayers == -1:
            frozenLayers = 600

    elif modelToUse == NASNetLarge_MODEL:
        img_width, img_height = 331, 331
        if frozenLayers == -1:
            frozenLayers = 600
    else:
        print("Model " + modelToUse + " unaivalable!")
        print(
            "Model available: " + VGG19_MODEL + ", " + ResNet50_MODEL + ", " + InceptionResNetV2_MODEL + ", " + NASNetLarge_MODEL)
        exit()

    train_data_dir = base_dir + "/Train"
    validation_data_dir = base_dir + "/Val"

    nb_train_samples = get_nb_files(train_data_dir)
    nb_validation_samples = get_nb_files(validation_data_dir)
    nb_classes = len(glob.glob(train_data_dir + "/*"))
    save_dir = 'Result2/' + modelToUse + '/' + base_dir[-5:]

    fileNameForModel = save_dir + "_TWO_STEP" + "_epo_" + str(nb_epochs) + "_pat_" + str(patience) + "_batch_" + str(
        batch_size) + "_frozen_" + str(frozenLayers)
    transferLearningModelName = fileNameForModel + "_tl.h5"
    fineTuneModelName = fileNameForModel + "_ft.h5"

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        rotation_range=20)

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        rotation_range=20)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical")

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        # batch_size=batch_size,
        class_mode="categorical")

    if modelToUse == VGG19_MODEL:
        base_model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    elif modelToUse == ResNet50_MODEL:
        base_model = applications.resnet50.ResNet50(weights="imagenet", include_top=False,
                                                    input_shape=(img_width, img_height, 3))
    elif modelToUse == InceptionResNetV2_MODEL:
        base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    elif modelToUse == InceptionV3_MODEL:
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    elif modelToUse == NASNetLarge_MODEL:
        base_model = NASNetLarge(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    model = add_new_last_layer(base_model, nb_classes, fc_size)

    if enableTransferlearning:

        print("*********************************************************************************************")
        print("*********************************************************************************************")
        print("                                TRANSFER LEARNING - " + transferLearningModelName)
        print("*********************************************************************************************")
        print("*********************************************************************************************")

        checkpoint = ModelCheckpoint(transferLearningModelName, monitor='val_acc', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir="logs_transfer/{}".format(time()))

        setup_to_transfer_learn(model, base_model)

        print(model.summary())

        if recoveryTraining:
            print("Load weights from : " + transferLearningModelName)
            model.load_weights(transferLearningModelName)
            print("Restart transfer learning ... ")
        else:
            print("Start transfer learning ... ")

        # print(model.metrics_names)
        # print(model.count_params())

        history_tl = model.fit_generator(
            train_generator,
            nb_epoch=nb_epochs,
            samples_per_epoch=nb_train_samples,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,
            class_weight='auto',
            callbacks=[checkpoint, early, tensorboard])

        print("*********************************************************************************************")
        print("*********************************************************************************************")
        print("                          TRANSFER LEARNING COMPLETED - " + transferLearningModelName)
        print("*********************************************************************************************")
        print("*********************************************************************************************")

    if enableFinetuning:
        print("*********************************************************************************************")
        print("*********************************************************************************************")
        print("                                FINE TUNING - " + fineTuneModelName)
        print("*********************************************************************************************")
        print("*********************************************************************************************")

        setup_to_finetune(model, frozenLayers)

        print(model.summary())

        if recoveryTraining:
            print("Load weights from : " + fineTuneModelName)
            model.load_weights(fineTuneModelName)
            print("Restart fine tuning... ")
        else:
            print("Load weights from : " + transferLearningModelName)
            model.load_weights(transferLearningModelName)
            print("Start fine tuning ... ")

        checkpoint = ModelCheckpoint(fineTuneModelName, monitor='val_acc', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=1, mode='auto')
        tensorboard = TensorBoard(log_dir="logs_fine/{}".format(time()))

        # print(model.count_params())
        # print(model.metrics_names)

        history_ft = model.fit_generator(
            train_generator,
            nb_epoch=nb_epochs,
            samples_per_epoch=nb_train_samples,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,
            class_weight='auto',
            callbacks=[checkpoint, early, tensorboard])

        if args.plot:
            plot_training(history_ft)

        print("*********************************************************************************************")
        print("*********************************************************************************************")
        print("                                FINE TUNING COMPLETED - " + fineTuneModelName)
        print("*********************************************************************************************")
        print("*********************************************************************************************")
    return model, fineTuneModelName, transferLearningModelName, img_width, img_height


def test(args, model, fineTuneModelName, img_width, img_height):
    print("*********************************************************************************************")
    print("*********************************************************************************************")
    print("                                TESTING WITH - " + fineTuneModelName)
    print("*********************************************************************************************")
    print("*********************************************************************************************")

    target_names = ["Pilatos", "isak", "Dioscorus", "Hermauos"]
    # "Abraamios", "Andreas","Dios",,"Hermauos","isak", "Kyros1", "Kyros3", "Menas", "Pilatos", "Victor"  "Dioscorus", "Pilatos", "Victor"
    if model == None:
        print("Load weights from : " + fineTuneModelName)
        model.load_weights(fineTuneModelName)

    base_dir = args.base_dir
    test_data_dir = base_dir + "/Test"

    nb_test_samples = get_nb_files(test_data_dir)
    nb_classes = len(glob.glob(test_data_dir + "/*"))

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=1)
    model.load_weights(fineTuneModelName)

    print("Prediction of " + str(nb_test_samples) + " 2-char.")
    probabilities = model.predict_generator(test_generator, nb_test_samples)

    probabilities = np.argmax(probabilities, axis=1)

    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, probabilities)
    print(cm)

    accuracy, accuracyByRow = evalAccuracy(cm)
    print("Accuracy: " + str(accuracy))
    print("Accuracy by 2-char:")

    for i in range(0, len(accuracyByRow)):
        print("\t" + target_names[i] + " => " + str(accuracyByRow[i]))
    print('Classification Report')

    print(classification_report(test_generator.classes, probabilities, target_names=target_names))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--base_dir", default=base_dir)
    a.add_argument("--nb_epoch", default=nb_epochs)
    a.add_argument("--batch_size", default=batch_size)
    a.add_argument("--patience", default=patience)
    a.add_argument("--fc_size", default=fc_size)
    a.add_argument("--base_model", default="VGG19")
    a.add_argument("--frozen_layers", default=-1)
    a.add_argument("--enable_transferlearning", default="y")
    a.add_argument("--enable_finetuning", default="y")
    a.add_argument("--plot", action="store_true")
    args = a.parse_args()

    model, fineTuneModelName, transferLearningModelName, img_width, img_height = train(args)

    test(args, model, transferLearningModelName, img_width, img_height)
    test(args, model, fineTuneModelName, img_width, img_height)


