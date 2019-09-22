import gc
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.models import Model
from keras.layers import GlobalAveragePooling2D 
from keras.layers import Dense
import os
import shutil
import pandas as pd
import random

random.seed(1)


def load_data(label_list, train_pct=0.8):
    """
    This method loads in the training and testing data from your file structure.
    :return: Training and testing dictionaries
    """

    df = pd.read_excel("/Users/aditya/Desktop/MobileNet/EyeDetection/Integration/Final-YEET/ODIR-5K_Training_Annotations(Updated)_V2.xlsx")

    left_filenames = df["Left-Fundus"].tolist()
    right_filenames = df["Right-Fundus"].tolist()
    filenames = left_filenames + right_filenames
    left_diseases = df["Left-Diagnostic Keywords"].tolist()
    right_diseases = df["Right-Diagnostic Keywords"].tolist()
    diseases = left_diseases + right_diseases
    
    train = {}
    test = {}
    for i in range(len(diseases)):
        split = diseases[i].split("，")
        disease_list = []
        for d in split:
            if d == "normal fundus":
                disease_list = []
                break
            elif d in label_list:
                disease_list.append(label_list.index(d))
        filepath = filenames[i].split('.')[0] + '.png'  # change this to be right
        if random.random() < train_pct:
            train[filepath] = disease_list
        else:
            test[filepath] = disease_list
    return train, test


class MobileNetSingleClassifier:
    def __init__(self, disease_idx, master_train, master_test, num_epochs=4, base_lr=0.01, lr_decay=0.5,
                 batch_size=36, base_dataset_filepath="/Users/aditya/Desktop/MobileNet/EyeDetection/Integration/Final-YEET/Cropped/",
                 working_dataset_filepath="/Users/aditya/Desktop/MobileNet/EyeDetection/Integration/Final-YEET/New-Images/"):  # change this up here
        """
        Makes the MobileNet model for 1 class
        :param disease_idx: The idx of the disease to make this model for
        :param master_train: The master train dataset
        :param master_test: The master test dataset
        :param num_epochs: Number of epochs to train for
        :param base_lr: The base learning rate
        :param lr_decay: The multiplier on the (normalized) learning rate decay rate
        :param batch_size: Batch size, might need to be lowered due to memory constraints
        """
        self.disease_idx = disease_idx
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._base_dataset_filepath = base_dataset_filepath
        #print(len(os.walk(base_dataset_filepath).next()[2]))
        self._working_dataset_filepath = working_dataset_filepath

        # Now we copy the data to the correct folders *for this particular model*
        self._copy_dataset(master_train, master_test)
        self._train_len = len(master_train.items())
        print(self._train_len)
        self._test_len = len(master_test.items())
        print(self._test_len)
        # Next, let's make the data augmenter
        path = os.path.join(self._working_dataset_filepath, "train")
        self._train_generator = self._make_datagen().flow_from_directory(directory=path,
                                                                         target_size=(224, 224),
                                                                         color_mode="rgb",
                                                                         batch_size=self._batch_size,
                                                                         class_mode="categorical",
                                                                         shuffle=True,
                                                                         seed=1)
        path = os.path.join(self._working_dataset_filepath, "test")
        self._test_generator = self._make_datagen().flow_from_directory(directory=path,
                                                                        target_size=(224, 224),
                                                                        color_mode="rgb",
                                                                        batch_size=self._batch_size,
                                                                        class_mode="categorical",
                                                                        shuffle=True,
                                                                        seed=1)

        # Make a few callbacks and logging stuff
        self.logdir = "/Users/aditya/Desktop/MobileNet/EyeDetection/Integration/Logs_{}".format(disease_idx)
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        checkpoint = ModelCheckpoint(os.path.join(self.logdir, "best_checkpoint.h5"),
                                     monitor="val_acc", verbose=1, save_best_only=True, save_weights_only=False)
        # TensorBoard is a really nice tool, use it
        tensorboard = TensorBoard(log_dir=self.logdir, update_freq=1000)
        # This is the early stopping, which might help to prevent overfitting but for now I've left it out
        early_stopping = EarlyStopping(monitor="val_acc", min_delta=0, patience=10)
        # This is learning rate scheduling, if you want it rather than decay, also left out for now
        #lr_scheduler = LearningRateScheduler(lambda step, lr: lr_schedule[step], verbose=1)
        self._callbacks = [checkpoint, tensorboard]

        # This is how we make the MobileNet model
        self._model = MobileNetV2(input_shape=(224, 224, 3), alpha = 1.0, include_top=False, weights='imagenet')

        # Top Model Block
        x = self._model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(2, activation='softmax')(x)

        # add your top layer block to your base model
        self._model = Model(self._model.input, predictions)
        
        # Here, we can choose to freeze any number of the layers, up to the last layer (there are 132 layers, so that
        # would be :131). For now I'm freezing none of them
        for layer in self._model.layers[:0]:
            layer.trainable = False

        # Define the optimizer (SGD with momentum is common so I use it here, can be changed)
        # LR decay is used, to change to fixed schedule use the callback
        decay_rate = lr_decay * base_lr / self._num_epochs
        optimizer = SGD(lr=base_lr, momentum=0.9, nesterov=True, decay=decay_rate)

        # This is the right loss function for most classification tasks
        loss = "categorical_crossentropy"

        # Actually compile the model
        self._model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    @staticmethod
    def _make_datagen():
        """
        Quick method to make the ImageDataGenerator
        :return: ImageDataGenerator
        """
        # I don't think anyone really knows the best parameters for this thing
        return ImageDataGenerator(rescale=1. / 255,
                                  horizontal_flip=True,
                                  fill_mode="nearest",
                                  zoom_range=0.3,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  rotation_range=30)

    @staticmethod
    def load_sample(filepath):
        return img_to_array(load_img(filepath))

    def _copy_dataset(self, train, test):
        """
        Makes the dataset for this particular disease_idx, and copies the images to the appropriate directories
        :param train: The train dataset to convert
        :param test: The test dataset to convert
        :return: None
        """
        train_positive_filepath = os.path.join(self._working_dataset_filepath, "train", "positive")
        train_negative_filepath = os.path.join(self._working_dataset_filepath, "train", "negative")
        test_positive_filepath = os.path.join(self._working_dataset_filepath, "test", "positive")
        test_negative_filepath = os.path.join(self._working_dataset_filepath, "test", "negative")

        # Nuke the directories then remake
        shutil.rmtree(self._working_dataset_filepath)
        os.mkdir(self._working_dataset_filepath)
        os.mkdir(os.path.join(self._working_dataset_filepath, "train"))
        os.mkdir(os.path.join(self._working_dataset_filepath, "test"))
        os.mkdir(train_positive_filepath)
        os.mkdir(train_negative_filepath)
        os.mkdir(test_positive_filepath)
        os.mkdir(test_negative_filepath)
        print(train.items())
        # Now copy over the files
        for f, disease_list in train.items():
            src = os.path.join(self._base_dataset_filepath, f)
            if self.disease_idx in disease_list:
                dst = os.path.join(train_positive_filepath, f)
            else:
                dst = os.path.join(train_negative_filepath, f)
            shutil.copyfile(src, dst)
        for f, disease_list in test.items():
            src = os.path.join(self._base_dataset_filepath, f)
            if self.disease_idx in disease_list:
                dst = os.path.join(test_positive_filepath, f)
            else:
                dst = os.path.join(test_negative_filepath, f)
            shutil.copyfile(src, dst)
        

    def train(self):
        """
        Train the model
        :return: The History object from Keras
        """
        steps_per_epoch = self._train_len // self._batch_size  # Quick math, round to int
        validation_steps = self._test_len // self._batch_size  # Following the docs
        history = self._model.fit_generator(self._train_generator,
                                            verbose=1,  # Either 1 or 2
                                            callbacks=self._callbacks,  # Add in our callbacks
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=self._num_epochs,
                                            validation_data=self._test_generator,
                                            validation_steps=validation_steps)

        return history

    def save(self):
        """
        Saves the model (can be reloaded with simply `keras.models.load_model(filepath)`)
        :return: None
        """
        self._model.save(os.path.join(self.logdir, "save.h5"))

    def evaluate(self):
        """
        I don't really feel like writing this right know but you should be able to do something like
        self._model.evaluate or self._model.evaluate_generator
        :return:
        """
        pass


if __name__ == "__main__":
    label_list = ["cataract", "moderate non proliferative retinopathy", "branch retinal artery occlusion", "macular epiretinal membrane", "mild nonproliferative retinopathy", "epiretinal membrane", "drusen", "vitreous degeneration", "hypertensive retinopathy", "retinal pigmentation", "pathological myopia", "myelinated nerve fibers", "rhegmatogenous retinal detachment", "lens dust", "depigmentation of the retinal pigment epithelium", "abnormal pigment ", "glaucoma", "spotted membranous change", "macular hole", "wet age-related macular degeneration", "dry age-related macular degeneration", "epiretinal membrane over the macula", "central retinal artery occlusion", "pigment epithelium proliferation", "diabetic retinopathy", "atrophy", "chorioretinal atrophy", "white vessel", "retinochoroidal coloboma", "atrophic change", "retinitis pigmentosa", "retina fold", "suspected glaucoma", "branch retinal vein occlusion", "optic disc edema", "retinal pigment epithelium atrophy", "severe nonproliferative retinopathy", "proliferative diabetic retinopathy", "refractive media opacity", "suspected microvascular anomalies", "severe proliferative diabetic retinopathy", "central retinal vein occlusion", "tessellated fundus", "maculopathy", "oval yellow-white atrophy", "suspected retinal vascular sheathing", "macular coloboma", "vessel tortuosity", "hypertensive retinopathy,diabetic retinopathy", "idiopathic choroidal neovascularization", "wedge-shaped change", "optic nerve atrophy", "old chorioretinopathy", "low image quality,maculopathy", "punctate inner choroidopathy", "myopia retinopathy", "old choroiditis", "myopic maculopathy", "chorioretinal atrophy with pigmentation proliferation", "congenital choroidal coloboma", "optic disk epiretinal membrane", "diabetic retinopathy" , "maculopathy", "morning glory syndrome", "retinal pigment epithelial hypertrophy", "old branch retinal vein occlusion", "asteroid hyalosis", "retinal artery macroaneurysm", "suspicious diabetic retinopathy", "suspected diabetic retinopathy", "vascular loops", "diffuse chorioretinal atrophy", "optic discitis", "intraretinal hemorrhage", "pigmentation disorder", "arteriosclerosis", "retinal vascular sheathing", "suspected retinitis pigmentosa", "old central retinal vein occlusion", "diffuse retinal atrophy", "fundus laser photocoagulation spots", "suspected abnormal color of  optic disc", "myopic retinopathy", "vitreous opacity", "macular pigmentation disorder", "suspected moderate non proliferative retinopathy", "suspected macular epimacular membrane", "peripapillary atrophy", "retinal detachment", "anterior segment image", "central serous chorioretinopathy", "suspected cataract", "age-related macular degeneration", "intraretinal microvascular abnormality"]  # You can just hardcode this or load in from a file
    master_train, master_test = load_data(label_list)

    # Nice summary of the model layers
    demo_model = MobileNetV2(include_top=False, classes=95, weights="imagenet")
    # demo_model.summary()
    del demo_model
    K.clear_session()
    gc.collect()

    logdirs = []

    # Go through every class, train a classifier, and save it
    for class_idx in range(len(label_list)):
        print("On model {} / {}".format(class_idx, len(label_list)))
        model = MobileNetSingleClassifier(class_idx, master_train, master_test)
        logdirs.append(model.logdir)
        model.train()
        model.save()
        del model
        K.clear_session()
        gc.collect()  # This should release the memory and VRAM of this model, but I'm not totally sure

    '''
    # Now to evaluate on a sample, you just do something like:
    sample = MobileNetSingleClassifier.load_sample("path/to/image.png")
    class_confidences = []
    for class_idx in range(len(label_list)):
        model = load_model(os.path.join(logdirs[class_idx], "save.h5"))
        y = model.predict(sample, batch_size=1)[0]
        class_confidences.append((label_list[class_idx], y[0]))
        del model
        K.clear_session()
        gc.collect()  # Again I think this is how to release the resources

    for t in class_confidences:
        print("The model thought the sample had a {0:.2f}% chance of having {1}".format(str(100 * t[1]), t[0]))
    '''