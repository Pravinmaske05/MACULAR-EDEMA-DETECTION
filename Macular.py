import tkinter as tk
import tkinter
from tkinter import *
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import tkinter.font as tkFont
from tkinter import messagebox, filedialog, Toplevel
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import cv2
from skimage.feature import hog
from sklearn import svm, neighbors, naive_bayes
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import datetime

output_path = None
image = None
file_path = None
hog_features = None
denoised_nlmeans = None
lbp_features = None
prediction_type = None


# Create the main window
root = tk.Tk()
root.title('Macular Edema Diagnostic Software')
root.geometry('2000x1600')

# Function to start diagnosis
def start_diagnosis():
    # Hide the patient info page
    patient_info_frame.pack_forget()

    # Create the diagnosis page
    diagnosis_frame.pack(fill='both', expand=True)

# Function to go back to the patient info page
def go_back():
    # Hide the diagnosis page
    diagnosis_frame.pack_forget()

    # Show the patient info page
    patient_info_frame.pack(fill='both', expand=True)

# Function to perform image filtering
def rootly_filter(filter_name):
    # Here, you can implement the image filtering logic based on the selected filter
    messagebox.showinfo('Filter Applied', f'Applied {filter_name} filter')

# Function to select a diagnostic model
def select_model(model_name):
    messagebox.showinfo('Model Selected', f'You selected the {model_name} model.')


def import_():
    global image
    global denoised_nlmeans
    global file_path

    file_path = filedialog.askopenfilename(title='Select OCT File', filetypes=(('bmp files', '*.bmp'),))
    if file_path:
        image = Image.open(file_path)
        # Display the image in a label
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(import_tab, image=photo)
        image_label.pack()

        if image_label:
            image_label.config(image=photo)
        else:

            image_label.image = photo  # Keep a reference to prevent image from being garbage collected

        # Store the loaded image
        loaded_image = image

        # Display the image in a label
        photo = ImageTk.PhotoImage(loaded_image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to prevent image from being garbage collected

        image_window = Toplevel()
        image_window.geometry('500x400')
        image_window.title('Image viewer')

        loaded_photo = ImageTk.PhotoImage(image)
        loaded_label = tk.Label(image_window, image=loaded_photo)
        loaded_label.photo = loaded_photo
        loaded_label.pack()

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is not None:
            # # Resize the image
            # new_width = 400
            # new_height = 300
            # resized_image = cv2.resize(image, (new_width, new_height))
            # # Scaling the image
            # scale_x = 0.5
            # scale_y = 0.5
            # new_width = int(image.shape[1] * scale_x)
            # new_height = int(image.shape[0] * scale_y)
            # scaled_image = cv2.resize(resized_image, (new_width, new_height))
            # Grayscale conversion
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Add Non-local Means Filter and calculate metrics
            denoised_nlmeans = cv2.fastNlMeansDenoising(gray_image, None, h=10, searchWindowSize=21)
            # Specify the path where you want to save the binary image
            output_path = 'filtered_image.jpg'  # Replace with the desired file path and format (e.g., .jpg, .png)
            # Save the binary image
            cv2.imwrite(output_path, denoised_nlmeans)
            # Load the image
            image = cv2.imread(output_path)  # Replace 'path_to_your_image.jpg' with the actual image file path

def hog_():
    global image
    global denoised_nlmeans
    global hog_features
    global labels
    global feature_vector

    # create empty lists to append extracted features and labels
    feature_vector = []
    labels = []
    # assign label; 1= normal
    label = 1
    # provide input path of images
    path = 'D:/University stuff/Advanced Programming/Dataset/original dataset/train/normal/'
    # provide path to which features and labels should be saved
    out_path = 'D:/University stuff/Advanced Programming/Dataset/hog/'
    process = 'hog'

    # create directory
    try:
        os.makedirs(out_path + process)
    except:
        pass

    dr = os.listdir(path)
    # open a new file and create csv file
    f = open(out_path + 'hog_features_normal.csv', 'w')

    num = len(dr)
    # looping the hog function for all images in the given path
    for ii in range(num):
        image = cv2.imread(path + dr[ii])
        gray_image = None  # Initialize gray_image here
        # Check if the image is loaded successfully
        if image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            print(f"Failed to load the image: {path + dr[ii]}. Deleting the image and moving on.")
            # Delete the image file from the specified location if it exists
            if os.path.exists(path + dr[ii]):
                os.remove(path + dr[ii])
            else:
                print("The image file does not exist at the specified location.")
        # hog function
        if gray_image is not None:
            hog_features, hog_image = hog(gray_image, visualize=True)
            print(hog_features[0:])
            print(hog_features[0:].shape)
            for jj in range(len(hog_features)):
                # writing to f
                f.write(str(hog_features[jj]))

            # append hog_features to the feature_vector list
            feature_vector.append(hog_features)
            # append label to the labels list
            labels.append(label)
            # close the opened file
    f.close()
    # label 0=abnormal
    label = 0
    path = 'D:/University stuff/Advanced Programming/Dataset/original dataset/train/abnormal/'
    out_path = 'D:/University stuff/Advanced Programming/Dataset/hog/'
    process = 'hog'

    # create directory
    try:
        os.makedirs(out_path + process)
    except:
        pass

    dr = os.listdir(path)
    # open a new file and create csv file
    f = open(out_path + 'hog_features_abnormal.csv', 'w')

    num = len(dr)
    # looping the hog function for all images in the given path
    for ii in range(num):
        image = cv2.imread(path + dr[ii])
        gray_image = None
        # Check if the image is loaded successfully
        if image is not None:
            # Convert the image to grayscale
            print(ii, dr[ii])
            # convert to gray scale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        else:
            print("Failed to load the image. Deleting the image and moving on.")

            # Delete the image file from the specified location
            if os.path.exists(path + dr[ii]):
                os.remove(path + dr[ii])
            else:
                print("The image file does not exist at the specified location.")
        # hog function
        if gray_image is not None:
            hog_features, hog_image = hog(gray_image, visualize=True)
            print(hog_features[0:].shape)
            # print(str(hog_features[0]))
            for jj in range(len(hog_features)):
                # writing to f
                f.write(str(hog_features[jj]))
            # append hog_features to the feature_vector list
            feature_vector.append(hog_features)
            # append label to the labels list
            labels.append(label)

    ft = np.array(feature_vector)
    import pickle
    f = open(out_path + 'hog_feature.pkl', 'wb')
    # save features as pickle file
    pickle.dump(feature_vector, f)
    f.close()
    f = open(out_path + 'hog_labels.pkl', 'wb')
    # save labels as pickle file
    pickle.dump(labels, f)
    # close the opened file
    f.close()

def lbp_():
    global image
    global denoised_nlmeans
    global ft
    global labels
    global feature_vector
    # create empty lists for features and labels
    feature_vector = []
    labels = []
    # label 1=normal
    label = 1
    # give path of images to be imported
    path = 'D:/University stuff/Advanced Programming/Dataset/original dataset/train/normal/'
    # give output path to save the extracted features and labels
    out_path = 'D:/University stuff/Advanced Programming/Dataset/lbp/'
    process = 'lbp'

    # create directory
    try:
        os.makedirs(out_path + process)
    except:
        pass

    dr = os.listdir(path)
    # open file
    f = open(out_path + 'lbp_features_normal.csv', 'w')

    num = len(dr)
    # loop the lbp function for all images in the path
    for ii in range(num):
        image = cv2.imread(path + dr[ii])
        gray_image = None
        # Check if the image is loaded successfully
        if image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            print(f"Failed to load the image: {path + dr[ii]}. Deleting the image and moving on.")
            # Delete the image file from the specified location if it exists
            if os.path.exists(path + dr[ii]):
                os.remove(path + dr[ii])
            else:
                print("The image file does not exist at the specified location.")
        # Compute LBP features
        radius = 1
        n_points = 8 * radius
        # lbp function
        if gray_image is not None:
            lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            # flatten the image data
            ft = np.array(lbp_image[0:].flatten())
            for jj in range(len(ft)):
                # write the features into file
                f.write(str(ft[jj]))
                # append features to feature_vector list
            feature_vector.append(ft)
            # append label to labels list
            labels.append(label)

    # 0=abnormal
    label = 0
    # give paths to import images
    path = 'D:/University stuff/Advanced Programming/Dataset/original dataset/train/abnormal/'
    # give output path for saving the features
    out_path = 'D:/University stuff/Advanced Programming/Dataset/lbp/'
    process = 'lbp'

    # create directory
    try:
        os.makedirs(out_path + process)
    except:
        pass

    dr = os.listdir(path)
    # open file
    f = open(out_path + 'lbp_features_abnormal.csv', 'w')

    num = len(dr)
    # loop lbp function in all images
    for ii in range(num):
        image = cv2.imread(path + dr[ii])
        gray_image = None
        # Check if the image is loaded successfully
        if image is not None:
            # Convert the image to grayscale
            print(ii, dr[ii])
            # convert to gray scale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        else:
            print("Failed to load the image. Deleting the image and moving on.")

            # Delete the image file from the specified location
            if os.path.exists(path + dr[ii]):
                os.remove(path + dr[ii])
            else:
                print("The image file does not exist at the specified location.")
        print(ii, dr[ii])
        # Compute LBP features
        radius = 1
        n_points = 8 * radius
        # lbp function
        if gray_image is not None:
            lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

            ft = np.array(lbp_image[0:].flatten())
            for jj in range(len(ft)):
                # write to file
                f.write(str(ft[jj]))
            # append to feature_vector list
            feature_vector.append(ft)
            # append to labels list
            labels.append(label)

    ft = np.array(feature_vector)
    print(ft.shape)
    import pickle
    f = open(out_path + 'lbp_feature_train.pkl', 'wb')
    # dump features into a pickle file
    pickle.dump(feature_vector, f)
    f.close()
    f = open(out_path + 'lbp_labels_train.pkl', 'wb')
    # dump labels into pickle file
    pickle.dump(labels, f)
    # close the opened file
    f.close()

def lbp_svm_model():
    global image
    global denoised_nlmeans
    global labels
    global feature_vector
    lbp_()

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    import pickle
    import numpy as np
    from matplotlib import pyplot as plt

    out_path = 'D:/University stuff/Advanced Programming/Dataset/lbp/'

    import pickle
    f = open(out_path + 'lbp_feature_train.pkl', 'rb')
    feature_vector = pickle.load(f)
    f.close()
    f = open(out_path + 'lbp_labels_train.pkl', 'rb')
    labels = pickle.load(f)
    f.close()

    # Create an SVM classifier and train it on the training data
    clf = svm.SVC()
    # clf = GaussianNB()
    # clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(feature_vector, labels)

    # Save the trained SVM classifier to a file using pickle
    with open(out_path + 'svm_lbp_model.pkl', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

    # f = open(out_path + 'lbp_feature_test.pkl', 'rb')
    # feature_vector1 = pickle.load(f)
    # f.close()
    # f = open(out_path + 'lbp_labels_test.pkl', 'rb')
    # labels1 = pickle.load(f)
    # f.close()
    #
    # # Use the trained classifier to make predictions on the test data
    # predicted = clf.predict(feature_vector1)
    #
    # feature_vector2 = np.array(feature_vector1)
    # print(feature_vector2.shape)
    # predicted = clf.predict(feature_vector2)
    #
    # error = sum(labels1 - predicted) / len(labels1)
    # print(error)
    #
    # target_names = ['Normal', 'Abnormal']
    # print(classification_report(labels1, predicted, target_names=target_names))
    #
    # # X_test=np.reshape(labels1 ,(-1,1))
    # y_test = labels1
    # X_test = feature_vector1
    #
    # # Plot non-normalized confusion matrix
    # titles_options = [
    #     ("Confusion matrix, without normalization", None),
    #     ("Normalized confusion matrix", "true"),
    # ]
    # for title, normalize in titles_options:
    #     disp = ConfusionMatrixDisplay.from_estimator(
    #         clf,
    #         X_test,
    #         y_test,
    #         display_labels=target_names,
    #         cmap=plt.cm.Blues,
    #         normalize=normalize,
    #     )
    #     disp.ax_.set_title(title)
    #
    #     # print(title)
    #     # print(disp.confusion_matrix)
    #
    # plt.show()
    #
    # print(title)
    #
    # cnf = disp.confusion_matrix
    # TP = cnf[0, 0]
    # FN = cnf[0, 1]
    # FP = cnf[1, 0]
    # TN = cnf[1, 1]
    #
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Sensisitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    # print('Accuracy (in %) :', round(accuracy * 100, 2))
    # print('Sensisitivity (in %) :', round(Sensisitivity * 100, 2))
    # print('Specificity (in %) :', round(Specificity * 100, 2))

def lbp_nb_model():
    global image
    global denoised_nlmeans
    global labels
    global feature_vector

    lbp_()

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    import pickle
    import numpy as np
    from matplotlib import pyplot as plt

    out_path = 'D:/University stuff/Advanced Programming/Dataset/lbp/'

    import pickle
    f = open(out_path + 'lbp_feature_train.pkl', 'rb')
    feature_vector = pickle.load(f)
    f.close()
    f = open(out_path + 'lbp_labels_train.pkl', 'rb')
    labels = pickle.load(f)
    f.close()

    # Create an SVM classifier and train it on the training data
    #clf = svm.SVC()
    clf = GaussianNB()
    # clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(feature_vector, labels)

    # Save the trained SVM classifier to a file using pickle
    with open(out_path + 'nb_lbp_model.pkl', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

    # f = open(out_path + 'lbp_feature_test.pkl', 'rb')
    # feature_vector1 = pickle.load(f)
    # f.close()
    # f = open(out_path + 'lbp_labels_test.pkl', 'rb')
    # labels1 = pickle.load(f)
    # f.close()
    #
    # # Use the trained classifier to make predictions on the test data
    # predicted = clf.predict(feature_vector1)
    #
    # feature_vector2 = np.array(feature_vector1)
    # print(feature_vector2.shape)
    # predicted = clf.predict(feature_vector2)
    #
    # error = sum(labels1 - predicted) / len(labels1)
    # print(error)
    #
    # target_names = ['Normal', 'Abnormal']
    # print(classification_report(labels1, predicted, target_names=target_names))
    #
    # # X_test=np.reshape(labels1 ,(-1,1))
    # y_test = labels1
    # X_test = feature_vector1
    #
    # # Plot non-normalized confusion matrix
    # titles_options = [
    #     ("Confusion matrix, without normalization", None),
    #     ("Normalized confusion matrix", "true"),
    # ]
    # for title, normalize in titles_options:
    #     disp = ConfusionMatrixDisplay.from_estimator(
    #         clf,
    #         X_test,
    #         y_test,
    #         display_labels=target_names,
    #         cmap=plt.cm.Blues,
    #         normalize=normalize,
    #     )
    #     disp.ax_.set_title(title)
    #
    #     # print(title)
    #     # print(disp.confusion_matrix)
    #
    # plt.show()
    #
    # print(title)
    #
    # cnf = disp.confusion_matrix
    # TP = cnf[0, 0]
    # FN = cnf[0, 1]
    # FP = cnf[1, 0]
    # TN = cnf[1, 1]
    #
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Sensisitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    # print('Accuracy (in %) :', round(accuracy * 100, 2))
    # print('Sensisitivity (in %) :', round(Sensisitivity * 100, 2))
    # print('Specificity (in %) :', round(Specificity * 100, 2))

def lbp_knn_model():
    global image
    global denoised_nlmeans
    global labels
    global feature_vector

    lbp_()

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    import pickle
    import numpy as np
    from matplotlib import pyplot as plt

    out_path = 'D:/University stuff/Advanced Programming/Dataset/lbp/'

    import pickle
    f = open(out_path + 'lbp_feature_train.pkl', 'rb')
    feature_vector = pickle.load(f)
    f.close()
    f = open(out_path + 'lbp_labels_train.pkl', 'rb')
    labels = pickle.load(f)
    f.close()

    # Create an SVM classifier and train it on the training data
    # clf = svm.SVC()
    # clf = GaussianNB()
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(feature_vector, labels)

    # Save the trained SVM classifier to a file using pickle
    with open(out_path + 'knn_lbp_model.pkl', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

    # f = open(out_path + 'lbp_feature_test.pkl', 'rb')
    # feature_vector1 = pickle.load(f)
    # f.close()
    # f = open(out_path + 'lbp_labels_test.pkl', 'rb')
    # labels1 = pickle.load(f)
    # f.close()
    #
    # # Use the trained classifier to make predictions on the test data
    # predicted = clf.predict(feature_vector1)
    #
    # feature_vector2 = np.array(feature_vector1)
    # print(feature_vector2.shape)
    # predicted = clf.predict(feature_vector2)
    #
    # error = sum(labels1 - predicted) / len(labels1)
    # print(error)
    #
    # target_names = ['Normal', 'Abnormal']
    # print(classification_report(labels1, predicted, target_names=target_names))
    #
    # # X_test=np.reshape(labels1 ,(-1,1))
    # y_test = labels1
    # X_test = feature_vector1
    #
    # # Plot non-normalized confusion matrix
    # titles_options = [
    #     ("Confusion matrix, without normalization", None),
    #     ("Normalized confusion matrix", "true"),
    # ]
    # for title, normalize in titles_options:
    #     disp = ConfusionMatrixDisplay.from_estimator(
    #         clf,
    #         X_test,
    #         y_test,
    #         display_labels=target_names,
    #         cmap=plt.cm.Blues,
    #         normalize=normalize,
    #     )
    #     disp.ax_.set_title(title)
    #
    #     # print(title)
    #     # print(disp.confusion_matrix)
    #
    # plt.show()
    #
    # print(title)
    #
    # cnf = disp.confusion_matrix
    # TP = cnf[0, 0]
    # FN = cnf[0, 1]
    # FP = cnf[1, 0]
    # TN = cnf[1, 1]
    #
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Sensisitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    # print('Accuracy (in %) :', round(accuracy * 100, 2))
    # print('Sensisitivity (in %) :', round(Sensisitivity * 100, 2))
    # print('Specificity (in %) :', round(Specificity * 100, 2))

def hog_svm_model():
    global image
    global denoised_nlmeans
    global labels
    global feature_vector

    hog_()

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    import pickle
    import numpy as np
    from matplotlib import pyplot as plt

    out_path = 'D:/University stuff/Advanced Programming/Dataset/hog/'

    import pickle
    f = open(out_path + 'hog_feature.pkl', 'rb')
    feature_vector = pickle.load(f)
    f.close()
    f = open(out_path + 'hog_labels.pkl', 'rb')
    labels = pickle.load(f)
    f.close()

    # Create an SVM classifier and train it on the training data
    clf = svm.SVC()
    # clf = GaussianNB()
    # clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(feature_vector, labels)

    # Save the trained SVM classifier to a file using pickle
    with open(out_path + 'svm_hog_model.pkl', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

    # f = open(out_path + 'hog_feature_test.pkl', 'rb')
    # feature_vector1 = pickle.load(f)
    # f.close()
    # f = open(out_path + 'hog_labels_test.pkl', 'rb')
    # labels1 = pickle.load(f)
    # f.close()
    #
    # # Use the trained classifier to make predictions on the test data
    # predicted = clf.predict(feature_vector1)
    #
    # feature_vector2 = np.array(feature_vector1)
    # print(feature_vector2.shape)
    # predicted = clf.predict(feature_vector2)
    #
    # error = sum(labels1 - predicted) / len(labels1)
    # print(error)
    #
    # target_names = ['Normal', 'Abnormal']
    # print(classification_report(labels1, predicted, target_names=target_names))
    #
    # # X_test=np.reshape(labels1 ,(-1,1))
    # y_test = labels1
    # X_test = feature_vector1
    #
    # # Plot non-normalized confusion matrix
    # titles_options = [
    #     ("Confusion matrix, without normalization", None),
    #     ("Normalized confusion matrix", "true"),
    # ]
    # for title, normalize in titles_options:
    #     disp = ConfusionMatrixDisplay.from_estimator(
    #         clf,
    #         X_test,
    #         y_test,
    #         display_labels=target_names,
    #         cmap=plt.cm.Blues,
    #         normalize=normalize,
    #     )
    #     disp.ax_.set_title(title)
    #
    #     # print(title)
    #     # print(disp.confusion_matrix)
    #
    # plt.show()
    #
    # print(title)
    #
    # cnf = disp.confusion_matrix
    # TP = cnf[0, 0]
    # FN = cnf[0, 1]
    # FP = cnf[1, 0]
    # TN = cnf[1, 1]
    #
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Sensisitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    # print('Accuracy (in %) :', round(accuracy * 100, 2))
    # print('Sensisitivity (in %) :', round(Sensisitivity * 100, 2))
    # print('Specificity (in %) :', round(Specificity * 100, 2))

def hog_nb_model():
    global image
    global denoised_nlmeans
    global labels
    global feature_vector

    hog_()

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    import pickle
    import numpy as np
    from matplotlib import pyplot as plt

    out_path = 'D:/University stuff/Advanced Programming/Dataset/hog/'

    import pickle
    f = open(out_path + 'hog_feature.pkl', 'rb')
    feature_vector = pickle.load(f)
    f.close()
    f = open(out_path + 'hog_labels.pkl', 'rb')
    labels = pickle.load(f)
    f.close()

    # Create an SVM classifier and train it on the training data
    # clf = svm.SVC()
    clf = GaussianNB()
    # clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(feature_vector, labels)

    # Save the trained SVM classifier to a file using pickle
    with open(out_path + 'nb_hog_model.pkl', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

def hog_knn_model():
    global image
    global denoised_nlmeans
    global labels
    global feature_vector

    hog_()

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    import pickle
    import numpy as np
    from matplotlib import pyplot as plt

    out_path = 'D:/University stuff/Advanced Programming/Dataset/hog/'

    import pickle
    f = open(out_path + 'hog_feature.pkl', 'rb')
    feature_vector = pickle.load(f)
    f.close()
    f = open(out_path + 'hog_labels.pkl', 'rb')
    labels = pickle.load(f)
    f.close()

    # Create an SVM classifier and train it on the training data
    #clf = svm.SVC()
    # clf = GaussianNB()
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(feature_vector, labels)

    # Save the trained SVM classifier to a file using pickle
    with open(out_path + 'knn_hog_model.pkl', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

def hog_svm():
    #global output_path
    global hog_features
    global image
    global denoised_nlmeans
    global prediction_type

    hog_svm_model()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    hog_features = hog(gray_image, visualize=False)
    # Print and use the HOG features as needed
    print(hog_features)
    print(hog_features.shape)
    if hog_features is not None:
        # Load the classifier model
        with open('D:/University stuff/Advanced Programming/Dataset/hog/svm_hog_model.pkl', 'rb') as model_file:
            svm_classifier = pickle.load(model_file)
        # Classify the image using the trained model
        prediction_type = svm_classifier.predict([hog_features])
        print(prediction_type)

        if prediction_type == 0:
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 1:
            messagebox.showinfo('Result', 'Normal image')
    else:
        messagebox.showwarning('Warning', 'Features not computed')


def hog_knn():
    #global output_path
    global hog_features
    global image
    global denoised_nlmeans
    global prediction_type

    hog_knn_model()

    # # Convert the image to grayscale
    # gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    hog_features = hog(denoised_nlmeans, visualize=False)
    # Print and use the HOG features as needed
    print(hog_features)
    print(hog_features.shape)
    if hog_features is not None:
        # Load the classifier model
        with open('D:/University stuff/Advanced Programming/Dataset/hog/knn_hog_model.pkl', 'rb') as model_file:
            knn_classifier = pickle.load(model_file)
        # Classify the image using the trained model
        prediction_type = knn_classifier.predict([hog_features])
        print(prediction_type)

        if prediction_type == 0:
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 1:
            messagebox.showinfo('Result', 'Normal image')
    else:
        messagebox.showwarning('Warning', 'Features not computed')

def hog_nb():
    #global output_path
    global hog_features
    global image
    global denoised_nlmeans
    global prediction_type

    hog_nb_model()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    hog_features = hog(gray_image, visualize=False)
    # Print and use the HOG features as needed
    print(hog_features)
    print(hog_features.shape)
    if hog_features is not None:
        # Load the KNN classifier model
        with open('D:/University stuff/Advanced Programming/Dataset/hog/nb_hog_model.pkl', 'rb') as model_file:
            nb_classifier = pickle.load(model_file)
        # Classify the image using the trained model
        prediction_type = nb_classifier.predict([hog_features])
        print(prediction_type)

        if prediction_type == 0:
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 1:
            messagebox.showinfo('Result', 'Normal image')
    else:
        messagebox.showwarning('Warning', 'Features not computed')



def lbp_svm():
    global image
    global denoised_nlmeans
    global lbp_features
    global prediction_type

    lbp_svm_model()

    gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)
    # Compute LBP features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_features = np.array(lbp_image[0:].flatten())
    print('lbp_features =', lbp_features)
    print(lbp_features.shape)
    if lbp_features is not None:
        with open('D:/University stuff/Advanced Programming/Dataset/lbp/svm_lbp_model.pkl', 'rb') as model_file:
            svm_classifier = pickle.load(model_file)
        # Classify the image using the trained model
        prediction_type = svm_classifier.predict([lbp_features])
        print(prediction_type)

        if prediction_type == 0:
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 1:
            messagebox.showinfo('Result', 'Normal image')

    else:
        messagebox.showwarning('Warning', 'Features not computed')

def lbp_knn():
    global image
    global denoised_nlmeans
    global lbp_features
    global prediction_type

    lbp_knn_model()

    gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)
    # Compute LBP features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_features = np.array(lbp_image[0:].flatten())
    print('lbp_features =', lbp_features)
    print(lbp_features.shape)
    if lbp_features is not None:
        with open('D:/University stuff/Advanced Programming/Dataset/lbp/knn_lbp_model.pkl', 'rb') as model_file:
            knn_classifier = pickle.load(model_file)
        # Classify the image using the trained model
        prediction_type = knn_classifier.predict([lbp_features])
        print(prediction_type)

        if prediction_type == 0:
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 1:
            messagebox.showinfo('Result', 'Normal image')

    else:
        messagebox.showwarning('Warning', 'Features not computed')


def lbp_nb():
    global image
    global denoised_nlmeans
    global lbp_features
    global prediction_type

    lbp_nb_model()

    gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)
    # Compute LBP features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_features = np.array(lbp_image[0:].flatten())
    print('lbp_features =', lbp_features)
    print(lbp_features.shape)
    if lbp_features is not None:
        with open('D:/University stuff/Advanced Programming/Dataset/lbp/nb_lbp_model.pkl', 'rb') as model_file:
            nb_classifier = pickle.load(model_file)
        # Classify the image using the trained model
        prediction_type_type = nb_classifier.predict([lbp_features])
        print(prediction_type)

        if prediction_type == 0:
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 1:
            messagebox.showinfo('Result', 'Normal image')

    else:
        messagebox.showwarning('Warning', 'Features not computed')

def cnn():

    global denoised_nlmeans
    global prediction_type

    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Activation
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras import backend as K
    import matplotlib
    matplotlib.use("Agg")
    import tensorflow as tf

    # import the necessary packages
    from sklearn.metrics import confusion_matrix
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.utils import to_categorical
    from imutils import paths
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse
    import random
    import cv2
    import os
    import pickle

    class cnn:
        @staticmethod
        def build(width, height, depth, classes):
            # initialize the model
            model = Sequential()
            inputShape = (height, width, depth)

            # first set of CONV => RELU => POOL layers
            model.add(Conv2D(20, (5, 5), padding="same",
                             input_shape=inputShape))

            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            # second set of CONV => RELU => POOL layers
            model.add(Conv2D(50, (5, 5), padding="same"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # first (and only) set of FC => RELU layers

            model.add(Flatten())
            model.add(Dense(500))
            model.add(Activation("relu"))

            # softmax classifier
            model.add(Dense(classes))
            model.add(Activation("softmax"))

            # return the constructed network architecture
            return model

    # DATASET2 Contains Spectro Full
    # directory_root = sorted(list(paths.list_images('/content/drive/MyDrive/Anchana_Project (1)/Spectrogram/')))
    directory_root = sorted(
        os.listdir('D:/University stuff/Advanced Programming/Dataset/original dataset/train/normal/'))

    data = []
    cnt = 0
    labels = []
    lbl = ['1', '2', '3', '4', '5']
    ind = 0
    label = 0
    print(len(directory_root))
    # loop over the input images
    for imagePath in directory_root:
    # load the image, pre-process it, and store it in the data list
        image = cv2.imread(
            'D:/University stuff/Advanced Programming/Dataset/original dataset/train/normal/'+imagePath,0)
    cnt = cnt + 1
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    # label = imagePath.split(os.path.sep)[-2]
    # ind=lbl.find(label)
    '''for ll in range(len(lbl)):
        #print(lbl[ll],end=',')
        if label == lbl[ll] :
            ind=ll+1'''

    labels.append(ind)
    # print(str(cnt)+'/'+str(len(directory_root))+'    '+str(ind)+'  '+imagePath)
    print('Abnormal data', cnt)
    directory_root = sorted(
        os.listdir('D:/University stuff/Advanced Programming/Dataset/original dataset/train/abnormal/'))

    print(len(directory_root))
    cnt = 0
    ind = 1
    label = 1
    # loop over the input images
    for imagePath in directory_root:
    # load the image, pre-process it, and store it in the data list
        image = cv2.imread(
            'D:/University stuff/Advanced Programming/Dataset/original dataset/train/abnormal/'+imagePath,0)
    cnt = cnt + 1
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    # label = imagePath.split(os.path.sep)[-2]
    # ind=lbl.find(label)
    '''for ll in range(len(lbl)):
        #print(lbl[ll],end=',')
        if label == lbl[ll] :
            ind=ll+1'''

    labels.append(ind)
    # print(str(cnt)+'/'+str(len(directory_root))+'    '+str(ind)+'  '+imagePath)
    print('Total data', cnt)
    import pickle
    f = open('D:/University stuff/Advanced Programming/Dataset/cnn_feature.pkl', 'wb')
    pickle.dump(data, f)
    f.close()
    f = open('D:/University stuff/Advanced Programming/Dataset/cnn_label.pkl', 'wb')
    pickle.dump(labels, f)
    f.close()
    # print(np.unique(data))

    labels = np.array(labels)
    print(labels.shape)

    data = np.array(data)
    print(data.shape)

    # initialize the model
    print(np.unique(labels))
    print(np.array(data).shape)
    print(np.array(labels).shape)
    (trainX, Val_X, trainY, Val_Y) = train_test_split(np.array(data), np.array(labels), test_size=0.15, random_state=42)
    (trainX, Val_X, trainY, Val_Y) = train_test_split(trainX, trainY, test_size=0.15, random_state=42)
    (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.15, random_state=42)
    #
    # trainX=tf.expand_dims(trainX, axis=1)
    # Val_X=tf.expand_dims(Val_X, axis=1)
    print(np.unique(trainY))
    print(trainX.shape)
    print(trainY.shape)
    # print(Val_X.shape)
    # print(Val_Y.shape)
    trainY = to_categorical(trainY, num_classes=2)
    Val_Y = to_categorical(Val_Y, num_classes=2)
    # testY = to_categorical(testY, num_classes=2)
    # construct the image generator for data augmentation
    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,	horizontal_flip=True, fill_mode="nearest")
    EPOCHS = 100
    INIT_LR = 0.001
    BS = 128
    print("[INFO] compiling model...")
    model = cnn.build(width=28, height=28, depth=1, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    # H = model.fit(data,labels, steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)

    H = model.fit(trainX, trainY, validation_data=(Val_X, Val_Y), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS,
                  verbose=1)

    model.save('D:/University stuff/Advanced Programming/Dataset/cnn_model.h5')

    if denoised_nlmeans is not None:

        # Convert the loaded image to grayscale
        gray_image = cv2.cvtColor(denoised_nlmeans, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image to (28, 28)
        resized_image = cv2.resize(gray_image, (28, 28))

        # Normalize the image pixel values to the range [0, 1]
        normalized_image = resized_image / 255.0

        # Expand dimensions to match the expected input shape (None, 28, 28, 1)
        input_image = np.expand_dims(normalized_image, axis=-1)
        # image = cv2.resize(image, (28, 28))

        # Make predictions using the model
        predictions = model.predict(np.array([input_image]))

        # Get the class with the highest probability as the predicted class
        predicted_class = np.argmax(predictions)

        # You may have a mapping of class indices to class labels
        # Replace 'class_mapping' with your actual mapping if needed
        class_mapping = {0: 'Class_0', 1: 'Class_1'}  # Adjust as per your dataset

        # Get the predicted class label
        prediction_type = class_mapping[predicted_class]

        if prediction_type == 'Class_0':
            messagebox.showinfo('Result', 'Abnormal image')
        elif prediction_type == 'Class_1':
            messagebox.showinfo('Result', 'Normal image')

    else:
        messagebox.showwarning('Warning', 'Features not computed')

    # Print the predicted class and its probability
    print(f'Predicted Class: {prediction_type}')
    print(f'Probability: {predictions[0][predicted_class]}')


    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    hist_json_file = model_name + '_history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = model_name + '_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


    color = sns.color_palette()
    if (isinception == False):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

    epochs = range(len(acc))

    sns.lineplot(epochs, acc, label='Training Accuracy')
    sns.lineplot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    # plt.legend()
    # plt.figure()
    # plt.show()


    color = sns.color_palette()
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    sns.lineplot(epochs, loss, label='Training Loss')
    sns.lineplot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # plt.figure()
    plt.show()
    plt.savefig('D:/University stuff/Advanced Programming/Dataset/cnnlosscurve.png')


    save_history(history, history_file_name)
    plot_accuracy_from_history(history, isinception)
    plot_loss_from_history(history)

    import pandas as pd
    import seaborn as sns
    filename = 'D:/University stuff/Advanced Programming/Dataset/cnnhistory1.sav'
    pickle.dump(H, open(filename, 'wb'))
    do_history_stuff(H, filename, isinception=False)
    # plt.show()

    predictions = model.predict(testX)
    pred = np.argmax(predictions, axis=1)
    print(pred)

    # testY1=np.argmax(testY,axis=1)
    print(testY)

    err = np.sum(abs(pred - testY)) / len(pred)


    print(1 - err)

    from sklearn import metrics


    confusion_matrix = metrics.confusion_matrix(np.array(pred), np.array(testY))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    import seaborn as sn
    cmap = "tab20"
    print(np.array(confusion_matrix))
    hm = sn.heatmap(np.array(confusion_matrix), cmap=cmap)
    # displaying the plotted heatmap
    plt.show()
    plt.figure(figsize=(10, 10))
    ax = sn.heatmap(confusion_matrix, annot=True, fmt="f")

    cnf = confusion_matrix
    TP = cnf[0, 0]
    FN = cnf[0, 1]
    FP = cnf[1, 0]
    TN = cnf[1, 1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensisitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    print('Accuracy (in %) :', round(accuracy * 100, 2))
    print('Sensisitivity (in %) :', round(Sensisitivity * 100, 2))
    print('Specificity (in %) :', round(Specificity * 100, 2))


    # # Load your pre-trained CNN model (replace 'your_model.h5' with your model's path)
    # model = load_model('D:/University stuff/Advanced Programming/Dataset/Models/cnn_model.h5')



import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer, Table, TableStyle, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def generate_report(patient_details, prediction):
    # Define a file name for the PDF report
    report_file_name = "macular_edema_report.pdf"


    # Define a list to store report elements
    elements = []


    # Create a story (collection of elements) for the frame
    story = []

    # Create a stylesheet for the report
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    # Create a heading paragraph
    heading_text = "Macular Edema Diagnosis Report"
    heading_style = styles["Heading1"]  # You can choose a suitable heading style
    heading_paragraph = Paragraph(heading_text, heading_style)

    # Add the heading paragraph to the elements list
    elements.append(heading_paragraph)


    # Add patient details to the report
    patient_info = [
        ["Patient Details"],
        ["Name:", patient_details["name"]],
        ["Age:", patient_details["age"]],
        ["Patient ID:", patient_details["patient_id"]],
    ]

    patient_info_table = Table(patient_info)
    patient_info_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                            ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    elements.append(Spacer(1, 12))
    elements.append(patient_info_table)
    elements.append(Spacer(1, 12))

    # Add prediction result to the report
    if prediction == 0 or prediction == 'class_0':
        prediction_text = "Prediction: Abnormal image"
    elif prediction == 1 or prediction == 'class_1':
        prediction_text = "Prediction: Normal image"
    else:
        prediction_text = f"Prediction: {prediction}"


    prediction_paragraph = Paragraph(prediction_text, normal_style)
    elements.append(prediction_paragraph)

    text1 = "This report contains information regarding the patient mentioned above."
    text2 = "Macular edema is a medical condition characterized by the accumulation of fluid in the macula, the central part of the retina, leading to distorted and blurred vision. \n It is often associated with conditions like diabetic retinopathy and age-related macular degeneration, and timely treatment is crucial to prevent further vision deterioration."


    paragraph1 = Paragraph(text1, normal_style)
    paragraph2 = Paragraph(text2, normal_style)

    elements.append(paragraph1)
    elements.append(paragraph2)

    # # Build the PDF document
    #
    # doc.build(elements)

    # Check if the report was created successfully
    if os.path.exists(report_file_name):
        return report_file_name
    else:
        return None



def create_report():
    global prediction_type
    # Get patient details from the GUI
    patient_details = {
        "name": name_entry.get(),
        "age": age_entry.get(),
        "patient_id": patient_id_entry.get(),
    }

    # Replace with your actual prediction logic
    prediction = prediction_type  # Example prediction, replace with the actual prediction result

    # Generate the PDF report
    report_file = generate_report(patient_details, prediction)

    if report_file:
        messagebox.showinfo("Report Created", f"The PDF report '{report_file}' has been created successfully.")
    else:
        messagebox.showerror("Report Error", "An error occurred while creating the PDF report.")

import_tab = tk.Frame(root)

# Set a background color
root.configure(bg='#e6e6e6')

# Create a custom font for the title
title_font = tkFont.Font(family='Helvetica', size=16, weight='bold')

# Create a tabbed widget
tab_control = ttk.Notebook(root)
patient_info_frame = ttk.Frame(tab_control)
diagnosis_frame = ttk.Frame(tab_control)
DeepLearning_frame = ttk.Frame(tab_control)
about_tab_frame = ttk.Frame(tab_control)

tab_control.add(patient_info_frame, text='Patient Info')
tab_control.add(diagnosis_frame, text='Machine Learning')
tab_control.add(DeepLearning_frame, text='Deep Learning Model')
tab_control.add(about_tab_frame, text="About")
tab_control.pack(fill='both', expand=True)

# Create and place the heading for patient information
patient_info_heading = tk.Label(patient_info_frame, text="MACULAR EDEMA DIAGNOSTIC SOFTWARE", font=("Helvetica", 35, "bold"))
patient_info_heading.place(x=450, y=50)

patient_info_heading = tk.Label(patient_info_frame, text="Patient Information", font=("Helvetica", 20, "bold"))
patient_info_heading.place(x=800, y=200)

diagnosis_heading = tk.Label(diagnosis_frame, text="Machine Learning Models", font=("Helvetica", 20, "bold"))
diagnosis_heading.place(x=800, y=200)

DeepLearning_heading = tk.Label(DeepLearning_frame, text="Deep Learning Model", font=("Helvetica", 20, "bold"))
DeepLearning_heading.place(x=800, y=200)


# Load the logo image
logo_image = Image.open('anhalt logo.png')

# Convert it to a PhotoImage object
logo_photo = ImageTk.PhotoImage(logo_image)

logo_label = tk.Label(patient_info_frame, image=logo_photo)
logo_label.place(x=10, y=10)  # Adjust the coordinates as needed

logo_label.photo = logo_photo

def update_time_date():
    current_time = time.strftime("%H:%M:%S")
    current_date = datetime.date.today().strftime("%d-%m-%Y")
    time_date_label.config(text=f"Time: {current_time}\nDate: {current_date}")
    # Update every 1000 milliseconds (1 second)
    root.after(1000, update_time_date)

# Add global variables to keep track of the current tab index
current_tab_index = 0

# Function to go to the next tab
def next_tab():
    global current_tab_index
    current_tab_index = (current_tab_index + 1) % tab_control.index("end")
    tab_control.select(current_tab_index)

# Function to go to the previous tab
def prev_tab():
    global current_tab_index
    current_tab_index = (current_tab_index - 1) % tab_control.index("end")
    tab_control.select(current_tab_index)

# Create a label for time and date
time_date_label = tk.Label(root, text="", font=("Helvetica", 12))
time_date_label.place(x=1700, y=20)

# Call the function to start updating time and date
update_time_date()

# Create and place labels and entry fields for patient information in the Patient Info tab
name_label = tk.Label(patient_info_frame, text='Name:', bg='#e6e6e6', font=("Helvetica", 12, "bold"))
name_label.place(x=700, y=250)

name_entry = tk.Entry(patient_info_frame, width= 40, bd=3)
name_entry.place(x=800, y=250)

age_label = tk.Label(patient_info_frame, text='Age:', bg='#e6e6e6', font=("Helvetica", 12, "bold"))
age_label.place(x=700, y=300)

age_entry = tk.Entry(patient_info_frame, width= 40, bd=3)
age_entry.place(x=800, y=300)

patient_id_label = tk.Label(patient_info_frame, text='Patient ID:', bg='#e6e6e6', font=("Helvetica", 12, "bold"))
patient_id_label.place(x=700, y=350)

patient_id_entry = tk.Entry(patient_info_frame, width= 40, bd=3)
patient_id_entry.place(x=800, y=350)

# Create the 'Import Image' button in the Patient Info tab
import_image_button = tk.Button(patient_info_frame, text='Import Image', command=import_, bg='#2196F3', fg='white', font=("Helvetica", 12, "bold"))
import_image_button.place(x=850, y=450)

# Create the 'Start Diagnosis' button in the Patient Info tab
start_diagnosis_button = tk.Button(patient_info_frame, text='Start Diagnosis', command=start_diagnosis, bg='#4caf50', fg='white', font=("Helvetica", 12, "bold"))
start_diagnosis_button.place(x=845, y=500)


# Create model selection buttons in the Diagnosis tab, grouped into 2 sets
model_group1_label = tk.Label(diagnosis_frame, text='HOG feature extraction', font=('Helvetica', 14, "bold"))
model_group1_label.place(x=700, y=250)

model_button1 = tk.Button(diagnosis_frame, text='SVM-HOG model', command=hog_svm, font=("Helvetica", 12, "bold"), bg='#1C86EE', fg='white')
model_button1.place(x=750, y=300)
model_button2 = tk.Button(diagnosis_frame, text='KNN-HOG model', command=hog_knn, font=("Helvetica", 12, "bold"), bg='#1C86EE', fg='white')
model_button2.place(x=750, y=400)
model_button3 = tk.Button(diagnosis_frame, text='NB-HOG model', command=hog_nb, font=("Helvetica", 12, "bold"), bg='#1C86EE', fg='white')
model_button3.place(x=750, y=500)

model_group2_label = tk.Label(diagnosis_frame, text='LBP feature extraction', font=('Helvetica', 14, "bold"))
model_group2_label.place(x=1100, y=250)

model_button4 = tk.Button(diagnosis_frame, text='SVM-LBP model', command=lbp_svm, font=("Helvetica", 12, "bold"), bg='#66CD00', fg='white')
model_button4.place(x=1150, y=300)
model_button5 = tk.Button(diagnosis_frame, text='KNN-LBP model', command=lbp_knn, font=("Helvetica", 12, "bold"), bg='#66CD00', fg='white')
model_button5.place(x=1150, y=400)
model_button6 = tk.Button(diagnosis_frame, text='NB-LBP model', command=lbp_nb, font=("Helvetica", 12, "bold"), bg='#66CD00', fg='white')
model_button6.place(x=1150, y=500)


# create filter button in the DeepLearning tab

filter_button4 = tk.Button(DeepLearning_frame, text='Convolutional Neural Network (CNN)', command=cnn, font=("Helvetica", 12, "bold"), bg='#66CD00', fg='white')
filter_button4.place(x=800, y=300)

# # Create a 'Next' button to go to the model selection page
# next_button = tk.Button(diagnosis_frame, text='Next', command=go_back, bg='#2196F3', fg='white')
# next_button.place(x=1500, y=500)

model_group3_label = tk.Label(about_tab_frame, text='Macular Edema Diagnostic Software. \n Macular edema is a condition characterized by the swelling of the macula, \n the central part of the retina responsible for sharp, central vision. \n It can be caused by various factors, such as diabetes or age-related \n macular degeneration. \n The prediction is done by machine learning algorithms, and users have freedom to pick the combination of algorithm for detection of the disease.  \n \n This project is developed by the Theres Jose, Pravin Maske, Ronal Roy, Aparna Rajamohanan and Nibin Noushad, \n under the guidance of Prof. Dr. Marianne Maktabi.', font=('Helvetica', 14, "bold"))
model_group3_label.place(x=250, y=150)

# Create "Next" and "Previous" buttons and place them on each tab
next_button = tk.Button(patient_info_frame, text='Next', command=next_tab, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
next_button.place(x=1500, y=700)

next_button = tk.Button(diagnosis_frame, text='Next', command=next_tab, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
next_button.place(x=1500, y=700)

next_button = tk.Button(DeepLearning_frame, text='Next', command=next_tab, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
next_button.place(x=1500, y=700)

prev_button = tk.Button(diagnosis_frame, text='Previous', command=prev_tab, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
prev_button.place(x=500, y=700)

prev_button = tk.Button(DeepLearning_frame, text='Previous', command=prev_tab, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
prev_button.place(x=500, y=700)

#create 'create report' button
report_button = tk.Button(diagnosis_frame, text='Create Report', command=create_report, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
report_button.place(x=1000, y=700)

report_button = tk.Button(DeepLearning_frame, text='Create Report', command=create_report, bg='#CD5B45', fg='white', font=("Helvetica", 12, "bold"))
report_button.place(x=1000, y=700)


# Hide the Diagnosis frame initially
diagnosis_frame.pack_forget()

# Start the Tkinter main loop
root.mainloop()
