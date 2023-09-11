import cv2
import datetime
import tkinter as tk
from tkinter import messagebox
import face_recognition
import os

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def mark_detected_face(camera_view, x_cord, y_cord, width_of_object, height_of_object):
    """
    Marks detected face by blue frame and saves it temporarily as PNG image.

    Args:
        camera_view: The image to mark the face on.
        x_cord: The x-coordinate of the top-left corner of the face.
        y_cord: The y-coordinate of the top-left corner of the face.
        width_of_object: The width of the face.
        height_of_object: The height of the face.
    Returns:
        None
    """
    cv2.rectangle(camera_view, (x_cord - 100, y_cord - 100), (x_cord + width_of_object + 100,
                                                              y_cord + height_of_object + 100), (255, 0, 0), 2)
    try:
        cv2.imwrite('faces\\' + 'temp.png', camera_view[y_cord - 100:y_cord + width_of_object + 100,
                                                        x_cord - 100:x_cord + height_of_object + 100])
        return True
    except cv2.error:
        show_info_box('Caution!', 'To ensure a correct detection, please keep your face in the centre of view. '
                                  'Please keep your face as straight as possible.')
        return False


def compare_faces_from_images(image_path_1, image_path_2):
    """
    Loads two images from file to create face encodings and then make comparison.

    Args:
        image_path_1: Path to the first image.
        image_path_2: Path to the second image.

    Returns:
        True when faces are similar, False in opposite situation.
    """
    image_1 = face_recognition.load_image_file(image_path_1)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_1_face_encodings = face_recognition.face_encodings(image_1)[0]
    image_2 = face_recognition.load_image_file(image_path_2)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_2_face_encodings = face_recognition.face_encodings(image_2)[0]

    return face_recognition.compare_faces([image_1_face_encodings], image_2_face_encodings)


def show_comparison_window(window_title, info_text, similar_face_path):
    """
    Show a comparison window to the user with information about successful comparison.

    Args:
        window_title: The title of the window.
        info_text: The text to be displayed in the input box.
        similar_face_path: The path to compared face that is similar.

    Returns:
        None
    """
    # Create the input box
    comparison_window = tk.Tk()
    comparison_window.title(window_title)
    comparison_window_label = tk.Label(comparison_window, text=info_text)
    face_image = tk.PhotoImage(file='faces\\temp.png')
    face_image_label = tk.Label(comparison_window, image=face_image)
    similar_face_image = tk.PhotoImage(file=similar_face_path)
    similar_face_image_label = tk.Label(comparison_window, image=similar_face_image)

    # Place the widgets on the screen
    face_image_label.pack()
    similar_face_image_label.pack()
    comparison_window_label.pack()

    # Start the mainloop
    comparison_window.attributes('-topmost', True)
    comparison_window.mainloop()


def show_input_box(window_title, label_text):
    """
    Show an input box to the user and return the entered text.

    Args:
        window_title: The title of the window.
        label_text: The text to be displayed in the input box.

    Returns:
        The text entered by the user.
    """
    # Create the input box
    input_box = tk.Tk()
    input_entry_text = tk.StringVar()
    input_box.title(window_title)
    input_label = tk.Label(input_box, text=label_text)
    input_entry = tk.Entry(input_box, textvariable=input_entry_text)
    submit_button = tk.Button(input_box, text="Submit", command=input_box.destroy)
    face_image = tk.PhotoImage(file='faces\\temp.png')
    label_image = tk.Label(input_box, image=face_image)

    # Place the widgets on the screen
    label_image.pack()
    input_label.pack()
    input_entry.pack()
    submit_button.pack()

    # Start the mainloop
    input_box.attributes('-topmost', True)
    input_box.mainloop()

    return input_entry_text.get()


def show_info_box(window_title, info_text):
    """
    Show a short information in Tkinter window.

    Args:
        window_title: The title of the window.
        info_text: The text to be displayed in the input box.

    Returns:
        None
    """
    # Create the info box
    info_box = messagebox.showinfo(window_title, info_text)


def main():
    # initiate video capture by camera with index 0
    cam = cv2.VideoCapture(0)

    # keep detecting faces until close
    while True:
        ret, img = cam.read()
        faces = face_classifier.detectMultiScale(
            img,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            if not mark_detected_face(img, x, y, w, h):
                break

            faces_paths = ['faces\\' + path for path in os.listdir('faces')]
            is_face_similar = False

            for face_path in faces_paths:
                if face_path != 'faces\\temp.png':
                    try:
                        if compare_faces_from_images(face_path, 'faces\\temp.png'):
                            path_segments = face_path.split('\\')
                            path_segments = path_segments[1].split('_')
                            is_face_similar = True
                            show_comparison_window('FaceComparison', "These images are similar. "
                                                                     "I know you! You're " +
                                                   path_segments[0], face_path)
                    except IndexError:
                        print("There's no face on image:", face_path)

            if not is_face_similar:
                person_name = show_input_box('FaceDetection', "I don't know you.\nWhat's your name?\n"
                                                              "(to skip just leave blank field)")
                if person_name != '':
                    cv2.imwrite('faces\\' + person_name + '_' + datetime.datetime.now().strftime("%B%d%Y%H%M%S") +
                                '.png', img[y - 100:y + h + 100, x - 100:x + w + 100])

        # shows camera image
        cv2.imshow('FaceDetector', img)
        # camera image freeze
        cv2.waitKey(30) & 0xff

        # if camera window is closed
        if cv2.getWindowProperty('FaceDetector', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
