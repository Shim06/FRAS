import customtkinter
import cv2
from datetime import datetime, date, timedelta, time as dt_time
from deepface import DeepFace
import glob 
import tkinter
import tkinter.messagebox
import face_recognition
from multiprocessing import freeze_support
import os, sys
import math
import numpy as np
from pyautogui import size
import sqlite3
import threading
from nameparser import HumanName

def face_confidence(face_distance, face_match_treshold=0.6):
    range = (1.0 - face_match_treshold)
    linear_value = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_treshold:
        return str(round(linear_value * 100, 2 )) + "%"
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Face Recognition Software")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Face Recognition", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.start_button = customtkinter.CTkButton(self.sidebar_frame, command=self.start_button_event,
                                                        text="Start")
        self.start_button.grid(row=1, column=0, padx=20, pady=10)
        self.stop_button = customtkinter.CTkButton(self.sidebar_frame, command=self.stop_button_event,
                                                        text="Stop")
        self.stop_button.grid(row=2, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.frame.grid(row=0, column=1, rowspan=6, sticky="nsew")
        self.frame.grid_rowconfigure(6, weight=1)
        self.name = customtkinter.CTkLabel(self.frame, text="Name:", font=customtkinter.CTkFont(size=30, weight="bold"))
        self.name.grid(row=0, column=0, padx=20, pady=(20, 0))
        self.student_name = customtkinter.CTkLabel(self.frame, text="", font=customtkinter.CTkFont(size=30, weight="bold"))
        self.student_name.grid(row=1, column=0, padx=20, pady=(0, 0))
        self.time_in = customtkinter.CTkLabel(self.frame, text="Time In:", font=customtkinter.CTkFont(size=40, weight="bold"))
        self.time_in.grid(row=2, column=0, padx=20, pady=(20, 0))
        self.time_in_time = customtkinter.CTkLabel(self.frame, text="", font=customtkinter.CTkFont(size=40, weight="bold"))
        self.time_in_time.grid(row=3, column=0, padx=20, pady=(0, 0))
        self.time_out = customtkinter.CTkLabel(self.frame, text="Time Out:", font=customtkinter.CTkFont(size=40, weight="bold"))
        self.time_out.grid(row=4, column=0, padx=20, pady=(20, 0))
        self.time_out_time = customtkinter.CTkLabel(self.frame, text="", font=customtkinter.CTkFont(size=40, weight="bold"))
        self.time_out_time.grid(row=5, column=0, padx=20, pady=(0, 0))


        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.stop_button.configure(state="disabled")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def start_button_event(self):
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=fr.run_recognition).start()

    def stop_button_event(self):
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        fr.stop()
        

    def sidebar_button_event(self):
        pass

    def update_attendace(self, student_id):
        database = sqlite3.connect("database.db", isolation_level=None)
        sql = database.cursor()
        t = datetime.now()
        student_name = sql.execute('SELECT FirstName, MiddleInitial, Lastname FROM students WHERE ID = ?', [student_id])
        for name in student_name:
            full_name = "".join(list(name))

        self.student_name.configure(text=full_name)
        Timein = sql.execute("SELECT TimeIn FROM attendance WHERE ID = ? AND Year = ? AND Month = ? AND Day = ?",
                            [student_id,
                            t.strftime("%Y"),
                            t.strftime("%m"),
                            t.strftime("%d")]).fetchall()
        if Timein and any(timee[0] is not None for timee in Timein):
            in_time = ""
            for timee in Timein:
                in_time = "".join(timee)
            self.time_in_time.configure(text=in_time)

        Timeout = sql.execute("SELECT TimeOut FROM attendance WHERE ID = ? AND Year = ? AND Month = ? AND Day = ?",
                            [student_id,
                            t.strftime("%Y"),
                            t.strftime("%m"),
                            t.strftime("%d")]).fetchall()
        if Timeout and any(timee[0] is not None for timee in Timeout):
            out_time = ""   
            for timee in Timeout:
                    out_time = "".join(timee)
            self.time_out_time.configure(text=out_time)

        database.close()

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    found_students = []
    run = False

    def __init__(self):
        # Encode faces
        self.encode_faces()
    
    def encode_faces(self):
        for image in os.listdir("faces"):
            if image.endswith(".png"): 
                print(image)
                face_image = face_recognition.load_image_file(f"faces/{image}")
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)

        print(self.known_face_names)

    def find_faces(self, frame):

        # Find all the faces in the current frame
        self.face_locations = face_recognition.face_locations(frame, model="hog")
        self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match = np.argmin(face_distances)

            if matches[best_match]:
                name = self.known_face_names[best_match]
                confidence = face_confidence(face_distances[best_match])

            self.found_students.append(name)
            self.face_names.append(f"{name} {confidence}")

    def run_recognition(self):
        self.run = True
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Video source not found")

        screen_size = size()

        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition", int(screen_size[0] / 2), int(screen_size[1]))

        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.run:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                self.find_faces(rgb_small_frame)

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            if self.found_students != None:
                database2 = sqlite3.connect("database.db", isolation_level=None)
                sql2 = database2.cursor()
                for student in self.found_students:
                    t = datetime.now()
                    student = student.replace(",", "")
                    student = student.removesuffix(".png")
                    student = student.removesuffix(".jpg")
                    student = student.split(" ")
                    student.append(student.pop(0))

                    student_ids = sql2.execute('SELECT ID FROM students WHERE LastName = ?', [student[-1]])

                    for id in student_ids:
                        student_id = id[0]
                        
                    time_in = sql2.execute("SELECT TimeIn FROM attendance WHERE ID = ? AND Year = ? AND Month = ? AND Day = ?",
                                          [student_id,
                                          t.strftime("%Y"),
                                          t.strftime("%m"),
                                          t.strftime("%d")])
                    time_out = sql2.execute("SELECT TimeOut FROM attendance WHERE ID = ? AND Year = ? AND Month = ? AND Day = ?",
                                           [student_id,
                                           t.strftime("%Y"),
                                           t.strftime("%m"),
                                           t.strftime("%d")])
                    time_in = time_in.fetchall()
                    time_out = time_out.fetchall()
                    if not time_in:
                        sql2.execute("INSERT INTO attendance (ID, Year, Month, Day, TimeIn) VALUES (?, ?, ?, ?, ?)",
                                    [student_id,
                                    t.strftime("%Y"),
                                    t.strftime("%m"),
                                    t.strftime("%d"),
                                    t.strftime("%H:%M:%S")])
                        database2.commit()
                    elif not time_out:
                        timein_time = 0
                        current_time = 0
                        cooldown = sql2.execute("SELECT TimeIn FROM attendance WHERE ID = ? AND Year = ? AND Month = ? AND Day = ?",
                                               [student_id,
                                               t.strftime("%Y"),
                                               t.strftime("%m"),
                                               t.strftime("%d")])
                        for time in cooldown:
                            timein_time = time[0].split(":")
                            timein_time = dt_time(int(timein_time[0]), int(timein_time[1]), int(timein_time[2]))
                            current_time = datetime.now().time()

                        timein_time = datetime.combine(date.today(), timein_time)
                        current_time = datetime.combine(date.today(), current_time)
            
                        if (current_time - timein_time) >= timedelta(minutes=10):
                            sql2.execute("UPDATE attendance SET TimeOut = ? WHERE ID = ? AND Year = ? AND Month = ? AND Day = ?",
                                        [t.strftime("%H:%M:%S"),
                                        student_id,
                                        t.strftime("%Y"),
                                        t.strftime("%m"),
                                        t.strftime("%d")])
                            database2.commit()
                        else:
                            pass
                    
                    database2.close()
                    app.update_attendace(student_id)


            self.found_students = []

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.run = False

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    # cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

    counter = 0
    face_match = False
    student_name = ""

    freeze_support()
    fr = FaceRecognition()
    app = App()
    app.mainloop()
