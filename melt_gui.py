import tkinter
from tkinter import filedialog
import tkinter.messagebox
import customtkinter
import numpy as np
import os
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import shutil
import onnxruntime as rt
from threading import Thread
import sys
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
import multiprocessing
from basicsr.utils.registry import ARCH_REGISTRY

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

##################### Swapping #######################

## Full credit for FileVideoStream: https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

class FileVideoStream:
    def __init__(self, path, queueSize=2500):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self


    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def get_fps(self):
        return self.fps

## Massive shoutout to the team at Insightface for thir continued brilliant work. https://github.com/deepinsight/insightface
## Also borrowed some bits from https://github.com/neuralchen/SimSwap
## Full credit to both

def video_swap(source_img_path, video_path, frame_dir, vid_dir, filename):

    print(rt.get_device())
    assert insightface.__version__>='0.7'

    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(video_path).start()
    fps = FileVideoStream(video_path).get_fps()
    time.sleep(1.0)

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)
    else:
        os.makedirs(frame_dir)

    source_face = cv2.imread(source_img_path)
    source_face = app.get(source_face)
    source_face = sorted(source_face, key = lambda x : x.bbox[0])
    source_face = source_face[0]

    # loop over frames from the video file stream
    token = 0
    while fvs.more():

        frame = fvs.read()
        detect_results = app.get(frame)
        
        if detect_results is not None:
            token = token + 1              
            detect_results = app.get(frame)
            detect_results = sorted(detect_results, key = lambda x : x.bbox[0])
            # assert len(detect_results)==1

            res = frame.copy()
            try:
                res = swapper.get(res, detect_results[0], source_face, paste_back=True) # (height, width, channels)
                cv2.imwrite(f"{frame_dir}/{token}_swapped.png", res)
                print(f"{frame_dir}/{token}_swapped.png")
            except:
                frame = frame.astype(np.uint8)
                cv2.imwrite(f"{frame_dir}/{token}_swapped.png", frame)
                print(f"{frame_dir}/{token}_swapped.png")

    path = os.path.join(frame_dir,'*.png')
    image_filenames = sorted(glob.glob(path), key=len)

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    clips.write_videofile(vid_dir+f"/{filename.rsplit('.', 1)[0]}.mp4",audio_codec='aac')


##################### GUI ################################

## Below UI is essentially a bastardized version of: https://github.com/TomSchimansky/CustomTkinter/blob/master/examples/complex_example.py

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Melt")
        self.geometry("1100x700")
        self.resizable(width=0, height=0)

        ########## Default Values ##########
        self.queue_array = []
        self.OUT_VID = './output_videos/'
        self.TARGET_VID = './target_videos/'
        self.OUT_FRAMES = './output_frames/'
        self.SOURCE_IMAGES = './source_images/'

        # configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # create sidebar frame with widgets

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Melt", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="melt_contact@proton.me", anchor="w")
        self.appearance_mode_label.grid(row=2, column=0, padx=20, pady=(20, 20))

        self.clear_queue_button = customtkinter.CTkButton(self.sidebar_frame, fg_color="transparent", text="Clear Queue", text_color=("#DCE4EE"), hover_color=("#E12901"), command=self.clear_queue)
        self.clear_queue_button.grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.frame_dir_button = customtkinter.CTkButton(self.sidebar_frame, text="Open Frame Directory", command=self.open_frame_dir)
        self.frame_dir_button.grid(row=7, column=0, padx=20, pady=10)

        self.start_swap = customtkinter.CTkButton(self.sidebar_frame, text="Start Faceswap", command=self.swap_queue)
        self.start_swap.grid(row=8, column=0, padx=20, pady=10)


        # create main entry and button

        self.select_target = customtkinter.CTkEntry(self, placeholder_text="Select Target Video")
        self.select_target.grid(row=2, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.select_button_2 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text="Select", text_color=("gray10", "#DCE4EE"), command=self.browse_target_file)
        self.select_button_2.grid(row=2, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.select_source = customtkinter.CTkEntry(self, placeholder_text="Select Source Photo")
        self.select_source.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.select_button_3 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text="Select", text_color=("gray10", "#DCE4EE"), command=self.browse_source_file)
        self.select_button_3.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # text box for link queue

        self.textbox = customtkinter.CTkTextbox(self, width=350, height=500)
        self.textbox.grid(row=0, column=1, columnspan=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.textbox.insert("0.0", "Target Video Queue:\n\n")
   
    
    def clear_queue(self):

        self.queue_array.clear()
        self.textbox.delete("1.0", "end")


    def browse_source_file(self):
        
        source_filepath = filedialog.askopenfilename(initialdir=self.SOURCE_IMAGES)
        self.select_source.delete(0, customtkinter.END)
        self.select_source.insert(0, source_filepath)

    def open_frame_dir(self):

        os.system(f"start {os.path.abspath(os.getcwd() + self.OUT_FRAMES)}")

    def browse_target_file(self):

        target_filepath = filedialog.askopenfilename(initialdir=self.TARGET_VID)
        # filename = target_filepath.rsplit('/', 1)[1]
        self.select_target.delete(0, customtkinter.END)
        self.select_target.insert(0, target_filepath)
        self.queue_array.append(target_filepath)
        self.textbox.insert('end', self.queue_array[-1] + '\n')

    def swap_queue(self):
        
        for count, path in enumerate(self.queue_array):
            out_frames = self.OUT_FRAMES + f"{count}"
            queue_filename = path.rsplit('/', 1)[1]
            source_filepath = self.select_source.get()
            new_process = multiprocessing.Process(target=video_swap, args = [source_filepath, path, out_frames, self.OUT_VID, queue_filename])
            new_process.daemon = True
            new_process.start()
            self.clear_queue()



if __name__ == "__main__":
    app = App()
    app.mainloop()