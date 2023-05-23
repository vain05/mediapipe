licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "demo_face_landmarker",
    srcs = ["demo_face_landmarker.cc"],
    # data = ["//mediapipe/tasks/examples/desktop/face_landmarker/assets/face_landmarker.task"],
    deps = [
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:image_opencv",
        # "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        # "//mediapipe/tasks/cc/vision/core:image_processing_options",
        # "//mediapipe/tasks/cc/vision/core:running_mode",
        # "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_blendshapes_graph",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ]
)