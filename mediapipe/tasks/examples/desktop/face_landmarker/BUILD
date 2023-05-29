licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_binary(
    name = "demo_face_landmarker",
    srcs = ["demo_face_landmarker.cc"],
    deps = [
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_opencv",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ]
)

cc_binary(
    name = "demo_face_landmarker_cpu",
    srcs = ["demo_face_landmarker_cpu.cc"],
    deps = [
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_opencv",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ]
)

cc_binary(
    name = "demo_face_landmarker_gpu",
    srcs = ["demo_face_landmarker_gpu.cc"],
    deps = [
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_opencv",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ]
)