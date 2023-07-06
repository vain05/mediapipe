licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "face_effect_gpu",
    data = [
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_detection",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_landmarks",
        "//mediapipe/graphs/face_effect/data:axis",
        "//mediapipe/graphs/face_effect/data:glasses",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_face_effect_gpu",
        "//mediapipe/graphs/face_effect:face_effect_gpu_deps",
        # "//mediapipe/graphs/face_effect:face_effect_gpu_binary_graph",
        # "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators"
    ],
)