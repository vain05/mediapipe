#include <cstdlib>
#include <map>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

constexpr char kWindowName[] = "Face Landmarker";

static constexpr std::array<absl::string_view, 52> kBlendshapeNames = {
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight"
};

static constexpr std::map<std::string, int> kBlendshapesNamesToIndex = {
   { "_neutral" = 0 },
   { "browDownLeft" = 1},
   { "browDownRight" = 2},
   { "browInnerUp" = 3},
   { "browOuterUpLeft" = 4},
   { "browOuterUpRight" = 5},
   { "cheekPuff" = 6},
   { "cheekSquintLeft" = 7},
   { "cheekSquintRight" = 8},
   { "eyeBlinkLeft" = 9},
   { "eyeBlinkRight" = 10},
   { "eyeLookDownLeft" = 11},
   { "eyeLookDownRight" = 12},
   { "eyeLookInLeft" = 13},
   { "eyeLookInRight" = 14},
   { "eyeLookOutLeft" = 15},
   { "eyeLookOutRight" = 16},
   { "eyeLookUpLeft" = 17},
   { "eyeLookUpRight" = 18},
   { "eyeSquintLeft" = 19},
   { "eyeSquintRight" = 20},
   { "eyeWideLeft" = 21},
   { "eyeWideRight" = 22},
   { "jawForward" = 23},
   { "jawLeft" = 24},
   { "jawOpen" = 25},
   { "jawRight" = 26},
   { "mouthClose" = 27},
   { "mouthDimpleLeft" = 28},
   { "mouthDimpleRight" = 29},
   { "mouthFrownLeft" = 30},
   { "mouthFrownRight" = 31},
   { "mouthFunnel" = 32},
   { "mouthLeft" = 33},
   { "mouthLowerDownLeft" = 34},
   { "mouthLowerDownRight" = 35},
   { "mouthPressLeft" = 36},
   { "mouthPressRight" = 37},
   { "mouthPucker" = 38},
   { "mouthRight" = 39},
   { "mouthRollLower" = 40},
   { "mouthRollUpper" = 41},
   { "mouthShrugLower" = 42},
   { "mouthShrugUpper" = 43},
   { "mouthSmileLeft" = 44},
   { "mouthSmileRight" = 45},
   { "mouthStretchLeft" = 46},
   { "mouthStretchRight" = 47},
   { "mouthUpperUpLeft" = 48},
   { "mouthUpperUpRight" = 49},
   { "noseSneerLeft" = 50},
   { "noseSneerRight" = 51}
}


ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

ABSL_FLAG(std::vector<std::string>, blendshapes, "",
          "List of the indices of blendshapes to show its prediction score. "
          "If not provided, show nothing.")

absl::Status RunFaceLandmarker()
{
    LOG(INFO) << "Initialize Face Landmarker options.";
    std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>
        options(new mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions());
    options->base_options.model_asset_path = "./mediapipe/tasks/examples/desktop/face_landmarker/assets/face_landmarker.task";
    options->running_mode = mediapipe::tasks::vision::core::RunningMode::VIDEO;
    options->output_face_blendshapes = true;

    LOG(INFO) << "Initialize Face Landmarker.";
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarker>>
        face_landmarker = mediapipe::tasks::vision::face_landmarker::FaceLandmarker::Create(std::move(options));

    LOG(INFO) << "Initialize the camera.";
    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video)
    {
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    }
    else
    {
        capture.open(0);
    }
    RET_CHECK(capture.isOpened());
    
    // cv::VideoWriter writer;
    const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
    if (!save_video)
    {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_FPS, 30);
#endif
    }

    LOG(INFO) << "Start grabbing and processing frames.";
    bool grab_frames = true;
    while (grab_frames)
    {
        size_t start = cv::getTickCount();

        // Capture opencv camera.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty())
        {
            if (!load_video)
            {
                LOG(INFO) << "Ignore empty frames from camera.";
                break;
            }

            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        if (!load_video)
        {
            cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
        }

        // Wrap Mat into an ImageFrame
        auto input_frame = std::make_shared<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary
        );
        auto input_image = std::make_shared<mediapipe::Image>(input_frame);

        std::shared_ptr<cv::Mat> input_image_mat = mediapipe::formats::MatView(input_image.get());
        camera_frame.copyTo(*input_image_mat);

        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult
            result = *(*face_landmarker)->DetectForVideo(*input_image, frame_timestamp_us);

        // show face landmarks
        bool show_landmarks = false;
        if (show_landmarks && !result.face_landmarks.empty())
        {
            int i = 0;
            // landmarks of each face
            for (auto &face_landmark : result.face_landmarks)
            {
                for (auto &landmark : face_landmark.landmarks)
                {
                    LOG(INFO) << "Landmark: " << i++;
                    LOG(INFO) << "x coordinate: " << landmark.x; 
                    LOG(INFO) << "y coordinate: " << landmark.y; 
                    LOG(INFO) << "z coordinate: " << landmark.z;
                }
            }
        }

        // show face blendshapes
        bool show_blendshapes = true;
        auto face_blendshapes = *std::move(result.face_blendshapes);
        float jaw_open_score = 0;
        if (show_blendshapes && !face_blendshapes.empty())
        {
            // blendshapes of each face
            for (auto &blendshape : face_blendshapes)
            {
                for (auto &category : blendshape.categories)
                {
                    // only show index = 25 which is jawOpen because it easy to see, 
                    // to show whether user's jaw is open (a float value between 0 and 1)
                    if (category.index == 25)
                    {
                        jaw_open_score = category.score;
                        // LOG(INFO) << kBlendshapeNames[category.index] << ' ' << category.score;
                    }
                } 
            }
        }

        int fps = cv::getTickFrequency() / (cv::getTickCount() - start);

        cv::putText(camera_frame_raw, std::string("FPS: ") + std::to_string(fps), 
                    cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(100, 255, 0), 2, cv::LINE_AA);

        cv::putText(camera_frame_raw, "Jaw Open Score",
                    cv::Point(560, 15), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(100, 255, 0), 2, cv::LINE_AA);

        cv::putText(camera_frame_raw, std::to_string(jaw_open_score),
                    cv::Point(560, 35), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(100, 255, 0), 2, cv::LINE_AA);

        cv::imshow(kWindowName, camera_frame_raw);
        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255)
            grab_frames = false;
    }

    LOG(INFO) << "Shutting down.";
    return absl::OkStatus();
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunFaceLandmarker();
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else
    {
        LOG(INFO) << "Success!";
    }

    return EXIT_SUCCESS;
}