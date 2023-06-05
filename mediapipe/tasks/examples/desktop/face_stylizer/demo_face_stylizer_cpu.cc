#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/tasks/cc/vision/face_stylizer/face_stylizer.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

constexpr char kWindowName[] = "Face Stylizer";

ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");

ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunFaceStylizer()
{
    LOG(INFO) << "Initialize Face Landmarker options.";
    std::unique_ptr<mediapipe::tasks::vision::face_stylizer::FaceStylizerOptions>
        options(new mediapipe::tasks::vision::face_stylizer::FaceStylizerOptions());
    options->base_options.model_asset_path = "./mediapipe/tasks/examples/desktop/face_stylizer/assets/face_stylizer.task";
    options->running_mode = mediapipe::tasks::vision::core::RunningMode::VIDEO;

    LOG(INFO) << "Initialize Face Landmarker.";
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::face_stylizer::FaceStylizer>>
        face_stylizer = mediapipe::tasks::vision::face_stylizer::FaceStylizer::Create(std::move(options));

    LOG(INFO) << "Initialize the camera.";
    cv::VideoCapture capture;

    const std::string input_video_path = absl::GetFlag(FLAGS_input_video_path);
    const bool load_video = !input_video_path.empty();
    if (load_video)
    {
        capture.open(input_video_path);
    }
    else
    {
        capture.open(0);
    }

    RET_CHECK(capture.isOpened());

    cv::VideoWriter writer;

    const std::string output_video_path = absl::GetFlag(FLAGS_output_video_path);
    const bool save_video = !output_video_path.empty();
    if (!save_video)
    {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
// #if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_FPS, 30);
// #endif
    }

    LOG(INFO) << "Start grabbing and processing frames.";

    int count_frames = 5 * 60 * 30;

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
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);

        // Convert ImageFrame to Image (input of FaceLandmaker model)
        auto input_image = std::make_shared<mediapipe::Image>(input_frame);

        // Create a MatView from Image, then copy openCV camera or video data to it
        std::shared_ptr<cv::Mat> input_image_mat = mediapipe::formats::MatView(input_image.get());
        camera_frame.copyTo(*input_image_mat);

        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        // Stylizer video
        std::optional<mediapipe::Image> output_image = *(*face_stylizer)->StylizeForVideo(*input_image, frame_timestamp_us);
        if (output_image == std::nullopt)
            continue;

        
        // Create a MatView from output Image
        cv::Mat output_image_mat = *mediapipe::formats::MatView(&*output_image);
        cv::cvtColor(output_image_mat, output_image_mat, cv::COLOR_RGB2BGR);

        int fps = cv::getTickFrequency() / (cv::getTickCount() - start);

        cv::putText(output_image_mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(100, 255, 0), 2, cv::LINE_AA);

        if (save_video)
        {
            if (count_frames-- == 1)
                grab_frames = false;
            
            if (!writer.isOpened())
            {
                LOG(INFO) << "Prepare video writer.";
                writer.open(output_video_path,
                            mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                            30, output_image_mat.size());
                RET_CHECK(writer.isOpened());
            }

            LOG(INFO) << count_frames;

            writer.write(output_image_mat);

        }
        else {
            cv::imshow(kWindowName, output_image_mat);
            const int pressed_key = cv::waitKey(5);
            if (pressed_key >= 0 && pressed_key != 255)
                grab_frames = false;
        }
    }

    LOG(INFO) << "Shutting down.";

    if (writer.isOpened())
        writer.release();
        
    return (*face_stylizer)->Close();
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunFaceStylizer();
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run FaceStylizer: " << run_status.message();
        return EXIT_FAILURE;
    }
    else
    {
        LOG(INFO) << "Success!";
    }

    return EXIT_SUCCESS;
}
