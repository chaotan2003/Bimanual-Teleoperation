# First import the library
import cv2
import numpy as np

from xrobotoolkit_teleop.hardware.interface.realsense import (
    RealSenseCameraInterface,
    get_supported_resolutions,
)


def main():
    try:
        with RealSenseCameraInterface() as camera_interface:
            print(get_supported_resolutions("215222077461"))
            while True:
                camera_interface.update_frames()  # Fetch new frames from cameras
                frames_dict = camera_interface.get_frames()

                for serial, frames in frames_dict.items():
                    color_image = frames["color"]
                    depth_image = frames["depth"]

                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    images = np.hstack((color_image, depth_colormap))

                    cv2.namedWindow(f"RealSense - {serial}", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(f"RealSense - {serial}", images)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except RuntimeError as e:
        print(e)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
