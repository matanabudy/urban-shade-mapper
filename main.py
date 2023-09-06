import csv
from datetime import date
from pathlib import Path
import shutil
import os

import numpy as np
from PIL import Image

from fisheye.gsv2fisheye import gsvLatLong2fisheye
from shade_segment_anything.ImageSegmenter import ImageSegmenter
from sunfinder.calculate_shade import get_shade_in_different_hours_of_day, get_shade_in_different_intervals_of_day

# Constants
IMAGE_DIR = Path("images")
DEFAULT_ZOOM = 4  # Controls the final image size from GSV
TEST_POINTS_PATH = Path("test_points.csv")
TIME_DIFFERENCE_FROM_UK = 2


def load_test_points():
    test_points = []
    with open(TEST_POINTS_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_points.append(row)

    return test_points


def get_sun_status(latitude: float, longitude: float, point_name: str, date_of_interest: date, zoom: int,
                   image_segmenter: ImageSegmenter, is_ten_minute_interval: bool):
    out_dir_path = IMAGE_DIR / point_name
    # Convert lat/long to fisheye image
    fish_eye_out_path, nearest_pano_coordinates = gsvLatLong2fisheye(latitude, longitude, out_dir_path, zoom)

    # Segment the image
    seg_fisheye_out_path, pixel_percentages = image_segmenter.segment_image(fish_eye_out_path, fish_eye_out_path)

    # Process the image
    img_array, img = convert_img_to_np_array(seg_fisheye_out_path)

    # Calculate the shade at different times of day
    if is_ten_minute_interval:
        sun_status = get_shade_in_different_intervals_of_day(longitude, latitude, img_array, img, date_of_interest, TIME_DIFFERENCE_FROM_UK,
                                                     seg_fisheye_out_path)
    else:
        sun_status = get_shade_in_different_hours_of_day(longitude, latitude, img_array, img, date_of_interest, TIME_DIFFERENCE_FROM_UK,
                                                     seg_fisheye_out_path)
    shutil.rmtree(out_dir_path)
    return sun_status, nearest_pano_coordinates, pixel_percentages

def main():
    CACHE_INTERVAL = 10
    already_processed_points = set()
    if os.path.exists("results.csv"):
        with open("results.csv", "r", newline='') as existing_file:
            reader = csv.DictReader(existing_file)
            for row in reader:
                already_processed_points.add(row["point"])

    result_file = open("results.csv", "a", newline='')  # append mode
    fieldnames = ["point", "time", "is_shade", "tree_pixels", "building_pixels", "sky_pixels",
                  "nearest_pano_coordinates"]
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    if not already_processed_points:
        writer.writeheader()  # writes the headers only if new file

    image_segmenter = ImageSegmenter()

    test_points = load_test_points()
    points_processed_in_current_run = 0
    for test_point in test_points:
        point_name = test_point["name"]
        if point_name in already_processed_points:
            print(f"Skipping {point_name} as it's already processed.")
            continue
        is_ten_minutes_intervals = test_point.get("short_intervals", "N") == "Y"
        print(f"Processing {point_name}")

        latitude = float(test_point["lat"])
        longitude = float(test_point["lon"])
        date_of_interest = date.fromisoformat(test_point["date_of_interest"])
        sun_status, nearest_pano_coordinates, pixel_percentages = get_sun_status(latitude, longitude, point_name, date_of_interest,
                                                              DEFAULT_ZOOM, image_segmenter, is_ten_minutes_intervals)
        tree_pixels_percentage =  pixel_percentages["tree"]
        building_pixels_percentage = pixel_percentages["building"]
        sky_pixels_percentage =  pixel_percentages["sky"]

        for time, shade in sun_status.items():
            if isinstance(time, tuple):
                formatted_time = f"{time[0]}:{time[1]:02}"
            else:
                formatted_time = f"{time}:00"
            writer.writerow({
                "point": point_name,
                "time": formatted_time,
                "is_shade": shade,
                "tree_pixels": tree_pixels_percentage,
                "building_pixels": building_pixels_percentage,
                "sky_pixels": sky_pixels_percentage,
                "nearest_pano_coordinates": f"({nearest_pano_coordinates[0]}, {nearest_pano_coordinates[1]})"
            })

        points_processed_in_current_run += 1
        if points_processed_in_current_run % CACHE_INTERVAL == 0:
            print(f"Processed {points_processed_in_current_run} points. Saving results to disk.")
            result_file.flush()

    result_file.close()


def convert_img_to_np_array(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)

    return img_array, img


if __name__ == '__main__':
    main()
