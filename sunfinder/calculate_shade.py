from datetime import datetime, date, timezone
from math import sin, pi, cos, radians

import numpy as np
from PIL import ImageDraw, ImageFont

from sunfinder.sunposition import sunpos
GSV_CAR_ELEVATION = 2.5
GLARE_SIZE = 4

def __coordinate_and_time_to_sun_azimuth_and_zenith(long: float, lat: float, time: datetime):
    return sunpos([time], lat, long, GSV_CAR_ELEVATION)[:2]


def __azmuth_and_zenith_to_x_and_y_on_fisheye_image(azimuth: float, elev: float, radius: int):
    azimuth_skyimg = -(azimuth - 90)
    if azimuth_skyimg < 0: azimuth_skyimg = azimuth_skyimg + 360

    azimuth_radians = radians(azimuth_skyimg)
    elev_radians = radians(elev)

    if elev_radians < 0: elev_radians = 0

    r = (90 - elev_radians * 180 / pi) / 90.0 * radius

    # get the coordinate of the point on the fisheye images
    px = int(r * cos(azimuth_radians) + radius) - 1
    py = int(radius - r * sin(azimuth_radians)) - 1

    return px, py


def __x_y_to_boolean_is_in_sun(px: int, py: int, glareSize, skyImg):
    n = skyImg.shape[0]
    boundXl = px - glareSize
    if boundXl < 0: boundXl = 0
    boundXu = px + glareSize
    if boundXu > n - 1: boundXu = n - 1
    boundYl = py - glareSize
    if boundYl < 0: boundYl = 0
    boundYu = py + glareSize
    if boundYu > n - 1: boundYu = n - 1

    condition = np.array([128, 0, 128, 255])
    idx = np.where(np.all(skyImg[int(boundYl):int(boundYu), int(boundXl):int(boundXu)] == condition, axis=-1))

    sun_percentage = len(idx[0]) / (4 * glareSize * glareSize)
    if sun_percentage > 0.9:
        shade = 0
    elif sun_percentage > 0.1:
        shade = 0.5
    else:
        shade = 1
    return shade


def get_shade_in_different_hours_of_day(longitude, latitude, skyImg, image, date: date, timeDiff, imagePath):
    sun_status = {}
    for hour in range(4, 21):  # For each hour of the day
        current_time = datetime(date.year, date.month, date.day, hour - timeDiff, tzinfo=timezone.utc)
        az, zen = __coordinate_and_time_to_sun_azimuth_and_zenith(longitude, latitude, current_time)
        if zen > 90: #if sun has already set then skip
            sun_status[hour] = 1
            continue
        x, y = __azmuth_and_zenith_to_x_and_y_on_fisheye_image(az, 90 - zen, skyImg.shape[0] / 2)
        result_image = draw_square(image, x, y, hour)
        isInSun = __x_y_to_boolean_is_in_sun(x, y, GLARE_SIZE, skyImg)
        sun_status[hour] = isInSun

    result_image.save(imagePath.split(".")[0] + "_sun_model.png")
    return sun_status


def get_shade_in_different_intervals_of_day(longitude, latitude, skyImg, image, date: date, timeDiff, imagePath):
    sun_status = {}

    for hour in range(4, 21):  # For each hour of the day
        for minute in range(0, 60, 10):  # For each 10-minute interval in the hour
            current_time = datetime(date.year, date.month, date.day, hour - timeDiff, minute, tzinfo=timezone.utc)
            az, zen = __coordinate_and_time_to_sun_azimuth_and_zenith(longitude, latitude, current_time)

            if zen > 90:  # if sun has already set then skip
                sun_status[(hour, minute)] = 1
                continue

            x, y = __azmuth_and_zenith_to_x_and_y_on_fisheye_image(az, 90 - zen, skyImg.shape[0] / 2)
            result_image = draw_square(image, x, y, hour)
            isInSun = __x_y_to_boolean_is_in_sun(x, y, GLARE_SIZE, skyImg)

            sun_status[(hour, minute)] = isInSun

    result_image.save(imagePath.split(".")[0] + "_sun_model.png")
    return sun_status


def draw_square(image, x, y, number):
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    # Calculate the square size relative to the image resolution
    image_size_ratio = min(image_width, image_height) / 1000
    square_size = int(25 * image_size_ratio)

    square_color = (255, 0, 0)
    text_color = (220, 0, 0)

    # Calculate the font size relative to the square size
    font_size = int(square_size * 0.85)

    distance_between_square_and_text = int(20 * image_size_ratio)
    font = ImageFont.load_default()

    # Calculate the half size of the square
    half_size = square_size // 2

    # Calculate the coordinates of the square
    x1 = x - half_size
    y1 = y - half_size
    x2 = x + half_size
    y2 = y + half_size

    # Draw the square on the image
    draw.rectangle([(x1, y1), (x2, y2)], fill=square_color)

    # Calculate the position for the text
    text_width, text_height = draw.textsize(str(number), font=font)
    text_x = x - text_width // 2
    text_y = y - text_height

    # Draw the text on the image
    draw.text((text_x, text_y - distance_between_square_and_text), str(number), fill=text_color, font=font)

    return image
