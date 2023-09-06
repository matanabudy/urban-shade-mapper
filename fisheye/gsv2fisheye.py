import math
import os
from pathlib import Path

import cv2
import numpy as np
import streetview
from PIL import Image
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from streetview.search import Panorama
from tqdm import tqdm
from retrying import retry


from fisheye.google_geocoding_client import GoogleGeoCodingClient

load_dotenv()

def convert_panorama_to_fisheye(panorama_file_path: Path, fisheye_out_path: Path, yaw_angle: float):
    """
    Adapted from: https://github.com/xiaojianggis/shadefinder

    This program is used to convert cylindrical panorama images to original image
    Copyright (C) Xiaojiang Li, UCONN, MIT Senseable City Lab
    First version June 25, 2016

    Be careful, for the GSV panoramas, the R2 and R22 are different, the R22
    or the height based method is 3times of the width based method,however,
    for the example fisheye image collected online the R2 and R22 are almost
    the same. This proves that Google SQUEEZED THE COLUMN OF GSV PANORAMA, in
    order to rescale the Google panorama, the columns should time 3

    vecX = xD - CSx
    vecY = yD - CSy
    theta1 = math.asin(vecX/(r+0.0001))
    theta2 = math.acos(vecX/(r+0.0001))

    Saves the rotated fish eye image to output_path
    """
    rotate_angle = 360 - yaw_angle
    pano_img = np.array(Image.open(panorama_file_path))
    source_height, source_width = pano_img.shape[0], pano_img.shape[1]
    half_pano_img = pano_img[0:int(source_height / 2), :]

    # get the radius of the fisheye
    R1 = 0
    R2 = int(2 * source_width / (2 * np.pi) - R1 + 0.5)

    # estimate the size of the sphere or fish-eye image
    dest_width = dest_height = int(source_width / np.pi) + 2

    # create empty matrices to store the affine parameters
    x_map = np.zeros((dest_height, dest_width), np.float32)
    y_map = np.zeros((dest_height, dest_width), np.float32)

    # the center of the destination image, or the sphere image
    CSx = int(0.5 * dest_width)
    CSy = int(0.5 * dest_height)

    # split the sphere image into four parts, and reproject the panorama for each section
    for yD in tqdm(range(dest_height)):
        for xD in range(1, CSx):
            r = math.sqrt((yD - CSy) ** 2 + (xD - CSx) ** 2)
            theta = 0.5 * np.pi + math.atan((yD - CSy) / (xD - CSx + 0.0001))

            xS = theta / (2 * np.pi) * source_width
            yS = (r - R1) / (R2 - R1) * source_height

            x_map.itemset((yD, xD), xS)
            y_map.itemset((yD, xD), yS)

        for xD in range(CSx + 1, dest_width):
            r = math.sqrt((yD - CSy) ** 2 + (xD - CSx) ** 2)
            theta = 1.5 * np.pi + math.atan((yD - CSy) / (xD - CSx + 0.0001))

            xS = theta / (2 * np.pi) * source_width
            yS = (r - R1) / (R2 - R1) * source_height

            x_map.itemset((yD, xD), xS)
            y_map.itemset((yD, xD), yS)

    output_img = cv2.remap(half_pano_img, x_map, y_map, cv2.INTER_CUBIC)

    # Rotate the generated fisheye image to ensure that the top of the fisheye image is facing north.
    rows, cols, _ = output_img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
    rotated_fisheye_img = cv2.warpAffine(output_img, rotation_matrix, (cols, rows))

    img = Image.fromarray(rotated_fisheye_img)
    img.save(fisheye_out_path)


def get_coordinates_by_address_geolocator(address: str) -> tuple[float, float]:
    geolocator = Nominatim(user_agent="gsv2hem")
    location = geolocator.geocode(address)
    if not location:
        raise ValueError(f"Could not find location for address {address}")

    print(f'Found address coordinates: {location.latitude}, {location.longitude}')
    return location.latitude, location.longitude


def get_coordinates_by_address_google(address: str) -> tuple[float, float]:
    geo_coding_client = GoogleGeoCodingClient(os.getenv('GOOGLE_API_KEY'))
    try:
        response = geo_coding_client.get_address_coordinates(address)
        coordinates = response.json()["results"][0]["geometry"]["location"]
        latitude = coordinates["lat"]
        longitude = coordinates["lng"]
        print(f'Found address coordinates: {latitude}, {longitude}')

    except Exception as e:
        raise Exception(f"Error in google geocoding api: {response.content}.")
    return latitude, longitude

@retry(stop_max_attempt_number=3, wait_fixed=5000)
def get_panorama_by_pano_id(pano_id: str, out_dir: Path, zoom: int) -> Path:
    out_path = out_dir / f"{pano_id}.jpg"
    if out_path.exists():
        print(f"Panorama already exists, not downloading again")
        return out_path

    print('Downloading panorama...')
    image = streetview.get_panorama(pano_id, zoom=zoom, should_crop=True)
    image.save(out_path)

    return out_path


def get_latest_closest_panorama_by_coordinates(latitude: float, longitude: float) -> Panorama:
    pano_ids = streetview.search_panoramas(lat=latitude, lon=longitude)
    if not pano_ids:
        raise ValueError(f"Could not find panorama for coordinates {latitude}, {longitude}")

    return pano_ids[-1]


def gsvLatLong2fisheye(latitude: float, longitude: float, out_dir_path: Path, zoom: int) -> tuple[Path, tuple[float, float]]:
    # TODO: Get actual latitude and longitude from the retrieved panorama
    panorama = get_latest_closest_panorama_by_coordinates(latitude, longitude)
    if not out_dir_path.exists():
        print(f"Creating output directory {out_dir_path}")
        out_dir_path.mkdir(parents=True)

    nearest_pano_coordinates = panorama.lat, panorama.lon
    panorama_file_path = get_panorama_by_pano_id(panorama.pano_id, out_dir_path, zoom)
    fisheye_out_path = out_dir_path / f"{panorama.pano_id}_fisheye.jpg"
    convert_panorama_to_fisheye(panorama_file_path, fisheye_out_path, panorama.heading)
    return fisheye_out_path, nearest_pano_coordinates
