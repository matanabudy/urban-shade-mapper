import math
from datetime import datetime, timezone


from sunfinder.calculate_shade import __coordinate_and_time_to_sun_azimuth_and_zenith

import math
import matplotlib.pyplot as plt
from PIL import Image


def create_combined_image(images, output_filename):

    num_images = len(images)

    if num_images == 0:
        print("No images to combine.")
        return

    image_width, image_height = images[0].size

    total_width = image_width
    total_height = image_height * num_images

    combined_image = Image.new('RGB', (total_width, total_height))

    current_y = 0
    for image in images:
        combined_image.paste(image, (0, current_y))
        current_y += image_height

    combined_image.save(output_filename)
    print(f"Combined image saved as {output_filename}")


def create_combined_image_main():
    image_names = ["shade_at_{}.png".format(hour) for hour in range(6, 20)]
    images = [Image.open(image_name) for image_name in image_names]

    # Define the number of images in each group
    group_sizes = [3, 3, 3, 3, 2]

    start_idx = 0
    group_num = 1

    for group_size in group_sizes:
        end_idx = start_idx + group_size
        group_images = images[start_idx:end_idx]
        create_combined_image(group_images, f"combined_image_{group_num}.png")

        start_idx = end_idx
        group_num += 1




def plot_street_with_shade(shade_length, shade_side, left_sidewalk_length, road_length, right_sidewalk_length, hour,left_side,right_side):
    # Calculate the total width of the street
    street_width = left_sidewalk_length + road_length + right_sidewalk_length

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot left sidewalk edge
    ax.plot([0, left_sidewalk_length], [0, 0], color='black', linewidth=3)
    ax.plot([0, 0], [0, 1], color='black', linewidth=3)
    ax.text(left_sidewalk_length / 2, 0.25, f'Left Sidewalk\n({left_side}°)', ha='center', va='top')

    # Plot road
    ax.plot([left_sidewalk_length, left_sidewalk_length + road_length], [0, 0], color='black', linewidth=1)
    ax.plot([left_sidewalk_length, left_sidewalk_length], [0, 1], color='black', linestyle='dotted')
    ax.plot([left_sidewalk_length + road_length, left_sidewalk_length + road_length], [0, 1], color='black',
            linestyle='dotted')
    ax.text(left_sidewalk_length + road_length / 2, 0.25, 'Road', ha='center', va='top')

    # Plot right sidewalk edge
    ax.plot([street_width - right_sidewalk_length, street_width], [0, 0], color='black', linewidth=3)
    ax.plot([street_width, street_width], [0, 1], color='black', linewidth=3)

    ax.text(street_width - right_sidewalk_length / 2, 0.25, f'Right Sidewalk\n({right_side}°)', ha='center', va='top')

    # Calculate shade starting point based on shade_side
    if shade_side == 'from left building':
        shade_start = 0
    else:
        shade_start = street_width

    # Calculate shade ending point
    if shade_side == 'from left building':
        if shade_length > street_width:
            shade_end = street_width
        else:
            shade_end = shade_start + shade_length
    else:
        if shade_length > street_width:
            shade_end = 0
        else:
            shade_end = shade_start - shade_length

    # Plot the shade as a blue horizontal line
    ax.plot([shade_start, shade_end], [0.5, 0.5], color='blue', linewidth=2)

    if shade_side == 'from left building':
        yellow_line_start = shade_end
    else:
        yellow_line_start = shade_end

    if shade_side == 'from left building':
        yellow_line_end = street_width
    else:
        yellow_line_end = 0

    # Plot the yellow line as a horizontal line
    ax.plot([yellow_line_start, yellow_line_end], [0.5, 0.5], color=(1.0, 0.7, 0.0), linewidth=2)

    # Set axis limits and labels
    ax.set_xlim(0, street_width)
    ax.set_ylim(-0.1, 1)  # Adjust y-limit to accommodate shade line and labels

    ax.set_yticks([])  # Remove y ticks
    ax.set_xticks([])  # Remove x ticks

    # Crop whitespace
    plt.tight_layout()

    # Set title and show the plot
    plt.savefig(f"shade_at_{hour}.png")

    # Save also as EPS file for use in LaTeX
    plt.savefig(f"paper/sidewalk_predictions/shade_at_{hour}.eps", format='eps', dpi=1200)


def corrected_calculate_shade_on_street(h_l, h_r, width, elevation, azimuth, beta):
    """
    Calculate where the shadow falls on a given street.

    Parameters:
    - h_l: Height of the buildings on the left side of the street.
    - h_r: Height of the buildings on the right side of the street.
    - width: Width of the street.
    - zenith: Zenith angle of the sun (angle between the sun and the vertical) in degrees.
    - elevation: Elevation angle of the sun (angle between the sun and the North, going clockwise) in degrees.
    - beta: Direction of the street (angle between the direction of the street and the North, going clockwise) in degrees.

    Returns:
    - Distance where the shadow falls on the street.
    - Side from which the shadow falls.
    """
    delta = (azimuth - beta + 360) % 360

    angle_diff = min(abs(delta - 90), abs(delta - 180))

    if 0 < delta < 180:
        b_tag = h_r / math.tan(math.radians(elevation))
        b = math.cos(math.radians(angle_diff)) * b_tag
        return b, "from right building"
    else:
        b_tag = h_l / math.tan(math.radians(elevation))
        b = math.cos(math.radians(angle_diff)) * b_tag
        return b, "from left building"


def describe_shadow(distance, width, side_angle, left_sidewalk_width, right_sidewalk_width, left_side, right_side):
    near_sun_shade = "sun"
    far_sun_shade = "sun"
    other_side_angle = (side_angle + 180) % 360
    if side_angle == left_side:
        near_sun_sidewalk_width = left_sidewalk_width
        far_sun_sidewalk_width = right_sidewalk_width
    else:
        near_sun_sidewalk_width = right_sidewalk_width
        far_sun_sidewalk_width = left_sidewalk_width

    if distance > 0.2*near_sun_sidewalk_width:
        near_sun_shade = "shade"

    if distance > width - 0.8*far_sun_sidewalk_width:
        far_sun_shade = "shade"

    base_message = f"There is shade at a distance of {distance:.2f}m from the side at {side_angle}°."
    if distance > width:
        base_message = f"The entire street is in the shade due to building at {side_angle}°."
    elif distance < 0:
        base_message = "The entire street is in the sun."


    pavement_message = f"\n{side_angle}° pavement is in the {near_sun_shade}. {other_side_angle}° pavement is in the {far_sun_shade}.\n"

    return base_message + " " + pavement_message


def process_hourly_shadows(height_l, height_r, width, left_pavement_width, right_pavement_width, street_dir, lat, lon,
                           date):
    left_side = (street_dir - 90) % 360
    right_side = (street_dir + 90) % 360

    # Loop through hours 4 to 19
    for hour in range(4, 20):
        current_datetime = datetime(date.year, date.month, date.day, hour, tzinfo=timezone.utc)
        az, zen = __coordinate_and_time_to_sun_azimuth_and_zenith(lat, lon, current_datetime)

        if zen >= 90:
            continue

        distance, side_str = corrected_calculate_shade_on_street(height_l, height_r, width, 90 - zen, az, street_dir)

        side_angle = left_side if "left" in side_str else right_side

        print(f"At {hour+2}:00:", describe_shadow(distance, width, side_angle, left_pavement_width, right_pavement_width, left_side, right_side))

        road_width = width - left_pavement_width - right_pavement_width
        plot_street_with_shade(distance,side_str,left_pavement_width,road_width,right_pavement_width,hour+2,left_side,right_side)



def main():
    height_l = 15
    height_r = 17.5
    width = 13.5
    left_pavement_width = 3.5
    right_pavement_width = 3.5
    street_dir = 211
    lat, lon = 32.0873861, 34.7712034
    date = datetime(2023, 7, 19)

    process_hourly_shadows(height_l, height_r, width, left_pavement_width, right_pavement_width, street_dir, lat, lon,
                           date)

    create_combined_image_main()

if __name__ == '__main__':
    main()