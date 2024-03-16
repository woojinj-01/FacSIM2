import colorsys

palette_bio = ['#0444BF', '#0584F2', '#0AAFF1', '#1D65A6', '#72A2C0']
palette_cs = ['#C1403D', '#F46A4E', '#F4874B', '#F3B05A', '#F3D4A0']
palette_phy = ['#107050', '#4DD8AD', '#55D9C0', '#C7F6EC', '#E7F5DE']


def hex_to_rgb(hex_color):

    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:], 16)

    return (red, green, blue)


def rgb_to_hex(rgb):

    red = min(max(rgb[0], 0), 255)
    green = min(max(rgb[1], 0), 255)
    blue = min(max(rgb[2], 0), 255)

    hex_color = "#{:02X}{:02X}{:02X}".format(red, green, blue)
    
    return hex_color


def hex_to_hsv(hex_color):

    rgb_color = hex_to_rgb(hex_color)

    hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
    
    return hsv_color


def hsv_to_hex(hsv_color):

    rgb_color = colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], hsv_color[2])
    
    hex_color = "#{:02X}{:02X}{:02X}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
    
    return hex_color


def split_color_by(hex_color, division=5):

    hsv_origin = hex_to_hsv(hex_color)

    base_saturation = 0.3

    delta = (hsv_origin[1] - base_saturation) / (division - 1)

    hex_colors = [hsv_to_hex((hsv_origin[0], hsv_origin[1] - delta * i, hsv_origin[2])) for i in range(0, division)]

    return hex_colors