import arcpy
import os
import imageio

# --- Settings ---
project_path = r"C:\Path\To\Your\ArcGISProject.aprx"
map_name = "Map"  # name of the map inside ArcGIS Pro
tif_folder = r"C:\Path\To\TIFs"
output_folder = r"C:\Path\To\ExportedFrames"
gif_output = r"C:\Path\To\Final\surf_o3_animation.gif"
frame_size = (800, 600)  # width, height in pixels
frame_delay = 0.5  # seconds between frames

# --- Load the project and map ---
aprx = arcpy.mp.ArcGISProject(project_path)
map_obj = aprx.listMaps(map_name)[0]
layout = aprx.listLayouts()[0]
mf = layout.listElements("MAPFRAME_ELEMENT")[0]

# --- Ensure output folder exists ---
os.makedirs(output_folder, exist_ok=True)

# --- Remove any existing rasters ---
for lyr in map_obj.listLayers():
    if lyr.name.endswith('.tif'):
        map_obj.removeLayer(lyr)

# --- Load rasters ---
tifs = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')])
png_paths = []

for i, tif in enumerate(tifs):
    tif_path = os.path.join(tif_folder, tif)

    # Add raster to map
    raster_lyr = map_obj.addDataFromPath(tif_path)

    # Turn off all other layers
    for lyr in map_obj.listLayers():
        lyr.visible = False
    raster_lyr.visible = True

    # Export the layout to PNG
    png_path = os.path.join(output_folder, f"frame_{i:03d}.png")
    layout.exportToPNG(png_path, resolution=150, height=frame_size[1], width=frame_size[0])
    png_paths.append(png_path)

# --- Create GIF using imageio ---
images = [imageio.v2.imread(png) for png in png_paths]
imageio.mimsave(gif_output, images, duration=frame_delay)

print(f"🎉 GIF saved to: {gif_output}")