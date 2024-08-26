import os

# Move to the parent directory
os.chdir("../..")

# Verify the current working directory
print(os.getcwd())

from model.segmentation.segmentation_generator import SegmentationGenerator

seg_generator = SegmentationGenerator()