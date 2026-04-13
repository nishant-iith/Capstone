import SimpleITK as sitk
import numpy as np
import os
from pathlib import Path

def elastic_registration(fixed_path, moving_path, output_path):
    """
    Performs B-Spline elastic registration to align unstained (moving) 
    to stained (fixed) images to correct tissue distortion.
    """
    # Load images
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # Initialize registration
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric: Mutual Information (Best for multi-modal/stain-diff images)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    # Optimizer: Gradient Descent
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, 
                                                     convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Transform: B-Spline (Elastic)
    # Use a coarse grid for the control points to ensure smooth warping
    transform_domain_mesh_size = [8] * fixed.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(fixed, transform_domain_mesh_size)
    registration_method.SetInitialTransform(initial_transform)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    try:
        final_transform = registration_method.Execute(fixed, moving)
        
        # Resample moving image to fixed image space
        resampled = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        sitk.WriteImage(resampled, output_path)
        return True
    except Exception as e:
        print(f"Registration failed for {moving_path}: {e}")
        return False

if __name__ == "__main__":
    # Mock test
    import sys
    if len(sys.argv) > 3:
        success = elastic_registration(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Success: {success}")
    else:
        print("Usage: python registration.py <fixed> <moving> <output>")
