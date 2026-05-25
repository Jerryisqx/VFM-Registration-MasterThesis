# Veriserum_3D_Registration
A 3D registration method for implant model for post-operative patients.

Our project proceeds in the following steps:

1. Download the Veriserum dataset and obtain 20 different implant models.
2. For each model, establish a ground truth pose and generate their dual-plane fluoroscopy images (dual plane X-ray).
3. Using the dual-plane fluoroscopy images, **reconstruct a 3D proxy**.
4. For this 3D proxy, perform 3D-3D registration (rigid transformation) against our existing model.
5. Apply the resulting transformation back to our model and compare it against the true ground truth poses.

In short: 2x2D-3D reconstruction + 3D-3D registration.

As you can see, the part above is a synthetic test using the model. You are free to choose any method, whether learning-based or a traditional algorithm; we will compare the accuracy of the various methods.

I will provide you with:
1. dupla_renderers, a differentiable renderer we have written (please do not modify it; let me know if you find a bug).
2. Code to generate renderings: generate_rendering.py, renderer.py, testcases.py, and utils.py (feel free to modify).

(Here our renderer depends on the Pytorch3d library. When setting up the environment, configure this first, then fill in any gaps afterwards.)
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

Then run generate_rendering.py to test and obtain the dual-plane X-ray images.

After completing these steps, we can test on real dual-plane datasets (such as Veriserum and our Zimmer postop). For real images, we need to:
1. Obtain the implant masks for both views in the image via the SAM GUI.
2. Read this mask through our algorithm or network and obtain the registered pose.

Note that our data input is not a single pair but multiple pairs, containing temporal information. The implant changes pose during motion and its angle varies, which can also be exploited.

NB: Do not commit to main; create your own new branch, as this is a multi-person collaboration.

<img src="000010_femur.png" width="90%">
