/* 
*   SSD Lite
*   Copyright (c) 2022 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML.Devices;
    using NatML.Devices.Outputs;
    using NatML.Features;
    using NatML.Vision;
    using Visualizers;

    public sealed class SSDLiteSample : MonoBehaviour {

        [Header(@"UI")]
        public SSDLiteVisualizer visualizer;

        private CameraDevice cameraDevice;
        private TextureOutput cameraTextureOutput;

        private MLModelData modelData;
        private MLModel model;
        private SSDLitePredictor predictor;

        async void Start () {
            // Request camera permissions
            var permissionStatus = await MediaDeviceQuery.RequestPermissions<CameraDevice>();
            if (permissionStatus != PermissionStatus.Authorized) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Discover a camera
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraTextureOutput = new TextureOutput();
            cameraDevice.StartRunning(cameraTextureOutput);
            // Display the camera preview
            var cameraTexture = await cameraTextureOutput;
            visualizer.image = cameraTexture;
            // Create the SSD Lite predictor
            modelData = await MLModelData.FromHub("@natsuite/ssd-lite");
            model = modelData.Deserialize();
            predictor = new SSDLitePredictor(model, modelData.labels);
        }

        void Update () {
            // Check that predictor has been created
            if (predictor == null)
                return;
            // Create image feature
            var inputFeature = new MLImageFeature(cameraTextureOutput.texture);
            (inputFeature.mean, inputFeature.std) = modelData.normalization;
            inputFeature.aspectMode = modelData.aspectMode;
            // Detect objects
            var detections = predictor.Predict(inputFeature);
            // Visualize detections
            visualizer.Render(detections);
        }

        void OnDisable () {
            // Dispose the model
            model?.Dispose();
        }
    }
}
