/* 
*   SSD Lite
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using NatML.Vision;
    using NatML.VideoKit;
    using Visualizers;

    public sealed class SSDLiteSample : MonoBehaviour {

        [Header(@"VideoKit")]
        public VideoKitCameraManager cameraManager;

        [Header(@"UI")]
        public SSDLiteVisualizer visualizer;

        private SSDLitePredictor predictor;

        private async void Start () {
            // Create the SSD Lite predictor
            predictor = await SSDLitePredictor.Create();
            // Listen for camera frames
            cameraManager.OnCameraFrame.AddListener(OnCameraFrame);
        }

        private void OnCameraFrame (CameraFrame frame) {
            // Detect objects
            var detections = predictor.Predict(frame);
            // Visualize detections
            visualizer.Render(detections);
        }

        private void OnDisable () {
            // Stop listening for camera frames
            cameraManager.OnCameraFrame.RemoveListener(OnCameraFrame);
            // Dispose the predictor
            predictor?.Dispose();
        }
    }
}
