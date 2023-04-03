/* 
*   SSD Lite
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Vision {

    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using UnityEngine;
    using NatML.Features;
    using NatML.Internal;
    using NatML.Types;

    /// <summary>
    /// Single Shot Detector Lite predictor for general object detection.
    /// This predictor accepts an image feature and produces a list of detections.
    /// </summary>
    public sealed class SSDLitePredictor : IMLPredictor<SSDLitePredictor.Detection[]> {

        #region --Types--

        public readonly struct Detection {

            /// <summary>
            /// Detection rectangle in normalized coordinates.
            /// </summary>
            public readonly Rect rect;
            /// <summary>
            /// Detection label.
            /// </summary>
            public readonly string label;
            /// <summary>
            /// Normalized detection score.
            /// </summary>
            public readonly float score;

            public Detection (Rect rect, string label, float score) {
                this.rect = rect;
                this.label = label;
                this.score = score;
            }
        }
        #endregion


        #region --Client API--
        /// <summary>
        /// Predictor tag.
        /// </summary>
        public const string Tag = "@natsuite/ssd-lite";

        /// <summary>
        /// Detect objects in an image.
        /// </summary>
        /// <param name="inputs">Input image.</param>
        /// <returns>Detected objects in normalized coordinates.</returns>
        public Detection[] Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"SSD Lite predictor expects a single feature", nameof(inputs));
            // Check type
            var input = inputs[0];
            var imageType = MLImageType.FromType(input.type);
            var imageFeature = input as MLImageFeature;
            if (!imageType)
                throw new ArgumentException(@"SSD Lite predictor expects an an array or image feature", nameof(inputs));
            // Apply normalization
            if (imageFeature != null) {
                (imageFeature.mean, imageFeature.std) = model.normalization;
                imageFeature.aspectMode = model.aspectMode;
            }
            // Predict
            var inputType = model.inputs[0] as MLImageType;
            using var inputFeature = (input as IMLEdgeFeature).Create(inputType);
            using var outputFeatures = model.Predict(inputFeature);
            // Marshal
            var scores = new MLArrayFeature<float>(outputFeatures[0]);  // (1,3000,21)
            var boxes = new MLArrayFeature<float>(outputFeatures[1]);   // (1,3000,4)
            var result = new List<Detection>();
            for (int c = 1, clen = scores.shape[2]; c < clen; ++c) {
                // Clear
                candidateBoxes.Clear();
                candidateScores.Clear();
                // Extract
                for (int p = 0, plen = scores.shape[1]; p < plen; ++p) {
                    var score = scores[0,p,c];
                    if (score < minScore)
                        continue;
                    var rawBox = Rect.MinMaxRect(boxes[0,p,0], 1f - boxes[0,p,3], boxes[0,p,2], 1f - boxes[0,p,1]);
                    var box = imageFeature != null ? imageFeature.TransformRect(rawBox, inputType) : rawBox;
                    candidateBoxes.Add(box);
                    candidateScores.Add(score);
                }
                // NMS
                var keepIdx = MLImageFeature.NonMaxSuppression(candidateBoxes, candidateScores, maxIoU);
                foreach (var idx in keepIdx)
                    result.Add(new Detection(candidateBoxes[idx], model.labels[c], candidateScores[idx]));
            }
            // Return
            return result.ToArray();
        }

        /// <summary>
        /// Dispose the predictor and release resources.
        /// </summary>
        public void Dispose () => model.Dispose();

        /// <summary>
        /// Create the SSD Lite predictor.
        /// </summary>
        /// <param name="model">MobileNet v2 SSD Lite ML model.</param>
        /// <param name="labels">Class labels.</param>
        /// <param name="minScore">Minimum candidate score.</param>
        /// <param name="maxIoU">Maximum intersection-over-union score for overlap removal.</param>
        public static async Task<SSDLitePredictor> Create (
            float minScore = 0.6f,
            float maxIoU = 0.5f,
            MLEdgeModel.Configuration configuration = null,
            string accessKey = null
        ) {
            var model = await MLEdgeModel.Create(Tag, configuration, accessKey);
            var predictor = new SSDLitePredictor(model, minScore, maxIoU);
            return predictor;
        }
        #endregion


        #region --Operations--
        private readonly MLEdgeModel model;
        private readonly float minScore;
        private readonly float maxIoU;
        private readonly List<Rect> candidateBoxes;
        private readonly List<float> candidateScores;

        private SSDLitePredictor (MLEdgeModel model, float minScore = 0.6f, float maxIoU = 0.5f) {
            this.model = model;
            this.minScore = minScore;
            this.maxIoU = maxIoU;
            this.candidateBoxes = new List<Rect>(1 << 4);   // should be large enough for most
            this.candidateScores = new List<float>(1 << 4);
        }
        #endregion
    }
}