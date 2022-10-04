/* 
*   SSD Lite
*   Copyright (c) 2022 NatML Inc. All Rights Reserved.
*/

namespace NatML.Vision {

    using System;
    using System.Collections.Generic;
    using UnityEngine;
    using NatML.Features;
    using NatML.Internal;
    using NatML.Types;

    /// <summary>
    /// Single Shot Detector Lite predictor for general object detection.
    /// This predictor accepts an image feature and produces a list of detections.
    /// Each detection is comprised of a normalized rect, label, and normalized detection score.
    /// </summary>
    public sealed class SSDLitePredictor : IMLPredictor<(Rect rect, string label, float score)[]> {

        #region --Client API--
        /// <summary>
        /// Class labels.
        /// </summary>
        public readonly string[] labels;

        /// <summary>
        /// Create the SSD Lite predictor.
        /// </summary>
        /// <param name="model">MobileNet v2 SSD Lite ML model.</param>
        /// <param name="labels">Class labels.</param>
        /// <param name="minScore">Minimum candidate score.</param>
        /// <param name="maxIoU">Maximum intersection-over-union score for overlap removal.</param>
        public SSDLitePredictor (MLModel model, string[] labels, float minScore = 0.6f, float maxIoU = 0.5f) {
            this.model = model as MLEdgeModel;
            this.labels = labels;
            this.minScore = minScore;
            this.maxIoU = maxIoU;
            this.candidateBoxes = new List<Rect>(1 << 4);   // Should be large enough for most
            this.candidateScores = new List<float>(1 << 4);
        }

        /// <summary>
        /// Detect objects in an image.
        /// </summary>
        /// <param name="inputs">Input image.</param>
        /// <returns>Detected objects in normalized coordinates.</returns>
        public (Rect rect, string label, float score)[] Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"SSD Lite predictor expects a single feature", nameof(inputs));
            // Check type
            var input = inputs[0];
            var imageType = MLImageType.FromType(input.type);
            var imageFeature = input as MLImageFeature;
            if (!imageType)
                throw new ArgumentException(@"SSD Lite predictor expects an an array or image feature", nameof(inputs));
            // Predict
            var inputType = model.inputs[0] as MLImageType;
            using var inputFeature = (input as IMLEdgeFeature).Create(inputType);
            using var outputFeatures = model.Predict(inputFeature);
            // Marshal
            var scores = new MLArrayFeature<float>(outputFeatures[0]);  // (1,3000,21)
            var boxes = new MLArrayFeature<float>(outputFeatures[1]);   // (1,3000,4)
            var result = new List<(Rect, string, float)>();
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
                    result.Add((candidateBoxes[idx], labels[c], candidateScores[idx]));
            }
            // Return
            return result.ToArray();
        }
        #endregion


        #region --Operations--
        private readonly MLEdgeModel model;
        private readonly float minScore;
        private readonly float maxIoU;
        private readonly List<Rect> candidateBoxes;
        private readonly List<float> candidateScores;

        void IDisposable.Dispose () { } // Not used
        #endregion
    }
}