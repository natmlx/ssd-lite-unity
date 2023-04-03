# Single Shot Detector Lite
[SSD Lite](https://arxiv.org/abs/1512.02325) high performance general object detection.

## Installing SSD Lite
Add the following items to your Unity project's `Packages/manifest.json`:
```json
{
  "scopedRegistries": [
    {
      "name": "NatML",
      "url": "https://registry.npmjs.com",
      "scopes": ["ai.natml"]
    }
  ],
  "dependencies": {
    "ai.natml.vision.ssd-lite": "1.0.1"
  }
}
```

## Detecting Objects in an Image
First, create the SSD Lite predictor:
```csharp
// Create the SSD Lite predictor
var predictor = await SSDLitePredictor.Create();
```

Then detect objects in the image:
```csharp
// Create image feature
Texture2D image = ...;
// Detect objects
SSDLitePredictor.Detection[] detections = predictor.Predict(image);
```

___

## Requirements
- Unity 2021.2+

## Quick Tips
- Join the [NatML community on Discord](https://natml.ai/community).
- Discover more ML models on [NatML Hub](https://hub.natml.ai).
- See the [NatML documentation](https://docs.natml.ai/unity).
- Contact us at [hi@natml.ai](mailto:hi@natml.ai).

Thank you very much!