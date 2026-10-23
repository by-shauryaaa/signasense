# Core ML Model Setup Instructions for SignaSense

This directory (`SignaSense/Resources/`) is the designated plug-and-play location for your trained Core ML model.

### **Where to Place the Model**
Place your trained `.mlmodel` or `.mlpackage` file right here:
```
SignaSense/Resources/SignClassifier.mlmodel
```

### **How It Works in Real Time**
1. `ModelManager.swift` automatically detects `SignClassifier.mlmodel` or `SignClassifier.mlmodelc` in the app bundle at launch.
2. When the model is present, real-time hand pose keypoints from Apple Vision (`VNDetectHumanHandPoseRequest`) are forwarded directly to the model.
3. Stabilized predictions are translated and spoken via `AVSpeechSynthesizer` in real time with zero hardcoded fallbacks.
4. While the model is pending, the application tracks hand landmarks (21 points) and facial expressions in real-time without faking predictions.
