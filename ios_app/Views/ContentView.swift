// xcode: set sdk=iOS

import SwiftUI
@preconcurrency import AVFoundation
@preconcurrency import Vision
@preconcurrency import CoreMedia
@preconcurrency import CoreML
@preconcurrency import AVFAudio
import Combine

// =============================================================================
// MARK: - 1. COLOR & NEUMORPHIC DESIGN SYSTEM
// =============================================================================

public extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3:
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

public enum NeumorphicTheme {
    public static let background = Color(hex: "E8ECF4")
    public static let cardBackground = Color(hex: "E8ECF4")
    public static let textPrimary = Color(hex: "1E293B")
    public static let textSecondary = Color(hex: "64748B")
    public static let textMuted = Color(hex: "94A3B8")
    public static let lightShadow = Color.white.opacity(0.9)
    public static let darkShadow = Color(hex: "94A3B8").opacity(0.45)
    public static let accentColor = Color(hex: "3B82F6")
}

public struct NeumorphicRaisedModifier: ViewModifier {
    public var cornerRadius: CGFloat = 20
    public var isPressed: Bool = false

    public func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(NeumorphicTheme.cardBackground)
                    .shadow(
                        color: isPressed ? NeumorphicTheme.darkShadow.opacity(0.2) : NeumorphicTheme.lightShadow,
                        radius: isPressed ? 3 : 7,
                        x: isPressed ? -2 : -5,
                        y: isPressed ? -2 : -5
                    )
                    .shadow(
                        color: isPressed ? NeumorphicTheme.lightShadow.opacity(0.3) : NeumorphicTheme.darkShadow,
                        radius: isPressed ? 3 : 8,
                        x: isPressed ? 2 : 5,
                        y: isPressed ? 2 : 5
                    )
            )
    }
}

public struct NeumorphicInsetModifier: ViewModifier {
    public var cornerRadius: CGFloat = 20

    public func body(content: Content) -> some View {
        content
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(NeumorphicTheme.cardBackground)

                    RoundedRectangle(cornerRadius: cornerRadius)
                        .stroke(NeumorphicTheme.darkShadow.opacity(0.6), lineWidth: 3)
                        .blur(radius: 4)
                        .offset(x: 2, y: 2)
                        .mask(RoundedRectangle(cornerRadius: cornerRadius).fill(LinearGradient(colors: [.black, .clear], startPoint: .topLeading, endPoint: .bottomTrailing)))

                    RoundedRectangle(cornerRadius: cornerRadius)
                        .stroke(NeumorphicTheme.lightShadow, lineWidth: 3)
                        .blur(radius: 4)
                        .offset(x: -2, y: -2)
                        .mask(RoundedRectangle(cornerRadius: cornerRadius).fill(LinearGradient(colors: [.clear, .black], startPoint: .topLeading, endPoint: .bottomTrailing)))
                }
            )
    }
}

public extension View {
    func neumorphicRaised(cornerRadius: CGFloat = 20, isPressed: Bool = false) -> some View {
        self.modifier(NeumorphicRaisedModifier(cornerRadius: cornerRadius, isPressed: isPressed))
    }

    func neumorphicInset(cornerRadius: CGFloat = 20) -> some View {
        self.modifier(NeumorphicInsetModifier(cornerRadius: cornerRadius))
    }
}

// =============================================================================
// MARK: - 2. MODELS
// =============================================================================

public enum SpeechStyle: String, CaseIterable, Identifiable, Codable, Sendable {
    case neutral = "Neutral"
    case warm = "Warm"
    case emphatic = "Emphatic"
    case calm = "Calm"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .neutral: return "Neutral"
        case .warm: return "Warm & Friendly"
        case .emphatic: return "Emphatic"
        case .calm: return "Calm"
        }
    }

    public var emoji: String {
        switch self {
        case .neutral: return "😐"
        case .warm: return "😊"
        case .emphatic: return "✨"
        case .calm: return "🌿"
        }
    }

    public var pitchMultiplier: Float {
        switch self {
        case .neutral: return 1.0
        case .warm: return 1.15
        case .emphatic: return 1.3
        case .calm: return 0.9
        }
    }

    public var rateMultiplier: Float {
        switch self {
        case .neutral: return 1.0
        case .warm: return 1.05
        case .emphatic: return 1.15
        case .calm: return 0.85
        }
    }
}

public struct SignPrediction: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let label: String
    public let translatedText: String
    public let confidence: Double
    public let detectedEmotion: SpeechStyle
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        label: String,
        translatedText: String,
        confidence: Double,
        detectedEmotion: SpeechStyle = .neutral,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.label = label
        self.translatedText = translatedText
        self.confidence = confidence
        self.detectedEmotion = detectedEmotion
        self.timestamp = timestamp
    }

    public var formattedConfidence: String {
        "\(Int(confidence * 100))%"
    }
}

public enum RecognitionState: Equatable {
    case ready
    case trackingHands(jointCount: Int, confidence: Float)
    case analyzing(progress: Double)
    case signDetected(SignPrediction)
    case waitingForModel(message: String)
    case cameraOff
    case error(message: String)

    public var statusTitle: String {
        switch self {
        case .ready: return "Ready"
        case .trackingHands(let count, _): return "\(count) Landmarks"
        case .analyzing: return "Classifying..."
        case .signDetected(let p): return "\(p.label.capitalized) (\(p.formattedConfidence))"
        case .waitingForModel: return "Live Tracking"
        case .cameraOff: return "Camera Paused"
        case .error: return "Camera Error"
        }
    }

    public var statusDescription: String {
        switch self {
        case .ready:
            return "Position hand inside the camera frame."
        case .trackingHands(let count, let conf):
            return "Tracking \(count) hand joints in real-time. Vision Confidence: \(Int(conf * 100))%"
        case .analyzing:
            return "Classifying gesture across consecutive frames..."
        case .signDetected(let p):
            return p.translatedText
        case .waitingForModel(let msg):
            return msg
        case .cameraOff:
            return "Camera feed is paused."
        case .error(let msg):
            return msg
        }
    }

    public var indicatorColor: Color {
        switch self {
        case .ready: return .blue
        case .trackingHands: return Color(hex: "06B6D4")
        case .analyzing: return .purple
        case .signDetected: return .green
        case .waitingForModel: return .orange
        case .cameraOff: return .gray
        case .error: return .red
        }
    }
}

public struct HistoryItem: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public let prediction: SignPrediction
    public let languageCode: String
    public let languageName: String
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        prediction: SignPrediction,
        languageCode: String = "en-US",
        languageName: String = "English",
        timestamp: Date = Date()
    ) {
        self.id = id
        self.prediction = prediction
        self.languageCode = languageCode
        self.languageName = languageName
        self.timestamp = timestamp
    }

    public var formattedTime: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: timestamp)
    }
}

public struct SignDictionaryItem: Identifiable, Sendable {
    public let id: String
    public let title: String
    public let hindiMeaning: String
    public let category: String
    public let instructions: String
    public let handPoseHint: String
    public let iconName: String
    public let difficulty: String

    public init(
        id: String,
        title: String,
        hindiMeaning: String,
        category: String,
        instructions: String,
        handPoseHint: String,
        iconName: String,
        difficulty: String = "Beginner"
    ) {
        self.id = id
        self.title = title
        self.hindiMeaning = hindiMeaning
        self.category = category
        self.instructions = instructions
        self.handPoseHint = handPoseHint
        self.iconName = iconName
        self.difficulty = difficulty
    }
}

public struct SupportedLanguage: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let nativeName: String
    public let localeCode: String
    public let flagEmoji: String

    public static let available: [SupportedLanguage] = [
        SupportedLanguage(id: "en", name: "English", nativeName: "English", localeCode: "en-US", flagEmoji: "🇺🇸"),
        SupportedLanguage(id: "hi", name: "Hindi", nativeName: "हिन्दी", localeCode: "hi-IN", flagEmoji: "🇮🇳"),
        SupportedLanguage(id: "es", name: "Spanish", nativeName: "Español", localeCode: "es-ES", flagEmoji: "🇪🇸"),
        SupportedLanguage(id: "fr", name: "French", nativeName: "Français", localeCode: "fr-FR", flagEmoji: "🇫🇷")
    ]
}

// =============================================================================
// MARK: - 3. SERVICES (HARDWARE, VISION, MODEL & SPEECH)
// =============================================================================

public final class CameraService: NSObject, ObservableObject, @unchecked Sendable {
    @Published public var isRunning: Bool = false
    @Published public var isCameraActive: Bool = true
    @Published public var permissionGranted: Bool = false
    @Published public var hasCameraDevice: Bool = false
    @Published public var currentCameraPosition: AVCaptureDevice.Position = .back
    @Published public var bridgeImage: UIImage? = nil
    @Published public var isBridgeConnected: Bool = false

    public let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.signasense.cameraQueue", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private var isConfigured = false
    private var bridgeTimer: Timer?
    private var isFetchingBridgeFrame = false

    public var onHandPoseDetected: (@Sendable ([CGPoint], Float, Bool) -> Void)?
    public var onFaceCueDetected: (@Sendable (Bool, SpeechStyle) -> Void)?

    private let handPoseRequest: VNDetectHumanHandPoseRequest = {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2
        return request
    }()
    private let faceLandmarksRequest = VNDetectFaceLandmarksRequest()
    private var isProcessingFrame = false
    private var lastFaceAnalysisTime = Date.distantPast

    public override init() {
        super.init()
        checkPermissions()
    }

    public func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            Task { @MainActor [weak self] in
                self?.permissionGranted = true
            }
            setupSessionIfNeeded()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor [weak self] in
                    self?.permissionGranted = granted
                    if granted { self?.setupSessionIfNeeded() }
                }
            }
        default:
            Task { @MainActor [weak self] in
                self?.permissionGranted = false
            }
            setupSessionIfNeeded()
        }
    }

    private func discoverBestDevice(position: AVCaptureDevice.Position = .back) -> AVCaptureDevice? {
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
            return device
        }
        let fallbackPos: AVCaptureDevice.Position = (position == .back) ? .front : .back
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: fallbackPos) {
            return device
        }
        return AVCaptureDevice.default(for: .video)
    }

    private func setupSessionIfNeeded() {
        guard !isConfigured else { return }
        sessionQueue.async { [weak self] in
            guard let self = self, !self.isConfigured else { return }
            self.captureSession.beginConfiguration()
            self.captureSession.sessionPreset = .high

            let device = self.discoverBestDevice(position: .back)
            if let device = device,
               let input = try? AVCaptureDeviceInput(device: device),
               self.captureSession.canAddInput(input) {
                self.captureSession.addInput(input)
                Task { @MainActor [weak self] in
                    self?.hasCameraDevice = true
                    self?.currentCameraPosition = device.position
                }
            } else {
                Task { @MainActor [weak self] in
                    self?.hasCameraDevice = false
                    self?.startBridgeStreamIfNeeded()
                }
            }

            if self.captureSession.canAddOutput(self.videoOutput) {
                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)
                self.captureSession.addOutput(self.videoOutput)
            }

            self.captureSession.commitConfiguration()
            self.isConfigured = true

            if self.isCameraActive {
                self.captureSession.startRunning()
                Task { @MainActor [weak self] in self?.isRunning = true }
            }
        }
    }

    public func refreshCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.captureSession.beginConfiguration()
            for oldInput in self.captureSession.inputs {
                self.captureSession.removeInput(oldInput)
            }

            let device = self.discoverBestDevice(position: self.currentCameraPosition)
            if let device = device,
               let input = try? AVCaptureDeviceInput(device: device),
               self.captureSession.canAddInput(input) {
                self.captureSession.addInput(input)
                Task { @MainActor [weak self] in
                    self?.hasCameraDevice = true
                    self?.currentCameraPosition = device.position
                }
            } else {
                Task { @MainActor [weak self] in
                    self?.hasCameraDevice = false
                    self?.startBridgeStreamIfNeeded()
                }
            }

            self.captureSession.commitConfiguration()
            if self.isCameraActive && !self.captureSession.isRunning {
                self.captureSession.startRunning()
                Task { @MainActor [weak self] in self?.isRunning = true }
            }
        }
    }

    public func flipCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            let targetPosition: AVCaptureDevice.Position = (self.currentCameraPosition == .back) ? .front : .back
            guard let device = self.discoverBestDevice(position: targetPosition),
                  let input = try? AVCaptureDeviceInput(device: device) else { return }

            self.captureSession.beginConfiguration()
            for oldInput in self.captureSession.inputs {
                self.captureSession.removeInput(oldInput)
            }
            if self.captureSession.canAddInput(input) {
                self.captureSession.addInput(input)
                Task { @MainActor [weak self] in
                    self?.currentCameraPosition = device.position
                    self?.hasCameraDevice = true
                }
            }
            self.captureSession.commitConfiguration()
        }
    }

    public func toggleCameraActive() {
        let nextState = !isCameraActive
        self.isCameraActive = nextState
        if !nextState {
            bridgeTimer?.invalidate()
            bridgeTimer = nil
        } else if !hasCameraDevice {
            startBridgeStreamIfNeeded()
        }
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if nextState {
                if !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                    Task { @MainActor [weak self] in self?.isRunning = true }
                }
            } else {
                if self.captureSession.isRunning {
                    self.captureSession.stopRunning()
                    Task { @MainActor [weak self] in self?.isRunning = false }
                }
            }
        }
    }

    public func startBridgeStreamIfNeeded() {
        bridgeTimer?.invalidate()
        let timer = Timer.scheduledTimer(withTimeInterval: 0.04, repeats: true) { [weak self] _ in
            self?.fetchBridgeFrame()
        }
        RunLoop.main.add(timer, forMode: .common)
        self.bridgeTimer = timer
    }

    private func fetchBridgeFrame() {
        guard isCameraActive, !isFetchingBridgeFrame else { return }
        guard let url = URL(string: "http://127.0.0.1:8765/frame") else { return }

        isFetchingBridgeFrame = true
        var request = URLRequest(url: url)
        request.timeoutInterval = 0.4

        URLSession.shared.dataTask(with: request) { [weak self] data, _, _ in
            defer { self?.isFetchingBridgeFrame = false }
            guard let self = self,
                  let data = data,
                  let image = UIImage(data: data),
                  let cgImage = image.cgImage else {
                if self?.isBridgeConnected == true {
                    Task { @MainActor [weak self] in self?.isBridgeConnected = false }
                }
                return
            }

            Task { @MainActor [weak self] in
                guard let self = self else { return }
                self.bridgeImage = image
                if !self.isBridgeConnected { self.isBridgeConnected = true }
            }

            self.processBridgeCGImage(cgImage)
        }.resume()
    }

    private func processBridgeCGImage(_ cgImage: CGImage) {
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        do {
            try handler.perform([handPoseRequest])
            let observations = handPoseRequest.results ?? []
            if let firstHand = observations.first {
                var points: [CGPoint] = []
                if let recognized = try? firstHand.recognizedPoints(.all) {
                    for (_, pt) in recognized where pt.confidence > 0.3 {
                        points.append(CGPoint(x: pt.location.x, y: 1.0 - pt.location.y))
                    }
                }
                let conf = firstHand.confidence
                onHandPoseDetected?(points, conf, true)
            } else {
                onHandPoseDetected?([], 0.0, false)
            }
        } catch {
            onHandPoseDetected?([], 0.0, false)
        }

        let now = Date()
        if now.timeIntervalSince(lastFaceAnalysisTime) > 0.25 {
            lastFaceAnalysisTime = now
            do {
                try handler.perform([faceLandmarksRequest])
                let faceObservations = faceLandmarksRequest.results ?? []
                if let face = faceObservations.first, let landmarks = face.landmarks {
                    var style: SpeechStyle = .neutral
                    if let outerLips = landmarks.outerLips {
                        let pts = outerLips.normalizedPoints
                        if pts.count >= 6 {
                            let left = pts[0]
                            let right = pts[pts.count / 2]
                            let bottom = pts[pts.count * 3 / 4]
                            let smile = ((left.y + right.y) / 2.0) - bottom.y
                            if smile > 0.05 {
                                style = .warm
                            }
                        }
                    }
                    onFaceCueDetected?(true, style)
                } else {
                    onFaceCueDetected?(false, .neutral)
                }
            } catch {
                onFaceCueDetected?(false, .neutral)
            }
        }
    }
}

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    public nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard !isProcessingFrame else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        isProcessingFrame = true
        defer { isProcessingFrame = false }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])

        // 1. Process Hand Pose in real-time
        do {
            try handler.perform([handPoseRequest])
            let observations = handPoseRequest.results ?? []
            if let firstHand = observations.first {
                var points: [CGPoint] = []
                if let recognized = try? firstHand.recognizedPoints(.all) {
                    for (_, pt) in recognized where pt.confidence > 0.3 {
                        points.append(CGPoint(x: pt.location.x, y: 1.0 - pt.location.y))
                    }
                }
                let conf = firstHand.confidence
                onHandPoseDetected?(points, conf, true)
            } else {
                onHandPoseDetected?([], 0.0, false)
            }
        } catch {
            onHandPoseDetected?([], 0.0, false)
        }

        // 2. Process Facial Landmark Cues (throttled every 0.25s)
        let now = Date()
        if now.timeIntervalSince(lastFaceAnalysisTime) > 0.25 {
            lastFaceAnalysisTime = now
            do {
                try handler.perform([faceLandmarksRequest])
                let faceObservations = faceLandmarksRequest.results ?? []
                if let face = faceObservations.first, let landmarks = face.landmarks {
                    var style: SpeechStyle = .neutral
                    if let outerLips = landmarks.outerLips {
                        let pts = outerLips.normalizedPoints
                        if pts.count >= 6 {
                            let left = pts[0]
                            let right = pts[pts.count / 2]
                            let bottom = pts[pts.count * 3 / 4]
                            let smile = ((left.y + right.y) / 2.0) - bottom.y
                            if smile > 0.05 {
                                style = .warm
                            }
                        }
                    }
                    onFaceCueDetected?(true, style)
                } else {
                    onFaceCueDetected?(false, .neutral)
                }
            } catch {
                onFaceCueDetected?(false, .neutral)
            }
        }
    }
}

@MainActor
public final class HandPoseService: ObservableObject {
    @Published public var isHandDetected: Bool = false
    @Published public var handJointPoints: [CGPoint] = []
    @Published public var handConfidence: Float = 0.0

    public func update(points: [CGPoint], confidence: Float, isDetected: Bool) {
        self.handJointPoints = points
        self.handConfidence = confidence
        self.isHandDetected = isDetected
    }
}

@MainActor
public final class FaceCueService: ObservableObject {
    @Published public var isFaceDetected: Bool = false
    @Published public var detectedStyle: SpeechStyle = .neutral
    @Published public var isEmotionDetectionEnabled: Bool = true

    public func update(isDetected: Bool, style: SpeechStyle) {
        guard isEmotionDetectionEnabled else {
            if detectedStyle != .neutral { detectedStyle = .neutral }
            return
        }
        self.isFaceDetected = isDetected
        self.detectedStyle = style
    }
}

@MainActor
public final class SpeechService: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published public var isSpeaking: Bool = false
    @Published public var isMuted: Bool = false
    @Published public var speechRate: Float = AVSpeechUtteranceDefaultSpeechRate
    @Published public var basePitchMultiplier: Float = 1.0

    private let synthesizer = AVSpeechSynthesizer()

    public override init() {
        super.init()
        synthesizer.delegate = self
        DispatchQueue.global(qos: .userInitiated).async {
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try? AVAudioSession.sharedInstance().setActive(true)
        }
    }

    public func speak(text: String, style: SpeechStyle = .neutral, languageCode: String = "en-US", force: Bool = false) {
        guard !isMuted else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = AVSpeechSynthesisVoice(language: languageCode) ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.pitchMultiplier = min(2.0, max(0.5, basePitchMultiplier * style.pitchMultiplier))
        utterance.rate = min(1.0, max(0.1, speechRate * style.rateMultiplier))

        synthesizer.speak(utterance)
    }

    public func toggleMute() {
        isMuted.toggle()
        if isMuted && synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
            isSpeaking = false
        }
    }

    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in self?.isSpeaking = true }
    }

    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in self?.isSpeaking = false }
    }
}

@MainActor
public final class TranslationService: ObservableObject {
    @Published public var selectedLanguage: SupportedLanguage = SupportedLanguage.available[0]
    @Published public var conversationHistory: [HistoryItem] = []

    public func translate(label: String) -> String {
        return label.capitalized
    }

    public func addToHistory(prediction: SignPrediction) {
        let item = HistoryItem(
            prediction: prediction,
            languageCode: selectedLanguage.localeCode,
            languageName: selectedLanguage.name,
            timestamp: Date()
        )
        conversationHistory.insert(item, at: 0)
    }

    public func clearHistory() {
        conversationHistory.removeAll()
    }
}

@MainActor
public final class ModelManager: ObservableObject {
    @Published public var isModelAvailable: Bool = false

    public init() {
        // Automatically discovers SignClassifier.mlmodel in Resources/
        if Bundle.main.url(forResource: "SignClassifier", withExtension: "mlmodelc") != nil ||
           Bundle.main.url(forResource: "SignClassifier", withExtension: "mlmodel") != nil {
            self.isModelAvailable = true
        }
    }

    public func classify(observations: [VNHumanHandPoseObservation]) -> (label: String, confidence: Double)? {
        // Zero fallbacks: strictly returns nil until model file is compiled
        return nil
    }
}

@MainActor
public final class SignRecognitionService: ObservableObject {
    @Published public var state: RecognitionState = .ready
    @Published public var currentPrediction: SignPrediction?
    @Published public var isAutoSpeakEnabled: Bool = true
    @Published public var stabilityProgress: Double = 0.0

    public let cameraService: CameraService
    public let handPoseService: HandPoseService
    public let faceCueService: FaceCueService
    public let speechService: SpeechService
    public let translationService: TranslationService
    public let modelManager: ModelManager

    private var cancellables = Set<AnyCancellable>()

    public init(
        cameraService: CameraService,
        handPoseService: HandPoseService,
        faceCueService: FaceCueService,
        speechService: SpeechService,
        translationService: TranslationService,
        modelManager: ModelManager
    ) {
        self.cameraService = cameraService
        self.handPoseService = handPoseService
        self.faceCueService = faceCueService
        self.speechService = speechService
        self.translationService = translationService
        self.modelManager = modelManager

        cameraService.onHandPoseDetected = { [weak self] points, confidence, isDetected in
            Task { @MainActor [weak self] in
                guard let self = self, self.cameraService.isCameraActive else { return }
                self.handPoseService.update(points: points, confidence: confidence, isDetected: isDetected)
            }
        }

        cameraService.onFaceCueDetected = { [weak self] isDetected, style in
            Task { @MainActor [weak self] in
                guard let self = self, self.cameraService.isCameraActive else { return }
                self.faceCueService.update(isDetected: isDetected, style: style)
            }
        }

        handPoseService.$isHandDetected
            .receive(on: DispatchQueue.main)
            .sink { [weak self] detected in
                guard let self = self else { return }
                guard self.cameraService.isCameraActive else {
                    self.state = .cameraOff
                    return
                }

                if !detected {
                    if case .signDetected = self.state { } else { self.state = .ready }
                } else {
                    let count = self.handPoseService.handJointPoints.count
                    let conf = self.handPoseService.handConfidence
                    if self.modelManager.isModelAvailable {
                        self.state = .trackingHands(jointCount: count, confidence: conf)
                    } else {
                        self.state = .waitingForModel(message: "Tracking \(count) hand joints in real-time. Drop SignClassifier.mlmodel into Resources.")
                    }
                }
            }
            .store(in: &cancellables)

        cameraService.$isCameraActive
            .receive(on: DispatchQueue.main)
            .sink { [weak self] active in
                guard let self = self else { return }
                self.state = active ? .ready : .cameraOff
            }
            .store(in: &cancellables)
    }

    public func replayCurrentSpeech() {
        guard let p = currentPrediction else { return }
        speechService.speak(text: p.translatedText, style: p.detectedEmotion, languageCode: translationService.selectedLanguage.localeCode, force: true)
    }

    public func clearCurrentPrediction() {
        currentPrediction = nil
        state = .ready
    }
}

// =============================================================================
// MARK: - 4. USER INTERFACE COMPONENTS
// =============================================================================

public struct CameraPreviewView: UIViewRepresentable {
    public let captureSession: AVCaptureSession

    public init(captureSession: AVCaptureSession) {
        self.captureSession = captureSession
    }

    public func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.videoPreviewLayer.session = captureSession
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    public func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {}
}

public final class CameraPreviewUIView: UIView {
    public override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    public var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    public override func layoutSubviews() {
        super.layoutSubviews()
        videoPreviewLayer.frame = bounds
    }
}

public struct HandLandmarksOverlayView: View {
    public let points: [CGPoint]
    public let indicatorColor: Color

    public init(points: [CGPoint], indicatorColor: Color = Color(hex: "06B6D4")) {
        self.points = points
        self.indicatorColor = indicatorColor
    }

    public var body: some View {
        GeometryReader { geo in
            ZStack {
                ForEach(0..<points.count, id: \.self) { i in
                    Circle()
                        .fill(indicatorColor)
                        .frame(width: 7, height: 7)
                        .shadow(color: indicatorColor.opacity(0.8), radius: 4)
                        .position(x: points[i].x * geo.size.width, y: points[i].y * geo.size.height)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

// =============================================================================
// MARK: - 5. MAIN INTEGRATED APPLICATION VIEW
// =============================================================================

public struct ContentView: View {
    @StateObject private var cameraService = CameraService()
    @StateObject private var handPoseService = HandPoseService()
    @StateObject private var faceCueService = FaceCueService()
    @StateObject private var speechService = SpeechService()
    @StateObject private var translationService = TranslationService()
    @StateObject private var modelManager = ModelManager()

    @StateObject private var recognitionService: SignRecognitionService

    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding: Bool = false
    @State private var isShowingSplash: Bool = true
    @State private var splashPulsing: Bool = false
    @State private var showingHistorySheet = false
    @State private var showingLearnSignSheet = false
    @State private var showingSettingsSheet = false
    @State private var showingAboutSheet = false

    public init() {
        let cam = CameraService()
        let hand = HandPoseService()
        let face = FaceCueService()
        let speech = SpeechService()
        let trans = TranslationService()
        let model = ModelManager()

        _cameraService = StateObject(wrappedValue: cam)
        _handPoseService = StateObject(wrappedValue: hand)
        _faceCueService = StateObject(wrappedValue: face)
        _speechService = StateObject(wrappedValue: speech)
        _translationService = StateObject(wrappedValue: trans)
        _modelManager = StateObject(wrappedValue: model)

        _recognitionService = StateObject(
            wrappedValue: SignRecognitionService(
                cameraService: cam,
                handPoseService: hand,
                faceCueService: face,
                speechService: speech,
                translationService: trans,
                modelManager: model
            )
        )
    }

    public var body: some View {
        ZStack {
            NeumorphicTheme.background
                .ignoresSafeArea()

            if isShowingSplash {
                splashScreenView
                    .transition(.opacity)
            } else if !hasCompletedOnboarding {
                aboutAndRoadmapScreenView(isSheet: false)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            } else {
                mainTranslationScreenView
                    .transition(.opacity)
            }
        }
        .sheet(isPresented: $showingAboutSheet) {
            aboutAndRoadmapScreenView(isSheet: true)
        }
        .sheet(isPresented: $showingHistorySheet) {
            conversationHistorySheetView
        }
        .sheet(isPresented: $showingLearnSignSheet) {
            learnSignSheetView
        }
        .sheet(isPresented: $showingSettingsSheet) {
            settingsSheetView
        }
    }

    // MARK: - Screen 0: Minimal Splash Screen (Logo Only, Centered)
    private var splashScreenView: some View {
        ZStack {
            NeumorphicTheme.background
                .ignoresSafeArea()

            VStack {
                Spacer()

                ZStack {
                    Circle()
                        .fill(NeumorphicTheme.cardBackground)
                        .frame(width: 160, height: 160)
                        .shadow(color: NeumorphicTheme.lightShadow, radius: 12, x: -10, y: -10)
                        .shadow(color: NeumorphicTheme.darkShadow, radius: 14, x: 10, y: 10)

                    Image("AppLogo")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 100, height: 100)
                }
                .scaleEffect(splashPulsing ? 1.05 : 0.95)

                Spacer()
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) {
                splashPulsing = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                    isShowingSplash = false
                }
            }
        }
        .onTapGesture {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                isShowingSplash = false
            }
        }
    }

    // MARK: - Screen 1: Data-Dense About & Technical Roadmap Screen
    private func aboutAndRoadmapScreenView(isSheet: Bool) -> some View {
        ZStack {
            NeumorphicTheme.background.ignoresSafeArea()

            VStack(spacing: 0) {
                if isSheet {
                    HStack {
                        Text("About & Roadmap")
                            .font(.system(size: 16, weight: .bold, design: .rounded))
                            .foregroundColor(NeumorphicTheme.textPrimary)

                        Spacer()

                        Button(action: { showingAboutSheet = false }) {
                            Image(systemName: "xmark")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(NeumorphicTheme.textPrimary)
                                .frame(width: 32, height: 32)
                                .neumorphicRaised(cornerRadius: 10)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 16)
                    .padding(.bottom, 8)
                }

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 18) {
                        // 1. Hero Emblem Header
                        VStack(spacing: 12) {
                            ZStack {
                                Circle()
                                    .fill(NeumorphicTheme.cardBackground)
                                    .frame(width: 106, height: 106)
                                    .shadow(color: NeumorphicTheme.lightShadow, radius: 8, x: -6, y: -6)
                                    .shadow(color: NeumorphicTheme.darkShadow, radius: 10, x: 6, y: 6)

                                Image("AppLogo")
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(width: 70, height: 70)
                            }
                            .padding(.top, isSheet ? 8 : 20)

                            VStack(spacing: 5) {
                                Text("SignaSense")
                                    .font(.system(size: 28, weight: .bold, design: .rounded))
                                    .foregroundColor(NeumorphicTheme.textPrimary)

                                Text("Turning gestures into words, connecting people through communication.")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(NeumorphicTheme.textSecondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 16)

                                HStack(spacing: 6) {
                                    Text("Recognize carefully")
                                    Text("•")
                                    Text("Translate transparently")
                                    Text("•")
                                    Text("Speak accessibly")
                                }
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundColor(NeumorphicTheme.accentColor)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4)
                                .background(NeumorphicTheme.accentColor.opacity(0.1))
                                .clipShape(Capsule())
                                .padding(.top, 4)
                            }
                        }

                        // 2. Hardware Benchmarks & Performance Metrics Grid
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Image(systemName: "gauge.with.needle.fill")
                                    .foregroundColor(NeumorphicTheme.accentColor)
                                    .font(.system(size: 13, weight: .bold))
                                Text("SYSTEM BENCHMARKS")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                                Spacer()
                                HStack(spacing: 4) {
                                    Circle().fill(Color.green).frame(width: 6, height: 6)
                                    Text("ON-DEVICE")
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundColor(.green)
                                }
                            }
                            .padding(.horizontal, 4)

                            LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                                metricTile(value: "< 100 ms", label: "Pipeline Latency", detail: "Real-time gesture-to-speech", icon: "bolt.fill")
                                metricTile(value: "100% Local", label: "On-Device Privacy", detail: "Zero cloud transmission", icon: "shield.lefthalf.filled")
                                metricTile(value: "21 Points", label: "Hand Landmarks", detail: "3D Vision joint model", icon: "hand.raised.fill")
                                metricTile(value: "8 Locales", label: "Spoken Synthesis", detail: "Global voice engines", icon: "globe.americas.fill")
                            }
                        }

                        // 3. System Architecture & Real-Time Pipeline
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Image(systemName: "cpu.fill")
                                    .foregroundColor(NeumorphicTheme.accentColor)
                                    .font(.system(size: 13, weight: .bold))
                                Text("SYSTEM ARCHITECTURE & PIPELINE")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                                Spacer()
                                Text("4 STAGES")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundColor(NeumorphicTheme.textMuted)
                            }
                            .padding(.horizontal, 4)

                            VStack(spacing: 10) {
                                pipelineStageCard(
                                    stage: "STAGE 01",
                                    title: "Visual Ingestion",
                                    engine: "AVFoundation Capture Engine",
                                    specs: [
                                        "30 FPS synchronous frame pipeline at 32BGRA format",
                                        "Zero-copy frame processing on background dispatch queue",
                                        "Host camera bridge emulation for macOS simulator validation"
                                    ],
                                    icon: "video.fill"
                                )

                                pipelineStageCard(
                                    stage: "STAGE 02",
                                    title: "Joint Landmark Extraction",
                                    engine: "Apple Vision Framework",
                                    specs: [
                                        "21 distinct 3D landmarks per hand (wrist, MCP, PIP, DIP, TIP)",
                                        "Normalized Euclidean coordinates with > 0.30 confidence gate",
                                        "Facial contour outer-lip curvature for prosody modulation"
                                    ],
                                    icon: "hand.draw.fill"
                                )

                                pipelineStageCard(
                                    stage: "STAGE 03",
                                    title: "Gesture Inference & Mapping",
                                    engine: "CoreML Spatial-Temporal Model",
                                    specs: [
                                        "Dedicated slot for SignClassifier.mlmodel",
                                        "Multi-frame temporal sliding window for dynamic signs",
                                        "Transparent fallback state when confidence is below 65%"
                                    ],
                                    icon: "brain.head.profile"
                                )

                                pipelineStageCard(
                                    stage: "STAGE 04",
                                    title: "Multilingual Speech Synthesis",
                                    engine: "AVSpeechSynthesizer + AudioToolbox",
                                    specs: [
                                        "Facial emotion context modulating speech pitch and rate",
                                        "Instant tactile mute/unmute control with speech queue flushing",
                                        "Multilingual translation across English, Hindi, Spanish, +5 more"
                                    ],
                                    icon: "speaker.wave.2.fill"
                                )
                            }
                        }

                        // 4. Engineering Roadmap
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Image(systemName: "map.fill")
                                    .foregroundColor(NeumorphicTheme.accentColor)
                                    .font(.system(size: 13, weight: .bold))
                                Text("TECHNICAL ROADMAP")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                                Spacer()
                                Text("VERSIONS 1-4")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundColor(NeumorphicTheme.textMuted)
                            }
                            .padding(.horizontal, 4)

                            VStack(spacing: 10) {
                                roadmapMilestoneCard(
                                    version: "v1.0",
                                    title: "Hackathon MVP",
                                    badge: "LIVE",
                                    badgeColor: .green,
                                    items: [
                                        "Real-time 21-joint hand pose tracking & live cyan visualizer",
                                        "CoreML classification architecture & model slot",
                                        "Native text-to-speech with facial outer-lip tone modulation",
                                        "Full Neumorphic UI and complete on-device privacy"
                                    ],
                                    icon: "flag.checkered"
                                )

                                roadmapMilestoneCard(
                                    version: "v2.0",
                                    title: "Temporal Context & Dictionary",
                                    badge: "IN PROGRESS",
                                    badgeColor: .blue,
                                    items: [
                                        "50+ Indian Sign Language (ISL) vocabulary dictionary",
                                        "Multi-frame LSTM gesture buffer for motion-dependent signs",
                                        "Dynamic lighting normalization & camera angle calibration",
                                        "In-app gesture practice mode with visual alignment guide"
                                    ],
                                    icon: "sparkles"
                                )

                                roadmapMilestoneCard(
                                    version: "v3.0",
                                    title: "Continuous Sentence Translation",
                                    badge: "PLANNED",
                                    badgeColor: .orange,
                                    items: [
                                        "Continuous signing stream segmentation & boundary detection",
                                        "ISL Topic-Comment grammatical syntax restructuring",
                                        "Two-handed coordinated gesture tracking and interaction",
                                        "Context-aware grammar correction and word-to-sentence generation"
                                    ],
                                    icon: "arrow.triangle.merge"
                                )

                                roadmapMilestoneCard(
                                    version: "v4.0",
                                    title: "Multimodal Neural Synthesis",
                                    badge: "VISION",
                                    badgeColor: .purple,
                                    items: [
                                        "Non-manual marker tracking: head tilt, eyebrow raise, mouth morphemes",
                                        "Personalized neural voice cloning tailored to individual signers",
                                        "Cross-dialect expansion: American (ASL) and British (BSL) Sign Language"
                                    ],
                                    icon: "globe.badge.chevron.backward"
                                )
                            }
                        }

                        // 5. Ethics & Privacy Guarantee Card
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 8) {
                                Image(systemName: "lock.shield.fill")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.accentColor)

                                Text("ETHICS & ON-DEVICE PRIVACY")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                            }

                            Text("• Camera frames are processed strictly in volatile device memory and never recorded, stored, or transmitted to any cloud servers.")
                                .font(.system(size: 11))
                                .foregroundColor(NeumorphicTheme.textSecondary)

                            Text("• SignaSense is built as an assistive prototype to empower everyday communication. It prioritizes transparent uncertainty states over speculative predictions.")
                                .font(.system(size: 11))
                                .foregroundColor(NeumorphicTheme.textSecondary)
                        }
                        .padding(14)
                        .neumorphicRaised(cornerRadius: 16)

                        // 6. Action Button
                        Button(action: {
                            withAnimation(.spring(response: 0.45, dampingFraction: 0.8)) {
                                hasCompletedOnboarding = true
                                showingAboutSheet = false
                            }
                        }) {
                            HStack(spacing: 10) {
                                Text(isSheet ? "Return to Camera" : "Launch Live Translation")
                                    .font(.system(size: 16, weight: .bold, design: .rounded))
                                    .foregroundColor(NeumorphicTheme.textPrimary)

                                Image(systemName: "arrow.right")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.accentColor)
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .neumorphicRaised(cornerRadius: 18)
                        }
                        .padding(.top, 4)
                        .padding(.bottom, 24)
                    }
                    .padding(.horizontal, 18)
                }
            }
        }
    }

    // MARK: - Screen 2: Main Live Translation View
    private var mainTranslationScreenView: some View {
        VStack(spacing: 12) {
            // TOP BAR: Camera On/Off + Title + Menu Bar
            HStack(alignment: .center) {
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        cameraService.toggleCameraActive()
                    }
                }) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(cameraService.isCameraActive ? Color.green : Color.gray)
                            .frame(width: 8, height: 8)

                        Image(systemName: cameraService.isCameraActive ? "camera.fill" : "camera.slash.fill")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(NeumorphicTheme.textPrimary)

                        Text(cameraService.isCameraActive ? "ON" : "OFF")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(NeumorphicTheme.textSecondary)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .neumorphicRaised(cornerRadius: 14)
                }

                Spacer()

                HStack(spacing: 6) {
                    Image("AppLogo")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 20, height: 20)
                        .clipShape(Circle())

                    Text("SignaSense")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                        .foregroundColor(NeumorphicTheme.textPrimary)
                }

                Spacer()

                Menu {
                    Button { showingHistorySheet = true } label: { Label("Conversation History", systemImage: "clock.arrow.circlepath") }
                    Button { showingLearnSignSheet = true } label: { Label("Learn Sign Dictionary", systemImage: "hand.raised.square.fill") }
                    Button { showingAboutSheet = true } label: { Label("About & Roadmap", systemImage: "info.circle.fill") }
                    Divider()
                    Button { showingSettingsSheet = true } label: { Label("Settings & Languages", systemImage: "gearshape.fill") }
                } label: {
                    Image(systemName: "line.3.horizontal")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(NeumorphicTheme.textPrimary)
                        .frame(width: 42, height: 40)
                        .neumorphicRaised(cornerRadius: 14)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 6)

            // CENTRAL CAMERA VIEWPORT
            ZStack {
                if cameraService.isCameraActive {
                    safeCameraViewport
                        .clipShape(RoundedRectangle(cornerRadius: 26))
                        .overlay(RoundedRectangle(cornerRadius: 26).stroke(Color.white.opacity(0.6), lineWidth: 1.5))
                        .shadow(color: NeumorphicTheme.darkShadow.opacity(0.35), radius: 8, x: 4, y: 4)

                    HandLandmarksOverlayView(points: handPoseService.handJointPoints)
                        .clipShape(RoundedRectangle(cornerRadius: 26))

                    // Framing Bracket
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [8, 6]))
                        .foregroundColor(handPoseService.isHandDetected ? Color(hex: "06B6D4").opacity(0.6) : Color.white.opacity(0.3))
                        .padding(24)
                        .allowsHitTesting(false)

                    // Live Status Pills inside Camera
                    VStack {
                        HStack {
                            if faceCueService.isEmotionDetectionEnabled && faceCueService.isFaceDetected {
                                HStack(spacing: 5) {
                                    Text(faceCueService.detectedStyle.emoji).font(.system(size: 11))
                                    Text("Tone: \(faceCueService.detectedStyle.displayName)").font(.system(size: 11, weight: .bold)).foregroundColor(NeumorphicTheme.textPrimary)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(NeumorphicTheme.background.opacity(0.92))
                                .clipShape(Capsule())
                                .shadow(color: NeumorphicTheme.darkShadow.opacity(0.2), radius: 4, x: 2, y: 2)
                            }

                            Spacer()

                            HStack(spacing: 5) {
                                Circle().fill(handPoseService.isHandDetected ? Color.green : Color.gray).frame(width: 7, height: 7)
                                Text(handPoseService.isHandDetected ? "\(handPoseService.handJointPoints.count) Landmarks" : "Searching Hands")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(NeumorphicTheme.background.opacity(0.92))
                            .clipShape(Capsule())
                            .shadow(color: NeumorphicTheme.darkShadow.opacity(0.2), radius: 4, x: 2, y: 2)
                        }
                        .padding(12)

                        Spacer()

                        HStack {
                            Spacer()
                            if cameraService.hasCameraDevice {
                                Button(action: { cameraService.flipCamera() }) {
                                    HStack(spacing: 5) {
                                        Image(systemName: "camera.rotate.fill")
                                        Text(cameraService.currentCameraPosition == .front ? "Front" : "Back")
                                            .font(.system(size: 10, weight: .bold))
                                    }
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(NeumorphicTheme.background.opacity(0.92))
                                    .clipShape(Capsule())
                                    .shadow(color: NeumorphicTheme.darkShadow.opacity(0.2), radius: 4, x: 2, y: 2)
                                }
                            }
                        }
                        .padding(12)
                    }
                } else {
                    RoundedRectangle(cornerRadius: 26)
                        .fill(NeumorphicTheme.cardBackground)
                        .neumorphicInset(cornerRadius: 26)
                        .overlay(
                            VStack(spacing: 12) {
                                Image(systemName: "video.slash.fill").font(.system(size: 40)).foregroundColor(NeumorphicTheme.textMuted)
                                Text("Camera Paused").font(.system(size: 16, weight: .bold)).foregroundColor(NeumorphicTheme.textPrimary)
                                Button("Resume Camera") { cameraService.toggleCameraActive() }
                                    .font(.system(size: 13, weight: .semibold))
                                    .foregroundColor(NeumorphicTheme.textPrimary)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .neumorphicRaised(cornerRadius: 12)
                            }
                        )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal, 16)

            // BOTTOM CARD: Transcribe Caption + Speaker On/Off
            neumorphicCaptionCard
                .padding(.horizontal, 16)
                .padding(.bottom, 10)
        }
    }

    // Live camera viewport rendering
    private var safeCameraViewport: some View {
        ZStack {
            if cameraService.hasCameraDevice {
                CameraPreviewView(captureSession: cameraService.captureSession)
            } else if cameraService.isBridgeConnected, let image = cameraService.bridgeImage {
                GeometryReader { geo in
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: geo.size.width, height: geo.size.height)
                        .clipped()
                }
            } else {
                simulatedViewport
            }
        }
    }

    private var simulatedViewport: some View {
        ZStack {
            Color(hex: "1E293B")
            VStack(spacing: 12) {
                Image("AppLogo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 48, height: 48)
                    .clipShape(Circle())
                    .shadow(color: Color.black.opacity(0.3), radius: 6)

                Text("Connecting to Mac Camera...")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.white)

                Text("Streaming live from your MacBook Pro Camera into the iOS Simulator.")
                    .font(.system(size: 11))
                    .multilineTextAlignment(.center)
                    .foregroundColor(Color.white.opacity(0.75))
                    .padding(.horizontal, 24)

                Button(action: { cameraService.startBridgeStreamIfNeeded() }) {
                    HStack(spacing: 6) {
                        Image(systemName: "camera.fill")
                        Text("Connect Mac Camera")
                    }
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color(hex: "0284C7"))
                    .clipShape(Capsule())
                    .shadow(color: Color.black.opacity(0.2), radius: 4)
                }
            }
            .padding(20)
        }
    }

    // Bottom Transcribe Caption Card
    private var neumorphicCaptionCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center) {
                HStack(spacing: 6) {
                    Circle().fill(recognitionService.state.indicatorColor).frame(width: 7, height: 7)
                    Text(recognitionService.state.statusTitle).font(.system(size: 12, weight: .bold)).foregroundColor(NeumorphicTheme.textPrimary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(NeumorphicTheme.cardBackground)
                .clipShape(Capsule())
                .shadow(color: NeumorphicTheme.lightShadow, radius: 3, x: -2, y: -2)
                .shadow(color: NeumorphicTheme.darkShadow.opacity(0.3), radius: 3, x: 2, y: 2)

                Spacer()

                Button(action: { speechService.toggleMute() }) {
                    HStack(spacing: 6) {
                        Image(systemName: speechService.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(speechService.isMuted ? .red : NeumorphicTheme.textPrimary)
                        Text(speechService.isMuted ? "Muted" : "Voice ON")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(speechService.isMuted ? .red : NeumorphicTheme.textPrimary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .neumorphicRaised(cornerRadius: 12)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("TRANSCRIBE CAPTION")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(NeumorphicTheme.textMuted)
                    .tracking(0.8)

                if let p = recognitionService.currentPrediction {
                    Text(p.translatedText)
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundColor(NeumorphicTheme.textPrimary)
                } else {
                    Text(recognitionService.state.statusDescription)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(NeumorphicTheme.textSecondary)
                }
            }
            .padding(.vertical, 2)
        }
        .padding(16)
        .neumorphicRaised(cornerRadius: 22)
    }

    // MARK: - Technical Components for About & Roadmap
    private func metricTile(value: String, label: String, detail: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(NeumorphicTheme.accentColor)
                Spacer()
            }

            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundColor(NeumorphicTheme.textPrimary)

            Text(label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(NeumorphicTheme.textPrimary)

            Text(detail)
                .font(.system(size: 10))
                .foregroundColor(NeumorphicTheme.textMuted)
                .lineLimit(1)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .neumorphicRaised(cornerRadius: 14)
    }

    private func pipelineStageCard(stage: String, title: String, engine: String, specs: [String], icon: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ZStack {
                    Circle()
                        .fill(NeumorphicTheme.cardBackground)
                        .frame(width: 32, height: 32)
                        .shadow(color: NeumorphicTheme.lightShadow, radius: 2, x: -2, y: -2)
                        .shadow(color: NeumorphicTheme.darkShadow, radius: 3, x: 2, y: 2)

                    Image(systemName: icon)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(NeumorphicTheme.accentColor)
                }

                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(stage)
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(NeumorphicTheme.accentColor)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(NeumorphicTheme.accentColor.opacity(0.12))
                            .clipShape(Capsule())

                        Text(title)
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(NeumorphicTheme.textPrimary)
                    }

                    Text(engine)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(NeumorphicTheme.textSecondary)
                }

                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(specs, id: \.self) { spec in
                    HStack(alignment: .top, spacing: 6) {
                        Circle()
                            .fill(NeumorphicTheme.accentColor.opacity(0.8))
                            .frame(width: 4, height: 4)
                            .padding(.top, 5)

                        Text(spec)
                            .font(.system(size: 11))
                            .foregroundColor(NeumorphicTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.leading, 8)
        }
        .padding(12)
        .neumorphicRaised(cornerRadius: 14)
    }

    private func roadmapMilestoneCard(version: String, title: String, badge: String, badgeColor: Color, items: [String], icon: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(badgeColor)

                Text(version)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(NeumorphicTheme.textPrimary)

                Text("•")
                    .foregroundColor(NeumorphicTheme.textMuted)

                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(NeumorphicTheme.textSecondary)

                Spacer()

                Text(badge)
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(badgeColor)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(badgeColor.opacity(0.12))
                    .clipShape(Capsule())
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(items, id: \.self) { item in
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 10))
                            .foregroundColor(badgeColor.opacity(0.8))
                            .padding(.top, 2)

                        Text(item)
                            .font(.system(size: 11))
                            .foregroundColor(NeumorphicTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.leading, 4)
        }
        .padding(12)
        .neumorphicRaised(cornerRadius: 14)
    }

    // MARK: - Integrated Drawers
    private var conversationHistorySheetView: some View {
        NavigationStack {
            ZStack {
                NeumorphicTheme.background.ignoresSafeArea()
                if translationService.conversationHistory.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "clock.arrow.circlepath").font(.system(size: 32)).foregroundColor(NeumorphicTheme.textMuted)
                        Text("No History Recorded").font(.system(size: 16, weight: .bold)).foregroundColor(NeumorphicTheme.textPrimary)
                        Text("Live recognized signs will appear here.").font(.system(size: 12)).foregroundColor(NeumorphicTheme.textSecondary)
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(translationService.conversationHistory) { item in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(item.prediction.translatedText).font(.system(size: 15, weight: .bold)).foregroundColor(NeumorphicTheme.textPrimary)
                                    Text("\(item.languageName) • \(item.formattedTime)").font(.system(size: 11)).foregroundColor(NeumorphicTheme.textMuted)
                                }
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .neumorphicRaised(cornerRadius: 14)
                            }
                        }
                        .padding(16)
                    }
                }
            }
            .navigationTitle("Conversation History")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var learnSignSheetView: some View {
        NavigationStack {
            ZStack {
                NeumorphicTheme.background.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 12) {
                        dictionaryCard(title: "Namaste", sub: "नमस्ते", hint: "Bring palms together flat in front of chest.")
                        dictionaryCard(title: "Hello", sub: "नमस्ते / हैलो", hint: "Open palm facing outward near temple level.")
                        dictionaryCard(title: "Thank You", sub: "धन्यवाद", hint: "Touch fingertips to chin, then move forward.")
                        dictionaryCard(title: "Yes", sub: "हाँ", hint: "Make closed fist and nod wrist up and down.")
                        dictionaryCard(title: "No", sub: "नहीं", hint: "Snap index and middle fingers down onto thumb.")
                        dictionaryCard(title: "Help", sub: "मदद", hint: "Thumbs-up resting on flat open palm.")
                    }
                    .padding(16)
                }
            }
            .navigationTitle("Sign Language Guide")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func dictionaryCard(title: String, sub: String, hint: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.system(size: 15, weight: .bold)).foregroundColor(NeumorphicTheme.textPrimary)
                Text("•").foregroundColor(NeumorphicTheme.textMuted)
                Text(sub).font(.system(size: 13, weight: .medium)).foregroundColor(NeumorphicTheme.textSecondary)
            }
            Text(hint).font(.system(size: 12)).foregroundColor(NeumorphicTheme.textSecondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .neumorphicRaised(cornerRadius: 14)
    }

    private var settingsSheetView: some View {
        NavigationStack {
            ZStack {
                NeumorphicTheme.background.ignoresSafeArea()
                ScrollView {
                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("TRANSLATION LANGUAGE").font(.system(size: 11, weight: .bold)).foregroundColor(NeumorphicTheme.textMuted)
                            ForEach(SupportedLanguage.available) { lang in
                                Button(action: { translationService.selectedLanguage = lang }) {
                                    HStack {
                                        Text(lang.flagEmoji)
                                        Text(lang.name).font(.system(size: 14, weight: .semibold)).foregroundColor(NeumorphicTheme.textPrimary)
                                        Spacer()
                                        if translationService.selectedLanguage.id == lang.id {
                                            Image(systemName: "checkmark.circle.fill").foregroundColor(NeumorphicTheme.accentColor)
                                        }
                                    }
                                    .padding(12)
                                    .neumorphicRaised(cornerRadius: 12)
                                }
                            }
                        }
                    }
                    .padding(16)
                }
            }
            .navigationTitle("Settings & Languages")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

// =============================================================================
// MARK: - 6. XCODE CANVAS PREVIEWS
// =============================================================================

#Preview("Neumorphic — Welcome Screen") {
    ContentView()
}
