import Foundation
import AVFoundation
import CoreMedia
import Network
import CoreImage

final class CameraBridge: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.queue", qos: .userInteractive)
    private let lock = NSLock()
    private var currentJPEGData: Data?

    func start() -> Bool {
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("No video device found")
            return false
        }

        session.beginConfiguration()
        session.sessionPreset = .medium
        if session.canAddInput(input) { session.addInput(input) }
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
        session.startRunning()
        print("Camera bridge capture session started on device:", device.localizedName)
        return true
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(imageBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly) }

        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext()
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let jpeg = context.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: [:]) else { return }

        lock.lock()
        currentJPEGData = jpeg
        lock.unlock()
    }

    func getLatestFrame() -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return currentJPEGData
    }
}

let bridge = CameraBridge()
guard bridge.start() else {
    exit(1)
}

let port = NWEndpoint.Port(rawValue: 8765)!
let listener = try NWListener(using: .tcp, on: port)

listener.newConnectionHandler = { connection in
    connection.start(queue: .global())
    connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { data, _, _, _ in
        guard let data = data, let req = String(data: data, encoding: .utf8) else {
            connection.cancel()
            return
        }

        if req.contains("GET /frame") {
            if let frame = bridge.getLatestFrame() {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: \(frame.count)\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                var response = header.data(using: .utf8)!
                response.append(frame)
                connection.send(content: response, completion: .contentProcessed({ _ in connection.cancel() }))
            } else {
                let notReady = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
                connection.send(content: notReady.data(using: .utf8)!, completion: .contentProcessed({ _ in connection.cancel() }))
            }
        } else {
            let ok = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\nOK"
            connection.send(content: ok.data(using: .utf8)!, completion: .contentProcessed({ _ in connection.cancel() }))
        }
    }
}

listener.start(queue: .main)
print("Bridge HTTP server listening on http://127.0.0.1:8765")
RunLoop.main.run()
