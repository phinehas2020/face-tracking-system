import Foundation
import Combine

class StatsViewModel: ObservableObject {
    @Published var stats: StatsResponse?
    @Published var isConnected = false
    @Published var errorMessage: String?
    @Published var serverAddress: String = "" {
        didSet {
            UserDefaults.standard.set(serverAddress, forKey: "serverAddress")
        }
    }

    private var timer: Timer?

    init() {
        // Load saved address
        if let saved = UserDefaults.standard.string(forKey: "serverAddress"), !saved.isEmpty {
            serverAddress = saved
        }
    }

    var grandTotal: Int {
        let local = stats?.uniqueVisitors ?? 0
        let peer = stats?.peerData?.uniqueVisitors ?? 0
        return local + peer
    }

    var flowRate: Int {
        guard let stats = stats else { return 0 }
        return stats.totalIn
    }

    /// Build the stats URL from user input
    /// Accepts: IP address (192.168.1.100), IP:port (192.168.1.100:8000),
    /// or full URL (https://xyz.trycloudflare.com)
    private func buildStatsURL() -> URL? {
        var input = serverAddress.trimmingCharacters(in: .whitespacesAndNewlines)

        if input.isEmpty {
            return nil
        }

        // If it's already a full URL
        if input.hasPrefix("http://") || input.hasPrefix("https://") {
            // Remove trailing slash if present
            if input.hasSuffix("/") {
                input = String(input.dropLast())
            }
            return URL(string: "\(input)/stats")
        }

        // If it's just an IP or hostname
        // Check if port is included
        if input.contains(":") {
            return URL(string: "http://\(input)/stats")
        } else {
            // Default to port 8000
            return URL(string: "http://\(input):8000/stats")
        }
    }

    func startPolling() {
        fetchStats()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.fetchStats()
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func fetchStats() {
        guard let url = buildStatsURL() else {
            errorMessage = "Enter a server address"
            isConnected = false
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 5

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.isConnected = false
                    self?.errorMessage = error.localizedDescription
                    return
                }

                guard let data = data else {
                    self?.isConnected = false
                    self?.errorMessage = "No data received"
                    return
                }

                do {
                    let decoder = JSONDecoder()
                    let stats = try decoder.decode(StatsResponse.self, from: data)
                    self?.stats = stats
                    self?.isConnected = true
                    self?.errorMessage = nil
                } catch {
                    self?.isConnected = false
                    self?.errorMessage = "Decode error: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
}
