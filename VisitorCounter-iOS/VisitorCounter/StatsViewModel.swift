import Foundation
import Combine

class StatsViewModel: ObservableObject {
    @Published var stats: StatsResponse?
    @Published var isConnected = false
    @Published var errorMessage: String?
    @Published var serverIP: String = "192.168.1.100"  // Change to your server IP

    private var timer: Timer?

    var grandTotal: Int {
        let local = stats?.uniqueVisitors ?? 0
        let peer = stats?.peerData?.uniqueVisitors ?? 0
        return local + peer
    }

    var flowRate: Int {
        // Simple approximation - entries per hour since app started
        guard let stats = stats else { return 0 }
        return stats.totalIn
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
        guard let url = URL(string: "http://\(serverIP):8000/stats") else {
            errorMessage = "Invalid URL"
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
