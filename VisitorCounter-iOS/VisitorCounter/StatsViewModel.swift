import Foundation
import Combine
import UserNotifications

class StatsViewModel: ObservableObject {
    @Published var stats: StatsResponse?
    @Published var isConnected = false
    @Published var errorMessage: String?
    @Published var alerts: [WatchlistAlert] = []
    @Published var unreadAlertCount = 0
    @Published var serverAddress: String = "" {
        didSet {
            UserDefaults.standard.set(serverAddress, forKey: "serverAddress")
        }
    }

    private var timer: Timer?
    private var seenAlertIds: Set<Int> = []

    init() {
        // Load saved address
        if let saved = UserDefaults.standard.string(forKey: "serverAddress"), !saved.isEmpty {
            serverAddress = saved
        }

        // Request notification permission
        requestNotificationPermission()
    }

    var grandTotal: Int {
        // Use known_faces which is synced across all stations
        return stats?.knownFaces ?? stats?.uniqueVisitors ?? 0
    }

    var flowRate: Int {
        guard let stats = stats else { return 0 }
        return stats.totalIn
    }

    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                print("Notification permission granted")
            }
        }
    }

    /// Build base URL from user input
    private func buildBaseURL() -> String? {
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
            return input
        }

        // If it's just an IP or hostname
        // Check if port is included
        if input.contains(":") {
            return "http://\(input)"
        } else {
            // Default to port 8000
            return "http://\(input):8000"
        }
    }

    private func buildStatsURL() -> URL? {
        guard let base = buildBaseURL() else { return nil }
        return URL(string: "\(base)/stats")
    }

    private func buildAlertsURL() -> URL? {
        guard let base = buildBaseURL() else { return nil }
        return URL(string: "\(base)/alerts")
    }

    func startPolling() {
        fetchStats()
        fetchAlerts()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.fetchStats()
            self?.fetchAlerts()
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

    func fetchAlerts() {
        guard let url = buildAlertsURL() else { return }

        var request = URLRequest(url: url)
        request.timeoutInterval = 5

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let data = data, error == nil else { return }

            do {
                let decoder = JSONDecoder()
                let response = try decoder.decode(AlertsResponse.self, from: data)

                DispatchQueue.main.async {
                    self?.alerts = response.alerts
                    self?.unreadAlertCount = response.alerts.filter { !$0.acknowledged }.count

                    // Check for new alerts and send notifications
                    for alert in response.alerts {
                        if !(self?.seenAlertIds.contains(alert.id) ?? true) {
                            self?.seenAlertIds.insert(alert.id)
                            self?.sendNotification(for: alert)
                        }
                    }
                }
            } catch {
                print("Failed to decode alerts: \(error)")
            }
        }.resume()
    }

    private func sendNotification(for alert: WatchlistAlert) {
        let content = UNMutableNotificationContent()
        content.title = "⚠️ Watchlist Alert"
        content.body = "\(alert.name) detected at \(alert.formattedTime)"
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: "alert-\(alert.id)",
            content: content,
            trigger: nil // Deliver immediately
        )

        UNUserNotificationCenter.current().add(request)
    }

    func acknowledgeAlert(_ alertId: Int) {
        guard let base = buildBaseURL() else { return }
        guard let url = URL(string: "\(base)/alerts/\(alertId)/acknowledge") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 5

        URLSession.shared.dataTask(with: request) { [weak self] _, _, _ in
            DispatchQueue.main.async {
                self?.fetchAlerts()
            }
        }.resume()
    }

    func acknowledgeAllAlerts() {
        guard let base = buildBaseURL() else { return }
        guard let url = URL(string: "\(base)/alerts/acknowledge-all") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 5

        URLSession.shared.dataTask(with: request) { [weak self] _, _, _ in
            DispatchQueue.main.async {
                self?.fetchAlerts()
            }
        }.resume()
    }
}
