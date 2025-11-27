import Foundation

struct AlertsResponse: Codable {
    let alerts: [WatchlistAlert]
    let count: Int
    let watchlistEnabled: Bool
    let watchlistNames: [String]?

    enum CodingKeys: String, CodingKey {
        case alerts
        case count
        case watchlistEnabled = "watchlist_enabled"
        case watchlistNames = "watchlist_names"
    }
}

struct WatchlistAlert: Codable, Identifiable {
    let id: Int
    let name: String
    let similarity: Double
    let photoPath: String?
    let detectedAt: Double
    let detectedAtIso: String
    let acknowledged: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case similarity
        case photoPath = "photo_path"
        case detectedAt = "detected_at"
        case detectedAtIso = "detected_at_iso"
        case acknowledged
    }

    var formattedTime: String {
        let date = Date(timeIntervalSince1970: detectedAt)
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}
