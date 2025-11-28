import Foundation

struct StatsResponse: Codable {
    let uniqueVisitors: Int
    let knownFaces: Int  // Total known faces synced across all stations
    let avgDwellMinutes: Double
    let totalIn: Int
    let totalOut: Int
    let currentOccupancy: Int
    let bodyIn: Int
    let bodyOut: Int
    let bodyNet: Int
    let countDrift: Int
    let lastBodyEvent: String?
    let peerStatus: String
    let peerData: PeerData?
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case uniqueVisitors = "unique_visitors"
        case knownFaces = "known_faces"
        case avgDwellMinutes = "avg_dwell_minutes"
        case totalIn = "total_in"
        case totalOut = "total_out"
        case currentOccupancy = "current_occupancy"
        case bodyIn = "body_in"
        case bodyOut = "body_out"
        case bodyNet = "body_net"
        case countDrift = "count_drift"
        case lastBodyEvent = "last_body_event"
        case peerStatus = "peer_status"
        case peerData = "peer_data"
        case timestamp
    }
}

struct PeerData: Codable {
    let uniqueVisitors: Int?
    let currentOccupancy: Int?
    let bodyIn: Int?

    enum CodingKeys: String, CodingKey {
        case uniqueVisitors = "unique_visitors"
        case currentOccupancy = "current_occupancy"
        case bodyIn = "body_in"
    }
}
