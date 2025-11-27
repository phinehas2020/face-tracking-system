import SwiftUI

struct AlertsView: View {
    @ObservedObject var viewModel: StatsViewModel
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    colors: [Color(hex: "1a1a2e"), Color(hex: "16213e"), Color(hex: "0f3460")],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                if viewModel.alerts.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "bell.slash")
                            .font(.system(size: 48))
                            .foregroundColor(.white.opacity(0.4))
                        Text("No alerts")
                            .font(.headline)
                            .foregroundColor(.white.opacity(0.6))
                        Text("Watchlist alerts will appear here")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.4))
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.alerts) { alert in
                                AlertCard(alert: alert) {
                                    viewModel.acknowledgeAlert(alert.id)
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Watchlist Alerts")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Color(hex: "1a1a2e"), for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.alerts.isEmpty {
                        Button("Clear All") {
                            viewModel.acknowledgeAllAlerts()
                        }
                    }
                }
            }
        }
    }
}

struct AlertCard: View {
    let alert: WatchlistAlert
    let onAcknowledge: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            // Alert icon
            ZStack {
                Circle()
                    .fill(alert.acknowledged ? Color.gray.opacity(0.3) : Color.red.opacity(0.3))
                    .frame(width: 50, height: 50)

                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundColor(alert.acknowledged ? .gray : .red)
            }

            // Alert info
            VStack(alignment: .leading, spacing: 4) {
                Text(alert.name)
                    .font(.headline)
                    .foregroundColor(.white)

                Text("Detected at \(alert.formattedTime)")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.6))

                Text("Similarity: \(Int(alert.similarity * 100))%")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.4))
            }

            Spacer()

            // Acknowledge button
            if !alert.acknowledged {
                Button(action: onAcknowledge) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.title2)
                        .foregroundColor(Color(hex: "00ff88"))
                }
            } else {
                Image(systemName: "checkmark.circle")
                    .font(.title2)
                    .foregroundColor(.gray)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(alert.acknowledged ? Color.white.opacity(0.03) : Color.red.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(alert.acknowledged ? Color.white.opacity(0.1) : Color.red.opacity(0.3), lineWidth: 1)
                )
        )
    }
}

#Preview {
    AlertsView(viewModel: StatsViewModel())
}
