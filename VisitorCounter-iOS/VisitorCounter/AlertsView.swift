import SwiftUI

struct AlertsView: View {
    @ObservedObject var viewModel: StatsViewModel
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            ZStack {
                backgroundGradient
                alertContent
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

    private var backgroundGradient: some View {
        LinearGradient(
            colors: [Color(hex: "1a1a2e"), Color(hex: "16213e"), Color(hex: "0f3460")],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }

    @ViewBuilder
    private var alertContent: some View {
        if viewModel.alerts.isEmpty {
            emptyState
        } else {
            alertList
        }
    }

    private var emptyState: some View {
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
    }

    private var alertList: some View {
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

struct AlertCard: View {
    let alert: WatchlistAlert
    let onAcknowledge: () -> Void

    private var iconColor: Color {
        alert.acknowledged ? Color.gray : Color.red
    }

    private var backgroundOpacity: Double {
        alert.acknowledged ? 0.03 : 0.1
    }

    private var borderOpacity: Double {
        alert.acknowledged ? 0.1 : 0.3
    }

    var body: some View {
        HStack(spacing: 16) {
            alertIcon
            alertInfo
            Spacer()
            acknowledgeButton
        }
        .padding()
        .background(cardBackground)
    }

    private var alertIcon: some View {
        ZStack {
            Circle()
                .fill(iconColor.opacity(0.3))
                .frame(width: 50, height: 50)

            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title2)
                .foregroundColor(iconColor)
        }
    }

    private var alertInfo: some View {
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
    }

    @ViewBuilder
    private var acknowledgeButton: some View {
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

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(iconColor.opacity(backgroundOpacity))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(iconColor.opacity(borderOpacity), lineWidth: 1)
            )
    }
}

#Preview {
    AlertsView(viewModel: StatsViewModel())
}
