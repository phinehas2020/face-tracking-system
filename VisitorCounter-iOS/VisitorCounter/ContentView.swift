import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StatsViewModel()
    @State private var showSettings = false

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

                ScrollView {
                    VStack(spacing: 20) {
                        // Connection status
                        HStack {
                            Circle()
                                .fill(viewModel.isConnected ? Color.green : Color.red)
                                .frame(width: 10, height: 10)
                            Text(viewModel.isConnected ? "Connected" : "Disconnected")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.7))

                            Spacer()

                            Button(action: { showSettings = true }) {
                                Image(systemName: "gear")
                                    .foregroundColor(.white.opacity(0.7))
                            }
                        }
                        .padding(.horizontal)

                        // Main stat - Total Visitors
                        FeaturedStatCard(
                            icon: "target",
                            value: "\(viewModel.grandTotal)",
                            label: "TOTAL VISITORS"
                        )

                        // Secondary stats grid
                        LazyVGrid(columns: [
                            GridItem(.flexible()),
                            GridItem(.flexible())
                        ], spacing: 16) {
                            StatCard(
                                icon: "a.circle.fill",
                                value: "\(viewModel.stats?.uniqueVisitors ?? 0)",
                                label: "STATION A",
                                subtitle: "This"
                            )

                            StatCard(
                                icon: "b.circle.fill",
                                value: "\(viewModel.stats?.peerData?.uniqueVisitors ?? 0)",
                                label: "STATION B",
                                subtitle: viewModel.stats?.peerStatus == "connected" ? "Peer" : "Offline"
                            )

                            StatCard(
                                icon: "chart.line.uptrend.xyaxis",
                                value: "\(viewModel.stats?.totalIn ?? 0)",
                                label: "TOTAL ENTRIES"
                            )

                            StatCard(
                                icon: "camera.fill",
                                value: "\(viewModel.stats?.bodyIn ?? 0)",
                                label: "SIDE CAMERA",
                                subtitle: viewModel.stats?.lastBodyEvent != nil ? "Active" : "Idle"
                            )
                        }
                        .padding(.horizontal)

                        // Visitor icons visualization
                        if let count = viewModel.stats?.totalIn, count > 0 {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Text("Live Entry Feed")
                                        .font(.headline)
                                        .foregroundColor(.white.opacity(0.8))
                                    Spacer()
                                    Text("\(count) visitors")
                                        .font(.caption)
                                        .foregroundColor(.white.opacity(0.5))
                                }

                                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 8), spacing: 8) {
                                    ForEach(0..<min(count, 40), id: \.self) { _ in
                                        Circle()
                                            .fill(
                                                LinearGradient(
                                                    colors: [Color(hex: "00d9ff").opacity(0.3), Color(hex: "00ff88").opacity(0.3)],
                                                    startPoint: .topLeading,
                                                    endPoint: .bottomTrailing
                                                )
                                            )
                                            .frame(width: 36, height: 36)
                                            .overlay(
                                                Text("👤")
                                                    .font(.system(size: 18))
                                            )
                                    }
                                }
                            }
                            .padding()
                            .background(Color.white.opacity(0.05))
                            .cornerRadius(20)
                            .padding(.horizontal)
                        }

                        // Error message
                        if let error = viewModel.errorMessage {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.red.opacity(0.8))
                                .padding()
                        }

                        Spacer(minLength: 50)
                    }
                    .padding(.top)
                }
            }
            .navigationTitle("Visitor Counter")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Color(hex: "1a1a2e"), for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
        }
        .onAppear {
            viewModel.startPolling()
        }
        .onDisappear {
            viewModel.stopPolling()
        }
    }
}

struct FeaturedStatCard: View {
    let icon: String
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 32))
                .foregroundColor(Color(hex: "00ff88"))

            Text(value)
                .font(.system(size: 64, weight: .bold, design: .rounded))
                .foregroundStyle(
                    LinearGradient(
                        colors: [Color(hex: "00d9ff"), Color(hex: "00ff88")],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )

            Text(label)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.white.opacity(0.6))
                .tracking(2)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
        .background(
            RoundedRectangle(cornerRadius: 24)
                .fill(
                    LinearGradient(
                        colors: [Color(hex: "00d9ff").opacity(0.15), Color(hex: "00ff88").opacity(0.15)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 24)
                        .stroke(Color(hex: "00ff88").opacity(0.3), lineWidth: 1)
                )
        )
        .padding(.horizontal)
    }
}

struct StatCard: View {
    let icon: String
    let value: String
    let label: String
    var subtitle: String? = nil
    var isText: Bool = false

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.white.opacity(0.6))

            if isText {
                Text(value)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
            } else {
                Text(value)
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
            }

            Text(label)
                .font(.caption2)
                .fontWeight(.medium)
                .foregroundColor(.white.opacity(0.5))
                .tracking(1)

            if let subtitle = subtitle {
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(Color(hex: "00ff88"))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
        .background(Color.white.opacity(0.05))
        .cornerRadius(20)
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.1), lineWidth: 1)
        )
    }
}

struct SettingsView: View {
    @ObservedObject var viewModel: StatsViewModel
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Server Configuration")) {
                    TextField("Server Address", text: $viewModel.serverAddress)
                        .keyboardType(.URL)
                        .autocapitalization(.none)
                        .autocorrectionDisabled()

                    Text("Enter one of:\n• IP address: 192.168.1.100\n• Cloudflare URL: https://xyz.trycloudflare.com")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Connection Status")) {
                    HStack {
                        Text("Status")
                        Spacer()
                        HStack {
                            Circle()
                                .fill(viewModel.isConnected ? Color.green : Color.red)
                                .frame(width: 8, height: 8)
                            Text(viewModel.isConnected ? "Connected" : "Disconnected")
                                .foregroundColor(.secondary)
                        }
                    }

                    Button("Test Connection") {
                        viewModel.fetchStats()
                    }
                }

                Section(header: Text("Quick Setup")) {
                    Text("1. Run ./start_all.sh on your Mac\n2. Run ./start_tunnel.sh for remote access\n3. Copy the tunnel URL here")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// Color extension for hex colors
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
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

#Preview {
    ContentView()
}
