import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/ibm-plex-sans";
import "@fontsource/ibm-plex-mono/400.css";
import { App } from "./App.jsx";
import { LiveView } from "./LiveView.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {window.location.pathname.startsWith("/live") ? <LiveView /> : <App />}
  </React.StrictMode>,
);
