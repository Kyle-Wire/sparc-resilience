import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Setup from "./pages/Setup";
import ErrorBoundary from "./components/common/ErrorBoundary";
import "./index.css";

const isSetupWindow = window.location.hash === "#/setup";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary page="root">
      {isSetupWindow ? <Setup /> : <App />}
    </ErrorBoundary>
  </React.StrictMode>,
);
