// Entry point: tokens and base styles, bundled fonts, the dev token bootstrap (npm run dev),
// then the app.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ensureDevAuth } from "./api/client";
import { installFonts } from "./theme/fonts";
import "./theme/tokens.css";
import "./theme/base.css";

installFonts();

const root = document.getElementById("root");
if (!root) throw new Error("index.html has no #root");

// In dev the page exchanges VITE_STUDIO_TOKEN for the session cookie first (SPEC §13.3);
// in production the server already set it through /auth?t=….
void ensureDevAuth().finally(() => {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
