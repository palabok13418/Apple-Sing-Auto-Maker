/* Trusted parent bridge for Apple Sing TTML Forge. */
(() => {
  const query = new URLSearchParams(window.location.search);
  if (query.get("mus-api-embed") !== "1" || window.parent === window) return;
  let parentOrigin = "";
  try { parentOrigin = new URL(document.referrer).origin; } catch { return; }
  // Keep this aligned with frame-ancestors in vercel.json.
  if (parentOrigin !== "https://mus-api.vercel.app") return;
  document.documentElement.classList.add("mus-api-embedded");
  window.addEventListener("message", (event) => {
    if (event.source !== window.parent || event.origin !== parentOrigin) return;
    const data = event.data;
    if (!data || data.type !== "MUS_API_APP_PING" || data.version !== 1 || data.app !== "apple-sing-ttml-forge") return;
    window.parent.postMessage({ type: "APPLE_SING_TTML_READY", version: 1, app: "apple-sing-ttml-forge", embedded: true }, event.origin);
  });
})();
