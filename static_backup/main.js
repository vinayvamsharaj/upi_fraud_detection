(function () {
  const root = document.documentElement;
  const btn = document.getElementById("themeBtn");

  function applyTheme(theme) {
    if (theme === "light") {
      root.setAttribute("data-theme", "light");
      if (btn) btn.textContent = "☀️";
    } else {
      root.removeAttribute("data-theme");
      if (btn) btn.textContent = "🌙";
    }
    localStorage.setItem("theme", theme);
  }

  const saved = localStorage.getItem("theme") || "dark";
  applyTheme(saved);

  if (btn) {
    btn.addEventListener("click", () => {
      const current = root.getAttribute("data-theme") === "light" ? "light" : "dark";
      applyTheme(current === "light" ? "dark" : "light");
    });
  }
})();
function fillSafe(){
  document.querySelector('input[name="amount"]').value = 200;
  document.querySelector('input[name="hour"]').value = 14;
  document.querySelector('select[name="new_device"]').value = "0";
  document.querySelector('input[name="transaction_count"]').value = 2;
}
function fillFraud(){
  document.querySelector('input[name="amount"]').value = 7000;
  document.querySelector('input[name="hour"]').value = 2;
  document.querySelector('select[name="new_device"]').value = "1";
  document.querySelector('input[name="transaction_count"]').value = 10;
}
async function clearHistory(){
  await fetch("/clear-history", { method: "POST" });
  location.reload();
}
// Real-time alert if result is fraud (based on body data attributes)
document.addEventListener("DOMContentLoaded", () => {
  const el = document.querySelector(".result.fraud");
  if (el) {
    setTimeout(() => alert("⚠️ High Risk Transaction Detected!"), 300);
  }
});