/**
 * app.js — MySearchEngine
 * Progressive interaction enhancements — no external dependencies.
 */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------------------------
    // 1. Auto-focus on empty search input
    // -----------------------------------------------------------------------
    const searchInput = document.getElementById("searchInput");
    if (searchInput && searchInput.value === "") {
        searchInput.focus();
    }

    // -----------------------------------------------------------------------
    // 2. Loading state while the form submits
    // -----------------------------------------------------------------------
    const searchForm = document.getElementById("searchForm");
    const searchBtn  = document.getElementById("searchBtn");

    if (searchForm && searchBtn) {
        searchForm.addEventListener("submit", (e) => {
            const query = searchInput ? searchInput.value.trim() : "";
            if (query.length === 0) {
                e.preventDefault();
                if (searchInput) searchInput.focus();
                return;
            }
            searchBtn.textContent = "Buscando…";
            searchBtn.disabled    = true;
        });
    }

    // -----------------------------------------------------------------------
    // 3. Staggered card entrance animation via IntersectionObserver
    //    (Highlighting is done server-side — see search_engine._highlight_text)
    // -----------------------------------------------------------------------
    const resultCards = document.querySelectorAll(".result-card");

    if (resultCards.length > 0 && "IntersectionObserver" in window) {
        resultCards.forEach((card, i) => {
            card.classList.add("card-hidden");
            card.style.transitionDelay = `${i * 40}ms`;
        });

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.remove("card-hidden");
                    entry.target.classList.add("card-visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.04, rootMargin: "0px 0px -20px 0px" });

        resultCards.forEach(card => observer.observe(card));
    }
});

