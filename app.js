
document.addEventListener("DOMContentLoaded", () => {
  const galleryCards = Array.from(document.querySelectorAll(".gallery-card"));
  const toggle = document.querySelector("[data-toggle-gallery]");
  const galleryCount = document.querySelector("[data-gallery-count]");
  let expanded = false;

  function updateGallery() {
    galleryCards.forEach((card, index) => {
      card.hidden = !expanded && index >= 12;
    });
    if (toggle) {
      toggle.textContent = expanded ? "Show first 12" : "Show all 100 panels";
      toggle.setAttribute("aria-expanded", String(expanded));
    }
    if (galleryCount) {
      galleryCount.textContent = expanded ? "Showing all 100 panels" : "Showing 12 of 100 panels";
    }
  }

  if (toggle) {
    toggle.addEventListener("click", () => {
      expanded = !expanded;
      updateGallery();
    });
  }
  updateGallery();

  const lightbox = document.querySelector(".lightbox");
  const lightboxImage = document.querySelector(".lightbox img");
  const lightboxClose = document.querySelector(".lightbox button");

  function openLightbox(src, alt) {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = src;
    lightboxImage.alt = alt || "Virtual staining comparison panel";
    lightbox.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    if (!lightbox || !lightboxImage) return;
    lightbox.classList.remove("open");
    lightboxImage.src = "";
    document.body.style.overflow = "";
  }

  document.querySelectorAll("[data-lightbox]").forEach((image) => {
    image.addEventListener("click", () => openLightbox(image.src, image.alt));
  });
  if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
  if (lightbox) {
    lightbox.addEventListener("click", (event) => {
      if (event.target === lightbox) closeLightbox();
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
});
