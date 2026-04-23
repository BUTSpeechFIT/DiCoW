function onScroll() {
  const header = document.getElementById("header");
  if (!header) return;
  if (window.scrollY > 0) {
    header.classList.add("scrolled");
  } else {
    header.classList.remove("scrolled");
  }
}

document.addEventListener("scroll", onScroll);
onScroll();
