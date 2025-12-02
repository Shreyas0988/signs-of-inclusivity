class Tabs {
  constructor(container) {
    this.container = container;
    this.tabsList = container.querySelector(".tabsList");
    this.tabs = Array.from(container.querySelectorAll(".tab"));
    this.indicator = container.querySelector(".tabIndicator");
    this.panels = [];

    this.tabs.forEach((tab) => {
      const panelId = tab.getAttribute("aria-controls");
      const panel = document.getElementById(panelId);
      if (panel) {
        this.panels.push(panel);
      }
    });

    this.init();
  }

  init() {
    this.tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => this.selectTab(index));
    });

    this.tabsList.addEventListener("keydown", (e) => this.handleKeydown(e));

    this.updateIndicator();

    window.addEventListener("resize", () => this.updateIndicator());
  }

  selectTab(index) {
    this.tabs.forEach((tab, i) => {
      const isSelected = i === index;
      tab.classList.toggle("active", isSelected);
      tab.setAttribute("aria-selected", isSelected);
      tab.setAttribute("tabindex", isSelected ? 0 : -1)

      if (this.panels[i]) {
        this.panels[i].classList.toggle("active", isSelected);
      }
    });

    this.updateIndicator();
  }

  updateIndicator() {
    const activeTab = this.tabs.find((tab) => tab.classList.contains("active"));
    if (activeTab) {
      const rect = activeTab.getBoundingClientRect();
      const listRect = this.tabsList.getBoundingClientRect();

      this.indicator.style.left = rect.left - listRect.left + "px";
      this.indicator.style.top = rect.top - listRect.top + "px";
      this.indicator.style.width = rect.width + "px";
      this.indicator.style.height = rect.height + "px";
    }
  }

  handleKeydown(e) {
    const currentIndex = this.tabs.findIndex(
      (tab) => tab === document.activeElement
    );

    if (currentIndex === -1) return;

    let newIndex = currentIndex;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        newIndex = currentIndex > 0 ? currentIndex - 1 : this.tabs.length - 1;
        break;
      case "ArrowRight":
        e.preventDefault();
        newIndex = currentIndex < this.tabs.length - 1 ? currentIndex + 1 : 0;
        break;
      case "Home":
        e.preventDefault();
        newIndex = 0;
        break;
      case "End":
        e.preventDefault();
        newIndex = this.tabs.length - 1;
        break;
      default:
        return;
    }

    this.tabs[newIndex].focus();
    this.selectTab(newIndex);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const tabContainer = document.querySelector(".tabContainer");
  if (tabContainer) {
    new Tabs(tabContainer);
  }
});
