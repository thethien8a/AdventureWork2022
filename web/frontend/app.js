const API_URL = "http://localhost:8000/predict";

const selectors = {
  form: document.getElementById("prediction-form"),
  submitButton: document.querySelector("button[type='submit']"),
  submitLabel: document.querySelector(".btn-label"),
  submitSpinner: document.querySelector(".btn-spinner"),
  resultContainer: document.getElementById("result"),
  resultPlaceholder: document.getElementById("result-placeholder"),
  rawDetails: document.querySelector(".raw-response"),
  rawJson: document.getElementById("raw-json"),
  resultTemplate: document.getElementById("result-template"),
  errorTemplate: document.getElementById("error-template"),
  themeToggle: document.getElementById("toggle-theme"),
};

async function bootstrap() {
  await loadOptions();
  hydrateFormWithDefaults();
  attachListeners();
}

async function loadOptions() {
  try {
    const res = await fetch("options.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`Không thể tải options.json (${res.status})`);
    const options = await res.json();

    populateSelect("PersonType", options.PersonType);
    populateSelect("ProductLine", options.ProductLine);
    populateSelect("Name_territory", options.Name_territory);
    populateSelect("CountryRegionCode", options.CountryRegionCode);
    populateSelect("Group", options.Group);
    populateProductDatalist(options.Name || []);
  } catch (error) {
    console.error("Failed to load options:", error);
    alert("Không thể tải dữ liệu danh mục (options.json). Hãy kiểm tra lại file hoặc quyền truy cập.");
  }
}

function populateSelect(id, values = []) {
  const select = document.getElementById(id);
  if (!select) return;
  select.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "-- Chọn --";
  placeholder.disabled = true;
  placeholder.selected = true;
  select.appendChild(placeholder);

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value || "(Không xác định)";
    select.appendChild(option);
  });
}

function populateProductDatalist(values) {
  const datalist = document.getElementById("product-options");
  if (!datalist) return;
  datalist.innerHTML = "";

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    datalist.appendChild(option);
  });
}

function hydrateFormWithDefaults() {
  const today = new Date();
  const formatted = today.toISOString().split("T")[0];
  const orderDateInput = document.getElementById("OrderDate");
  if (orderDateInput && !orderDateInput.value) {
    orderDateInput.value = formatted;
  }
}

function attachListeners() {
  selectors.form?.addEventListener("submit", handleSubmit);
  selectors.form?.addEventListener("reset", resetResults);
  selectors.themeToggle?.addEventListener("click", toggleTheme);

  const storedTheme = localStorage.getItem("aw-theme");
  if (storedTheme === "light") {
    document.body.classList.add("is-light");
    selectors.themeToggle?.setAttribute("aria-pressed", "true");
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const formData = new FormData(selectors.form);

  const payload = {
    PersonType: formData.get("PersonType"),
    OrderQty: parseInt(formData.get("OrderQty"), 10),
    Name: (formData.get("Name") || "").toString().trim(),
    ProductLine: formData.get("ProductLine"),
    Name_territory: formData.get("Name_territory"),
    CountryRegionCode: formData.get("CountryRegionCode"),
    Group: formData.get("Group"),
    OrderDate: normalizeDate(formData.get("OrderDate")),
  };

  const validationError = validatePayload(payload);
  if (validationError) {
    renderError({
      message: "Dữ liệu không hợp lệ",
      detail: validationError,
    });
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await safeJson(response);
      throw new Error(errorBody?.detail || response.statusText || "Yêu cầu thất bại");
    }

    const data = await response.json();
    renderSuccess(data);
  } catch (error) {
    console.error("Prediction failed", error);
    renderError({ message: "Không thể dự đoán", detail: error.message });
  } finally {
    setLoading(false);
  }
}

function normalizeDate(value) {
  if (!value) return value;
  // value already yyyy-mm-dd from input[type=date]
  return value;
}

function validatePayload(payload) {
  if (!payload.PersonType) return "Vui lòng chọn loại khách hàng.";
  if (!Number.isFinite(payload.OrderQty) || payload.OrderQty <= 0) {
    return "Số lượng phải lớn hơn 0.";
  }
  if (!payload.Name) return "Vui lòng nhập sản phẩm.";
  if (!payload.ProductLine) return "Vui lòng chọn dòng sản phẩm.";
  if (!payload.Name_territory) return "Vui lòng chọn lãnh thổ.";
  if (!payload.CountryRegionCode) return "Vui lòng chọn mã quốc gia.";
  if (!payload.Group) return "Vui lòng chọn nhóm khu vực.";
  if (!payload.OrderDate) return "Vui lòng chọn ngày đặt hàng.";
  return null;
}

function setLoading(isLoading) {
  if (!selectors.submitButton) return;
  selectors.submitButton.disabled = isLoading;
  selectors.submitSpinner.hidden = !isLoading;
  selectors.submitLabel.textContent = isLoading ? "Đang dự đoán..." : "Dự đoán ngay";
}

function renderSuccess(data) {
  if (!selectors.resultContainer || !selectors.resultTemplate) return;

  selectors.resultPlaceholder.hidden = true;
  selectors.resultContainer.hidden = false;
  selectors.rawDetails.hidden = false;

  const fragment = selectors.resultTemplate.content.cloneNode(true);
  const timeEl = fragment.querySelector("time");
  const amountEl = fragment.querySelector(".prediction-value");
  const metaList = fragment.querySelector(".prediction-meta");

  const timestamp = data.timestamp ? new Date(data.timestamp) : new Date();
  timeEl.textContent = `Lúc ${timestamp.toLocaleString("vi-VN")}`;

  // If model returns a negative prediction, display 0 instead
  const rawPrediction = Number(data.prediction);
  const displayPrediction = Number.isFinite(rawPrediction) ? Math.max(0, rawPrediction) : 0;
  amountEl.textContent = formatCurrency(displayPrediction);

  const inputEntries = Object.entries(data.input_data || {});
  metaList.innerHTML = "";

  inputEntries.forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    metaList.append(dt, dd);
  });

  selectors.resultContainer.replaceChildren(fragment);
  selectors.rawJson.textContent = JSON.stringify(data, null, 2);
}

function renderError({ message, detail }) {
  if (!selectors.resultContainer || !selectors.errorTemplate) return;

  selectors.resultPlaceholder.hidden = true;
  selectors.resultContainer.hidden = false;
  selectors.rawDetails.hidden = true;

  const fragment = selectors.errorTemplate.content.cloneNode(true);
  const timeEl = fragment.querySelector("time");
  const messageEl = fragment.querySelector(".error-message");
  const detailEl = fragment.querySelector(".error-detail");

  timeEl.textContent = `Lúc ${new Date().toLocaleString("vi-VN")}`;
  messageEl.textContent = message;
  detailEl.textContent = detail;

  selectors.resultContainer.replaceChildren(fragment);
}

function resetResults() {
  selectors.resultContainer.hidden = true;
  selectors.resultPlaceholder.hidden = false;
  selectors.rawDetails.hidden = true;
  selectors.rawJson.textContent = "";
  selectors.submitLabel.textContent = "Dự đoán ngay";
  selectors.submitSpinner.hidden = true;
  selectors.submitButton.disabled = false;
}

function formatCurrency(value) {
  if (typeof value !== "number") return "Không xác định";
  return value.toLocaleString("vi-VN", { style: "currency", currency: "USD" });
}

async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function toggleTheme() {
  const isLight = document.body.classList.toggle("is-light");
  selectors.themeToggle?.setAttribute("aria-pressed", String(isLight));
  localStorage.setItem("aw-theme", isLight ? "light" : "dark");
}

bootstrap();

