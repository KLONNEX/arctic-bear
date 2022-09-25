function chng2load() {
  var btn = document.getElementById("btnDownload");
  var txt = document.getElementById("txtDownload");
  txt.innerHTML = "<span class=\"spinner-border spinner-border-sm\" role=\"status\" aria-hidden=\"true\"></span>"
  btn.value = "Обработка данных...";
  btn.style.visibility = "visible";
  btn.disabled = true;
}

function chng2dnld() {
  var btn = document.getElementById("btnDownload");
  var txt = document.getElementById("txtDownload");
  btn.value = "Загрузить";
  btn.disabled = false;
  txt.innerHTML = "";
}

function chng2hide() {
  var btn = document.getElementById("btnDownload");
  var txt = document.getElementById("txtDownload");
  btn.disabled = true;
  btn.style.visibility = "hidden";
  txt.innerHTML = "";
}
