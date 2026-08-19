/* ERP AI Support — lógica de interface */

"use strict";

(function () {
  /* ── Upload ─────────────────────────────────────────────────── */

  var uploadForm     = document.getElementById("upload-form");
  var uploadInput    = document.getElementById("upload-file");
  var uploadBtn      = document.getElementById("upload-btn");
  var uploadStatus   = document.getElementById("upload-status");
  var uploadCounters = document.getElementById("upload-counters");
  var countDocs      = document.getElementById("count-docs");
  var countPages     = document.getElementById("count-pages");
  var countChunks    = document.getElementById("count-chunks");
  var warningsList   = document.getElementById("upload-warnings");

  var uploadBusy = false;

  function setUploadState(busy) {
    uploadBusy = busy;
    uploadBtn.disabled  = busy;
    uploadInput.disabled = busy;
    uploadBtn.setAttribute("aria-busy", busy ? "true" : "false");
    uploadBtn.textContent = busy ? "Processando…" : "Importar";
  }

  function showUploadStatus(cssClass, text) {
    uploadStatus.className = "status-area " + cssClass;
    uploadStatus.removeAttribute("hidden");
    uploadStatus.textContent = "";
    uploadStatus.appendChild(document.createTextNode(text));
  }

  function clearUploadResult() {
    uploadCounters.setAttribute("hidden", "");
    warningsList.setAttribute("hidden", "");
    warningsList.textContent = "";
  }

  function showCounters(docs, pages, chunks) {
    countDocs.textContent   = docs;
    countPages.textContent  = pages;
    countChunks.textContent = chunks;
    uploadCounters.removeAttribute("hidden");
  }

  function showWarnings(warnings) {
    if (!warnings || warnings.length === 0) return;
    warningsList.textContent = "";
    warnings.forEach(function (msg) {
      var li = document.createElement("li");
      li.appendChild(document.createTextNode(msg));
      warningsList.appendChild(li);
    });
    warningsList.removeAttribute("hidden");
  }

  if (uploadForm) {
    uploadForm.addEventListener("submit", function (evt) {
      evt.preventDefault();
      if (uploadBusy) return;

      var file = uploadInput.files && uploadInput.files[0];
      if (!file) {
        showUploadStatus("status-error", "Selecione um arquivo .zip antes de importar.");
        return;
      }

      clearUploadResult();
      setUploadState(true);
      uploadStatus.setAttribute("hidden", "");

      var formData = new FormData();
      formData.append("file", file);

      fetch("/upload", {
        method: "POST",
        body:   formData,
      })
        .then(function (response) {
          return response.json().then(function (data) {
            return { ok: response.ok, status: response.status, data: data };
          });
        })
        .then(function (result) {
          var data = result.data;
          if (result.ok && data && data.success) {
            showUploadStatus(
              "status-ok",
              "Importação concluída: " +
                data.documents + " documento(s), " +
                data.pages + " página(s), " +
                data.chunks + " chunk(s)."
            );
            showCounters(data.documents, data.pages, data.chunks);
            showWarnings(data.warnings);
          } else {
            var msg = (data && data.message)
              ? data.message
              : "Erro ao importar o arquivo (HTTP " + result.status + ").";
            showUploadStatus("status-error", msg);
          }
        })
        .catch(function (err) {
          showUploadStatus(
            "status-error",
            "Falha de rede ao importar. Verifique a conexão e tente novamente."
          );
        })
        .finally(function () {
          setUploadState(false);
        });
    });
  }

  /* ── Chat ───────────────────────────────────────────────────── */

  var chatForm      = document.getElementById("chat-form");
  var questionInput = document.getElementById("question-input");
  var chatBtn       = document.getElementById("chat-btn");
  var answerArea    = document.getElementById("answer-area");
  var sourcesArea   = document.getElementById("sources-area");
  var sourcesList   = document.getElementById("sources-list");

  var chatBusy = false;

  function setChatState(busy) {
    chatBusy = busy;
    chatBtn.disabled          = busy;
    questionInput.disabled    = busy;
    chatBtn.setAttribute("aria-busy", busy ? "true" : "false");
    chatBtn.textContent = busy ? "Consultando…" : "Consultar";
  }

  function showAnswer(text) {
    answerArea.textContent = "";
    answerArea.appendChild(document.createTextNode(text));
    answerArea.removeAttribute("hidden");
  }

  function showSources(sources) {
    sourcesArea.setAttribute("hidden", "");
    sourcesList.textContent = "";
    if (!sources || sources.length === 0) return;
    sources.forEach(function (src) {
      var li   = document.createElement("li");
      var text = src.document ? src.document + " — p. " + src.page : "";
      li.appendChild(document.createTextNode(text));
      sourcesList.appendChild(li);
    });
    sourcesArea.removeAttribute("hidden");
  }

  if (chatForm) {
    chatForm.addEventListener("submit", function (evt) {
      evt.preventDefault();
      if (chatBusy) return;

      var question = questionInput.value;
      if (!question || !question.trim()) {
        showAnswer("Escreva uma pergunta antes de consultar.");
        sourcesArea.setAttribute("hidden", "");
        return;
      }

      setChatState(true);
      answerArea.setAttribute("hidden", "");
      sourcesArea.setAttribute("hidden", "");

      fetch("/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ question: question }),
      })
        .then(function (response) {
          return response.json().then(function (data) {
            return { ok: response.ok, status: response.status, data: data };
          });
        })
        .then(function (result) {
          var data = result.data;
          if (result.ok && data && data.answer !== undefined) {
            showAnswer(data.answer);
            showSources(data.sources);
          } else {
            var msg = (data && data.message)
              ? data.message
              : "Erro ao consultar (HTTP " + result.status + ").";
            showAnswer(msg);
            sourcesArea.setAttribute("hidden", "");
          }
        })
        .catch(function () {
          showAnswer("Falha de rede ao consultar. A pergunta foi preservada.");
          sourcesArea.setAttribute("hidden", "");
        })
        .finally(function () {
          setChatState(false);
        });
    });
  }
})();
