document.addEventListener("DOMContentLoaded", () => {
  const quizPanel = document.getElementById("learning-panel");
  const videoPlayer = document.getElementById("quizVideo");
  const answerInput = document.getElementById("answer");
  const submitBtn = document.getElementById("submitAnswer");
  const nextBtn = document.getElementById("nextQuestion");
  const feedbackMsg = document.getElementById("feedbackMessage");
  const statDisplay = document.getElementById("statsDisplay");

  let currentQuestionToken = null;

  if (!quizPanel) return;

  loadQuestion();

  submitBtn.addEventListener("click", submitAnswer);
  nextBtn.addEventListener("click", loadQuestion);

  answerInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") submitAnswer();
  });

  async function loadQuestion() {
    answerInput.value = "";
    answerInput.disabled = false;
    submitBtn.style.display = "inline-block";
    nextBtn.style.display = "none";
    feedbackMsg.textContent = "Loading video...";

    videoPlayer.src = "";

    try {
      const res = await fetch("/get_question");
      const data = await res.json();

      if (!data.question_token) {
        throw new Error("Server failed to provide a question token.");
      }

      currentQuestionToken = data.question_token;
      updateStats(data.stats);

      const videoRes = await fetch(
        `/get_video_content?token=${currentQuestionToken}`
      );

      if (!videoRes.ok) {
        throw new Error("Video not found");
      }

      const videoBlob = await videoRes.blob();
      const videoUrl = URL.createObjectURL(videoBlob);

      videoPlayer.src = videoUrl;
      videoPlayer.play().catch((e) => console.log("Autoplay blocked:", e));
      feedbackMsg.textContent = "";
    } catch (err) {
      console.error(err);
      feedbackMsg.textContent =
        "Error loading question. Please ensure video files exist on server.";
    }
  }

  async function submitAnswer() {
    const text = answerInput.value.trim();
    if (!text || !currentQuestionToken) return;

    answerInput.disabled = true;
    submitBtn.disabled = true;

    try {
      const res = await fetch("/submit_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer: text,
          question_token: currentQuestionToken,
        }),
      });

      const result = await res.json();

      if (result.correct) {
        feedbackMsg.textContent = "Correct! Great job.";
      } else {
        feedbackMsg.textContent = `Incorrect. The answer was: "${result.correct_answer}"`;
      }

      currentQuestionToken = null;

      submitBtn.style.display = "none";
      submitBtn.disabled = false;
      nextBtn.style.display = "inline-block";
      nextBtn.focus();
    } catch (err) {
      console.error(err);
      feedbackMsg.textContent = "Error submitting answer.";
      answerInput.disabled = false;
      submitBtn.disabled = false;
    }
  }

  function updateStats(stats) {
    if (stats) {
      statDisplay.textContent = `Your score (last 24 hours): ${stats.correct_24h} / ${stats.total_24h}`;
    }
  }
});
