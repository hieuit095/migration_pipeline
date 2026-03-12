var childProcess = require("child_process");

function escapeShellValue(value) {
  return String(value).replace(/(["`\\$])/g, "\\$1");
}

function buildTelegramCommand(token, chatId, text) {
  return (
    'curl -s -X POST "https://api.telegram.org/bot' +
    escapeShellValue(token) +
    '/sendMessage" ' +
    '-d chat_id="' +
    escapeShellValue(chatId) +
    '" ' +
    '--data-urlencode text="' +
    escapeShellValue(text) +
    '"'
  );
}

function currentRevision() {
  return childProcess
    .execSync("git rev-parse --short HEAD", {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"]
    })
    .trim();
}

function sendTelegramMessage(token, chatId, text, callback) {
  try {
    var command = buildTelegramCommand(token, chatId, text);
    var rawOutput = childProcess.execSync(command, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"]
    });
    var payload = rawOutput ? JSON.parse(rawOutput) : { ok: false };
    callback(null, payload);
  } catch (error) {
    callback(error, null);
  }
}

function notifyDeployment(environmentName, version, callback) {
  var chatId = process.env.TELEGRAM_CHAT_ID || "ops-room";
  var token = process.env.TELEGRAM_BOT_TOKEN || "missing-token";
  var revision = currentRevision();
  var message =
    "[deploy] " +
    environmentName +
    " -> " +
    version +
    " (" +
    revision +
    ")";

  sendTelegramMessage(token, chatId, message, callback);
}

module.exports = {
  buildTelegramCommand: buildTelegramCommand,
  currentRevision: currentRevision,
  sendTelegramMessage: sendTelegramMessage,
  notifyDeployment: notifyDeployment
};
