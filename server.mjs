import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import dotenv from "dotenv";
import { askLegalQuestion } from "./qa.mjs";

dotenv.config();

const app = express();
const server = createServer(app);
const io = new Server(server);

app.use(express.static("public"));

io.on("connection", socket => {
  console.log("⚡ New client connected");

  socket.on("chat message", async msg => {
    console.log("❓ User:", msg);
    const reply = await askLegalQuestion(msg);
    socket.emit("chat response", reply);
  });

  socket.on("disconnect", () => {
    console.log("❌ Client disconnected");
  });
});

const PORT = 3000;
server.listen(PORT, () => {
  console.log(`✅ Chatbot running: http://localhost:${PORT}`);
});
