require('dotenv').config();
const express = require("express");
const mongoose = require("mongoose"); // Add mongoose
const cors = require("cors");
const axios = require("axios");
const multer = require("multer");
const FormData = require("form-data");
const fs = require("fs");

const app = express();

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/pdf-extractor')
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));


app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Setup multer for file uploads
const upload = multer({ dest: "uploads/" });

app.get("/", (req, res) => {
  res.json({ message: "Welcome to the application." });
});

// Route to handle file upload and extraction
app.post("/api/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("No file uploaded.");
  }

  const filePath = req.file.path;

  try {
    // Create a form and append the file
    const form = new FormData();
    form.append("file", fs.createReadStream(filePath));

    // Forward the file to the Python service
    const pythonServiceUrl = "http://localhost:8001/api/extract";
    const response = await axios.post(pythonServiceUrl, form, {
      headers: {
        ...form.getHeaders(),
      },
    });

    // Send the extracted data back to the client
    res.json(response.data);
  } catch (error) {
    console.error("Error calling Python service:", error.message);
    res.status(500).send("Error processing file.");
  } finally {
    // Clean up the uploaded file
    fs.unlinkSync(filePath);
  }
});

const authRoutes = require("./routes/auth.routes");
app.use("/api/auth", authRoutes);

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}.`);
}); 