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

// Setup multer for file uploads with size limit
const upload = multer({ 
  dest: "uploads/",
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  }
});

app.get("/", (req, res) => {
  res.json({ message: "Welcome to the application." });
});

// Route to handle file upload and extraction
app.post("/api/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded." });
  }

  // Check file type
  if (req.file.mimetype !== 'application/pdf') {
    return res.status(400).json({ error: "Only PDF files are allowed." });
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
      timeout: 300000, // 5 minutes timeout for large images
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    // Log the response from Python service for debugging
    console.log("Python service response:", JSON.stringify(response.data, null, 2));
    
    // Check if visualization data exists
    if (response.data && response.data.data) {
      response.data.data.forEach((table, index) => {
        console.log(`Table ${index + 1} visualizations:`, table.visualizations ? 'Present' : 'Missing');
        if (table.visualizations) {
          console.log(`  - table_detection_image: ${table.visualizations.table_detection_image ? 'Present' : 'Missing'}`);
          console.log(`  - cell_segmentation_image: ${table.visualizations.cell_segmentation_image ? 'Present' : 'Missing'}`);
        }
      });
    }

    // Send the extracted data back to the client
    res.json(response.data);
  } catch (error) {
    console.error("Error calling Python service:", error.message);
    
    // Handle file size limit error
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: "File too large. Maximum size is 50MB." });
    }
    
    res.status(500).json({ error: "Error processing file." });
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