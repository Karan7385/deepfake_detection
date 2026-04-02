const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

// --- 1. Configuration and Constants ---
const PORT = process.env.PORT || 3000;
const IMAGE_API_URL = "http://127.0.0.1:8000/predict_image";
const VIDEO_API_URL = "http://127.0.0.1:8080/predict_video/";
const uploadDir = path.join(__dirname, "uploads");

// --- 2. Custom Error Class ---

/**
 * Purpose: A custom error class for consistent API error reporting.
 */
class ApiError extends Error {
    constructor(message, statusCode = 500) {
        super(message);
        this.statusCode = statusCode;
        this.name = 'ApiError';
    }
}

// --- 3. Utility Functions ---

/**
 * Purpose: Calls the external Python prediction service (FastAPI) for image analysis.
 * @param {string} filePath - Local path to the uploaded image file.
 * @returns {Promise<object>} The parsed JSON response from the Python server.
 * @throws {ApiError} If the Python service fails or returns an error status.
 */
async function callImageService(filePath) {
    try {
        const response = await axios.post(
            IMAGE_API_URL,
            { file_path: filePath },
            { headers: { "Content-Type": "application/json" } }
        );
        return response.data;
    } catch (error) {
        // Handle network errors (Axios error with no response)
        if (!error.response) {
            throw new ApiError(`Connection error: Could not reach Python service at ${IMAGE_API_URL}`, 503);
        }

        // Handle errors returned from the Python server (HTTP status code >= 400)
        const status = error.response.status;
        const detail = error.response.data?.detail || error.response.data?.error || "Unknown error from prediction service.";

        console.error(`Python API returned ${status}: ${detail}`);
        throw new ApiError(`Prediction failed: ${detail}`, status);
    }
}

const FormData = require("form-data");

async function callVideoService(filePath) {

    try {

        const form = new FormData();

        form.append(
            "file",
            fs.createReadStream(filePath)
        );

        const response = await axios.post(

            VIDEO_API_URL,
            form,
            {
                headers: form.getHeaders()
            }

        );

        return response.data;

    }
    catch(error){

        if(!error.response){

            throw new ApiError(
                `Connection error: Could not reach Python service`,
                503
            );

        }

        const status = error.response.status;

        const detail =
        error.response.data?.detail ||
        error.response.data?.error ||
        "Unknown error";

        throw new ApiError(
            `Prediction failed: ${detail}`,
            status
        );

    }

}

/**
 * Purpose: Configures Multer for file storage and filtering.
 * @returns {Function} Multer middleware configured for single file upload.
 */
function configureMulter() {
    const storage = multer.diskStorage({
        destination: (req, file, cb) => cb(null, uploadDir),
        filename: (req, file, cb) => cb(null, Date.now() + "_" + file.originalname),
    });

    return multer({
        storage,
        limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
        fileFilter: (req, file, cb) => {
            // Only allow common image and video formats
            if (file.mimetype.startsWith("image/") || file.mimetype.startsWith("video/")) {
                cb(null, true);
            } else {
                // Return an error to Multer
                cb(new Error("Unsupported file type. Only images and videos are allowed."), false);
            }
        },
    });
}

/**
 * Purpose: Handles the core logic for the /upload endpoint. Orchestrates file
 * type checking, Python service call, and response formatting. Uses next(err) 
 * to pass errors to the global error handler.
 */
async function handleFileUploadAndPrediction(req, res, next) {
    const filePath = req.file.path;
    const ext = path.extname(req.file.originalname).toLowerCase();
    const imageExts = [".jpg", ".jpeg", ".png", ".bmp", ".gif"];
    const videoExts = [".mp4", ".mov", ".avi", ".mkv", ".webm"];

    // Ensure cleanup happens regardless of processing errors
    try {
        let finalResponse;

        if (imageExts.includes(ext)) {
            // Send image file path to Python service for prediction
            const resultData = await callImageService(filePath);

            finalResponse = {
                prediction: {
                    label: resultData.predicted_class,
                    confidence: resultData.confidence,
                    explanation: resultData.explanation,
                },
                result_file: resultData.gradcam_image_url,
            };
        } else if (videoExts.includes(ext)) {
            const resultData = await callVideoService(filePath);

            finalResponse = {
                prediction: {
                    label: resultData.prediction,
                    confidence: resultData.confidence,
                    explanation: resultData.message,
                },
                result_file: resultData.gradcam_image_url,
            };
        } else {
            throw new ApiError("File type handler error.", 400);
        }

        res.json(finalResponse);

    } catch (err) {
        // Pass any error (including ApiError from callPythonService) to the centralized handler
        next(err);

    } finally {
        // IMPORTANT: Cleanup the temporary file regardless of success or failure
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            console.log(`Cleaned up uploaded file: ${filePath}`);
        }
    }
}

/**
 * Purpose: Centralized Express error handler middleware.
 * It formats all errors into a consistent JSON response for the client.
 */
function errorHandler(err, req, res, next) {
    let statusCode = err.statusCode || 500;
    let message = err.message || "An unknown internal server error occurred.";

    if (err instanceof multer.MulterError) {
        // Handle Multer specific errors (e.g., file size limit)
        statusCode = 400;
        if (err.code === 'LIMIT_FILE_SIZE') {
            message = "File is too large. Limit is 50MB.";
        }
    } else if (err.message && err.message.includes("Unsupported file type")) {
        // Handle Multer's file filter error
        statusCode = 415; // Unsupported Media Type
    }

    // Log the error for server-side debugging
    if (statusCode === 500) {
        console.error("Critical Server Error:", err);
    } else {
        console.warn(`Client Error (${statusCode}): ${message}`);
    }

    // Send consistent error response
    res.status(statusCode).json({
        error: {
            message: message,
            code: statusCode,
        }
    });
}

// --- 4. App Setup and Routing ---

// Initialize Express app
const app = express();
app.use(cors());
app.use(express.json());

// Ensure upload and results directories exist
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

// Configure Multer middleware
const upload = configureMulter();

// Define the file upload endpoint
// We use a custom function to catch Multer errors before passing to the next handler
app.post("/upload", (req, res, next) => {
    upload.single("file")(req, res, (err) => {
        if (err) {
            // If Multer throws an error (e.g., file size limit, bad file type), pass it to the error handler
            return next(err);
        }
        if (!req.file) {
            // No file uploaded field
            return next(new ApiError("No file provided in the 'file' field.", 400));
        }
        // If successful, proceed to prediction logic
        handleFileUploadAndPrediction(req, res, next);
    });
});

// Centralized Error Handling Middleware (must be defined last)
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
    console.log(`Node API Gateway running at http://localhost:${PORT}`);
    console.log(`Image Prediction Service expected at ${IMAGE_API_URL}`);
    console.log(`Video Prediction Service expected at ${VIDEO_API_URL}`);
});