import { useState, useEffect, useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import toast, { Toaster } from "react-hot-toast";

// --- Modular Component: Processing Indicator ---
const ProcessingIndicator = () => (
  <svg
    className="animate-spin h-5 w-5 text-white"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
       {" "}
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    ></circle>
       {" "}
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    ></path>
     {" "}
  </svg>
);

// --- Custom Hook for Logging (Modularisation) ---
const useActivityLogger = () => {
  useEffect(() => {
    // Log visit (Example implementation)
    fetch("http://localhost:3000/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event: "interface_loaded" }),
    }).catch((e) => console.error("Logging failed:", e)); // Log exit using sendBeacon

    const handleBeforeUnload = () => {
      navigator.sendBeacon(
        "http://localhost:3000/log",
        JSON.stringify({ event: "interface_closed" })
      );
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, []);
};

// --- Main Component: Asset Submission Interface ---
function AssetSubmissionInterface() {
  useActivityLogger();
  const navigate = useNavigate();

  const [state, setState] = useState({
    file: null,
    previewUrl: null,
    isProcessing: false,
  }); // --- Utility: Cleanup previous preview URL on state change ---

  useEffect(() => {
    // This effect runs on component mount/unmount and when previewUrl changes.
    const cleanup = state.previewUrl;
    return () => {
      if (cleanup) {
        URL.revokeObjectURL(cleanup); // Efficient memory management
      }
    };
  }, [state.previewUrl]); // --- Handler: Asset Selection and Validation ---

  const handleFileSelection = useCallback(
    (e) => {
      const selectedFile = e.target.files[0]; // Strict media type validation
      const isPermittedAsset =
        selectedFile &&
        (selectedFile.type.startsWith("image/") ||
          selectedFile.type.startsWith("video/")); // Ensure immediate cleanup of any prior object URL
      if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
      }

      if (isPermittedAsset) {
        const newPreviewUrl = URL.createObjectURL(selectedFile);
        setState((s) => ({
          ...s,
          file: selectedFile,
          previewUrl: newPreviewUrl,
        }));
      } else {
        setState((s) => ({ ...s, file: null, previewUrl: null }));
        toast.error(
          "Asset type validation failed. Only standard image or video formats are accepted."
        );
      }
    },
    [state.previewUrl]
  ); // --- Handler: Data Transmission Logic (Upload) ---

  const initiateAnalysis = useCallback(async () => {
    const { file } = state;
    if (!file) return toast.error("Please select an asset for submission. 👆");

    setState((s) => ({ ...s, isProcessing: true })); // Lock interface
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:3000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        toast.success(
          data.message ||
            "Asset successfully transmitted. Awaiting analysis results... 🎉"
        );
        navigate("/result", { state: { data } });
      } else {
        // Handle non-200 responses (e.g., server-side validation error)
        const errorMessage =
          data.error ||
          `Submission failed with HTTP status ${res.status}. Review server logs.`;
        toast.error(`Submission Error: ${errorMessage}`);
        console.error("❌ Submission Failed Response:", data);
      }
    } catch (err) {
      console.error("❌ Network or Server Connectivity Error:", err);
      toast.error(
        "Critical connectivity failure. Ensure backend services are operational."
      );
    } finally {
      setState((s) => ({ ...s, isProcessing: false })); // Unlock interface
    }
  }, [state.file, navigate]); // --- Preview Renderer (Modularisation) ---

  const AssetPreview = useMemo(() => {
    const { previewUrl, file } = state;
    if (!previewUrl) {
      return (
        <div className="flex items-center justify-center w-full h-full border-2 border-dashed rounded-xl border-gray-600/40 text-gray-500 text-lg">
                    **Asset Preview Window** <br /> (Maximized view of submitted
          file)        {" "}
        </div>
      );
    }
    const isVideo = file.type.startsWith("video/");
    const mediaProps = {
      src: previewUrl,
      className:
        "rounded-xl shadow-2xl h-full w-full object-contain bg-black/50",
    };

    return isVideo ? (
      <video {...mediaProps} controls autoPlay loop muted />
    ) : (
      <img {...mediaProps} alt="Submitted Asset Preview" />
    );
  }, [state.previewUrl, state.file]); // --- Rendered Component ---

  return (
    <div className="bg-gray-950 min-h-screen p-10 flex gap-8">
      {/* --- Submission Control Panel (Left Side) --- */}     {" "}
      <div className="w-1/3 flex flex-col items-center justify-start gap-8 bg-gray-900 p-8 rounded-2xl shadow-2xl border border-gray-800">
        <h1 className="text-3xl font-extrabold text-blue-400 mb-4 w-full border-b border-gray-700 pb-3 text-left">
          Asset Submission Portal
        </h1>
                {/* File Dropzone/Input Label */}       {" "}
        <label
          htmlFor="asset-input-file"
          className="flex flex-col items-center justify-center w-full h-80 border-4 border-dashed rounded-xl cursor-pointer bg-gray-800/60 hover:bg-gray-700/60 border-blue-700/50 transition duration-300 ease-in-out shadow-inner shadow-gray-700"
        >
                    {/* SVG and Text */}         {" "}
          <div className="flex flex-col items-center justify-center p-6 text-center">
                       {" "}
            <svg
              className="w-16 h-16 mb-4 text-blue-500"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
                           {" "}
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 16v-1a3 3 0 013-3h10a3 3 0 013 3v1m-4-7l-4-4m0 0l-4 4m4-4v12"
              />
                         {" "}
            </svg>
                       {" "}
            <p className="mb-2 text-xl font-bold text-gray-200">
              Select File or Drag & Drop
            </p>
                       {" "}
            <p className="text-sm text-gray-400">
              Supported Formats:{" "}
              <span className="font-semibold text-blue-300">
                Image & Video Assets
              </span>
            </p>
                     {" "}
          </div>
                    {/* File Input */}
                   {" "}
          <input
            id="asset-input-file"
            type="file"
            className="hidden"
            accept="image/*,video/*"
            onChange={handleFileSelection}
            disabled={state.isProcessing} // Disable input while processing
          />
                 {" "}
        </label>
                {/* Upload Button with Indicator */}       {" "}
        <button
          onClick={initiateAnalysis}
          disabled={!state.file || state.isProcessing}
          className={`
                px-8 py-3 rounded-xl text-white font-extrabold text-lg shadow-xl transition-all w-full 
                flex items-center justify-center transform hover:scale-[1.02]
                ${
                  state.file && !state.isProcessing
                    ? "bg-green-600 hover:bg-green-700 cursor-pointer"
                    : "bg-gray-600 text-gray-400 cursor-not-allowed"
                }
            `}
        >
                   {" "}
          {state.isProcessing ? (
            <>
                            <ProcessingIndicator />             {" "}
              <span className="ml-3">Processing Request...</span>           {" "}
            </>
          ) : state.file ? (
            "Initiate Deepfake Analysis"
          ) : (
            "Ready for Asset Selection"
          )}
                 {" "}
        </button>
        <p className="text-xs text-gray-500 mt-2 w-full text-center">
          Data transmission is secured via HTTPS/TLS. Results will be displayed
          post-analysis.
        </p>
             {" "}
      </div>
      {/* --- Asset Preview Panel (Right Side) --- */}     {" "}
      <div className="w-2/3 flex flex-col gap-4">
        <h2 className="text-2xl font-bold text-gray-300 border-b border-gray-800 pb-2">
          Verification Asset Preview
        </h2>
        <div className="flex-grow flex items-center justify-center bg-gray-900 rounded-2xl shadow-inner shadow-black p-6 border border-gray-800">
                  {AssetPreview}       {" "}
        </div>
             {" "}
      </div>
            {/* Toast Provider (Global Notifications) */}
           {" "}
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: "#1f2937", // Dark background for toasts
            color: "#fff",
            border: "1px solid #374151",
          },
        }}
      />
         {" "}
    </div>
  );
}

export default AssetSubmissionInterface;