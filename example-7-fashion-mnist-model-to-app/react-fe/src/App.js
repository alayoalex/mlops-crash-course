import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Upload } from "lucide-react";
import { Alert, AlertDescription } from "./components/ui/alert";

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const onFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setImagePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const onFileUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first!");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      console.log(process.env.REACT_APP_API_URL)
      const api_url = window._env_?.REACT_APP_API_URL || process.env.REACT_APP_API_URL
      const response = await fetch("https://apifashion.mizpa.io/predict", {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      setError("Error during prediction. Please try again.");
      console.error("Error during prediction request:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-4 py-12 bg-gray-50 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto">
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-center text-gray-900">
              FashionMNIST Predictor
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* File Upload Section */}
            <div className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="file-upload"
                  className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    {imagePreview ? (
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="object-contain mb-4 max-h-48"
                      />
                    ) : (
                      <>
                        <Upload className="w-12 h-12 mb-4 text-gray-400" />
                        <p className="text-sm text-gray-500">
                          Click or drag to upload an image
                        </p>
                      </>
                    )}
                  </div>
                  <input
                    id="file-upload"
                    type="file"
                    className="hidden"
                    accept="image/png, image/jpeg"
                    onChange={onFileChange}
                  />
                </label>
              </div>

              <Button
                className="w-full"
                onClick={onFileUpload}
                disabled={isLoading || !selectedFile}
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <div className="mr-2 animate-spin">
                      <Upload className="w-4 h-4" />
                    </div>
                    Processing...
                  </div>
                ) : (
                  "Predict"
                )}
              </Button>
            </div>

            {/* Error Message */}
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Prediction Result */}
            {prediction && !error && (
              <div className="p-4 rounded-lg bg-green-50">
                <h3 className="text-lg font-medium text-green-800">
                  Prediction Result
                </h3>
                <p className="mt-2 text-green-700">
                  {prediction.class_name}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default App;
