import React, { useState, useRef } from "react";
import Roboflow from "roboflow";
import { Upload, Play, FileText, Lock, User, Download } from "lucide-react";
import Modal from "./components/Modal";
import ReportModal from "./components/ReportModal";
import Toast from "./components/Toast";
import LoginImage from "./assets/login_picture.png";
import Similarity from "./assets/image_simalrity.png";
import Segmentation from "./assets/sematic_segmentation.png";
			



function App() {
	const [isLoggedIn, setIsLoggedIn] = useState(false);
	const [showModal, setShowModal] = useState<"similarity" | "segmentation" | null>(null);
	const [showReportModal, setShowReportModal] = useState(false);
	const [toast, setToast] = useState<{ message: string; show: boolean }>({ message: "", show: false });
	const segmentationInputRef = useRef<HTMLInputElement>(null);
    const similarityInputRef = useRef<HTMLInputElement>(null);
	


	const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
	const [imageFile, setImageFile] = useState<File | null>(null); // For storing the uploaded file

	const rf = new Roboflow({
		apiKey: "zZmFTpQFBxJPAk8HUSy9", // Replace with your Roboflow API key
	});

	const handleLogin = (e: React.FormEvent) => {
		e.preventDefault();
		setIsLoggedIn(true);
	};

	const showToast = (message: string) => {
		setToast({ message, show: true });
		setTimeout(() => setToast({ message: "", show: false }), 3000);
	};

	const handleUpload = () => {
		if (showModal === "similarity" && similarityInputRef.current) {
			similarityInputRef.current.click();
		} else if (showModal === "segmentation" && segmentationInputRef.current) {
			segmentationInputRef.current.click();
		}		
	};

	// Handle file change for semantic segmentation
	const handleSegmentationFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files && e.target.files[0]) {
			const file = e.target.files[0];
			setImageFile(file);  // Save the uploaded file for further use

			const formData = new FormData();
			formData.append("file", file);

			try {
				// Assuming you have a Roboflow model for semantic segmentation
				const prediction = await rf.model("semantic-segmentation-cav0a/1").predict(imageFile); // Pass file directly // Replace with your model name
				setAnnotatedImage(prediction.annotated_image_url); // Set the annotated image URL
				setShowModal("segmentation"); // Open segmentation modal
				showToast("Uploaded for segmentation!");
			} catch (error) {
				console.error("Error during Roboflow API request:", error);
				showToast("Error in segmentation!");
			}
		}
	};

	// Handle file change for similarity analysis
	const handleSimilarityFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files && e.target.files[0]) {
			// You can implement file upload logic for similarity analysis here
			showToast("Uploaded for similarity!");
			setShowModal("similarity"); // Open similarity modal
		}
	};

	const handleDownload = () => {
		if (annotatedImage) {
			const link = document.createElement("a");
			link.href = annotatedImage;
			link.download = "annotated_image.png"; // Set the desired file name
			document.body.appendChild(link);
			link.click();
			document.body.removeChild(link);
			showToast("Downloaded!");
		} else {
			showToast("No annotated image available for download.");
		}
	};

	const handleAction = (action: string) => {
		switch (action) {
			case "execute":
				if (showModal === "similarity") {
					executeSimilarityAnalysis();  // Placeholder for Similarity Logic
				} else if (showModal === "segmentation") {
					executeSegmentationAnalysis();  // Segmentation logic with Roboflow API
				}
				break;
			case "report":
				setShowReportModal(true);
				break;
			case "download":
				handleDownload();
				break;
		}
		setShowModal(null);
	};

	// Similarity Execution Logic (Placeholder)
	const executeSimilarityAnalysis = () => {
		showToast("Similarity Analysis Executed!");
		// Insert your logic for executing similarity analysis (e.g., API call, processing, etc.)
		// Example:
		// const similarityScore = await compareImages(image1, image2);
		// Update your state or UI to display the results
	};

	// Segmentation Execution Logic (Roboflow API Call)
	const executeSegmentationAnalysis = () => {
		showToast("Segmentation Analysis Executed!");
		if (imageFile) {
		  const formData = new FormData();
		  formData.append("file", imageFile);
	  
		  rf.model("semantic-segmentation-cav0a/1").predict(formData)
			.then((response: { annotated_image_url: string }) => {
				const annotatedImageUrl = response.annotated_image_url;
				setAnnotatedImage(annotatedImageUrl);
			})
			  
			.catch((error: any) => {
				console.error("Error during segmentation:", error);
				showToast("Segmentation failed. Please try again.");
			  });
			  
		}
	  };
	// Handle file input change for similarity analysis	  

	if (!isLoggedIn) {
		return (
			<div className="min-h-screen relative">
				<div className="absolute inset-0">
					<div className="absolute inset-0 bg-black opacity-40 z-10"></div>
					<img
						src={LoginImage}
						alt="Satellite Map"
						className="w-full h-full object-cover brightness-150"
					/>
				</div>

				<div className="relative z-20 min-h-screen flex justify-end">
					<div className="w-full md:w-1/2 lg:w-[45%] bg-white bg-opacity-95 p-8 flex items-center">
						<div className="w-full max-w-md mx-auto space-y-8">
							<div>
								<h2 className="text-3xl font-bold text-gray-900">
									Welcome to GIC: Gespatial Image Comparison!{" "}
								</h2>
								<p className="mt-2 text-gray-600">
									Please sign in to access the dashboard
								</p>
							</div>
							<form onSubmit={handleLogin} className="space-y-6">
								<div>
									<label className="block text-sm font-medium text-gray-700">
										Username
									</label>
									<div className="mt-1 relative">
										<User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
										<input
											type="text"
											required
											className="pl-10 block w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 focus:border-blue-500 focus:ring-blue-500"
											placeholder="Enter your username"
										/>
									</div>
								</div>
								<div>
									<label className="block text-sm font-medium text-gray-700">
										Password
									</label>
									<div className="mt-1 relative">
										<Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
										<input
											type="password"
											required
											className="pl-10 block w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-900 focus:border-blue-500 focus:ring-blue-500"
											placeholder="Enter your password"
										/>
									</div>
								</div>
								<button
									type="submit"
									className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200"
								>
									Sign in
								</button>
							</form>
						</div>
					</div>
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-gray-50 p-8 flex gap-8">
			{/* Image Similarity Section */}
			<div
				className="w-1/2 h-[calc(100vh-4rem)] relative cursor-pointer group rounded-2xl overflow-hidden shadow-xl"
				onClick={() => setShowModal("similarity")}
			>
				<div className="relative w-full h-full">
					<div className="absolute inset-0 flex">
						<img
							src={Similarity}
							alt="Before"
							className="w-full h-full object-contain"
						/>
					</div>
					<div className="absolute inset-0 bg-black/20 backdrop-blur-sm transition-all duration-300 group-hover:backdrop-blur-none"></div>
					<div className="absolute inset-0 flex flex-col items-center justify-center p-8 transition-opacity duration-300 group-hover:opacity-0">
						<h2 className="text-4xl font-bold text-white mb-4 drop-shadow-lg">
							Image Similarity
						</h2>
						<p className="text-lg text-white text-center drop-shadow-lg">
							Click to analyze image similarity and compare
							geospatial data
						</p>
					</div>
				</div>
			</div>

			{/* Semantic Segmentation Section */}
			<div
				className="w-1/2 h-[calc(100vh-4rem)] relative cursor-pointer group rounded-2xl overflow-hidden shadow-xl"
				onClick={() => setShowModal("segmentation")}
			>
				<img
					src={Segmentation}
					alt="Semantic Segmentation"
					className="w-full h-full object-contain"
				/>
				<div className="absolute inset-0 bg-black/20 backdrop-blur-sm transition-all duration-300 group-hover:backdrop-blur-none"></div>
				<div className="absolute inset-0 flex flex-col items-center justify-center p-8 transition-opacity duration-300 group-hover:opacity-0">
					<h2 className="text-4xl font-bold text-white mb-4 drop-shadow-lg">
						Semantic Segmentation
					</h2>
					<p className="text-lg text-white text-center drop-shadow-lg">
						Click to perform semantic segmentation analysis
					</p>
				</div>
			</div>

			

			{/* Hidden file input for similarity */}
            <input
	            type="file"
	            ref={similarityInputRef}
	            onChange={handleSimilarityFileChange}
	            className="hidden"
	            accept="image/*"
            />


            {/* Hidden file input for segmentation */}
            <input
	            type="file"
	            ref={segmentationInputRef}
	            onChange={handleSegmentationFileChange}
	            className="hidden"
	            accept="image/*"
            />
		

			{/* Modal */}
			<Modal
				isOpen={showModal !== null}
				onClose={() => setShowModal(null)}
				title={
					showModal === "similarity"
						? "Image Similarity Analysis"
						: "Semantic Segmentation Analysis"
				}
			>
				<div className="space-y-4">
					<button
						onClick={handleUpload}
						className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200"
					>
						<Upload className="h-5 w-5" />
						Upload Image
					</button>
					<button
						onClick={() => handleAction("execute")}
						className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors duration-200"
					>
						<Play className="h-5 w-5" />
						Execute Analysis
					</button>
					{showModal === "similarity" ? (
						<button
							onClick={() => handleAction("report")}
							className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors duration-200"
						>
							<FileText className="h-5 w-5" />
							Generate Report
						</button>
					) : (
						<a
						 onClick={() => handleAction("download")}
						 className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors duration-200 cursor-pointer"
						 download={"annotation_image1.png"} // You can dynamically name it, or keep it constant.
						 href={annotatedImage || "#"}
						>
							{/* Use the annotated image URL for download */}
							{/* If the annotated image is not available, disable the link */}	
							<Download className="h-5 w-5" />
							Annotated Image
						</a>
					)}
				</div>
			</Modal>

			{/* Report Modal */}
			<ReportModal
				isOpen={showReportModal}
				onClose={() => setShowReportModal(false)}
			/>

			{/* Toast */}
			<Toast message={toast.message} show={toast.show} />
		</div>
	);
}

export default App;
