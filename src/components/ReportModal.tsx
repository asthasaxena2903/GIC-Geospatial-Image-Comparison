import React from "react";
import { X, Image, FileJson, BarChart3, Download } from "lucide-react";

interface ReportModalProps {
	isOpen: boolean;
	onClose: () => void;
}

const ReportModal: React.FC<ReportModalProps> = ({ isOpen, onClose }) => {
	if (!isOpen) return null;

	const handleDownload = (type: string) => {
		// Simulate download
		console.log(`Downloading ${type}`);
	};

	return (
		<div className="fixed inset-0 z-50 overflow-y-auto">
			<div className="flex min-h-screen items-center justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
				<div
					className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
					onClick={onClose}
				/>

				<div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
					<div className="absolute right-0 top-0 pr-4 pt-4">
						<button
							type="button"
							className="rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none"
							onClick={onClose}
						>
							<X className="h-6 w-6" />
						</button>
					</div>

					<div className="space-y-6">
						<h3 className="text-lg font-medium text-gray-900">
							Generated Reports
						</h3>

						{/* Compared Images */}
						<div className="bg-gray-50 p-4 rounded-lg hover:bg-gray-100 transition-colors">
							<div className="flex items-center justify-between">
								<div className="flex items-center space-x-3">
									<Image className="h-6 w-6 text-blue-500" />
									<span className="font-medium">
										Compared Images
									</span>
								</div>
								<a
									onClick={() => handleDownload("images")}
									className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 cursor-pointer"
									download={"compared_images.png"}
									href={"compared_images.png"}
								>
									<Download className="h-4 w-4" />
									<span>Download</span>
								</a>
							</div>
							<div className="mt-4">
								<img
									src="https://images.unsplash.com/photo-1548263594-a71ea65a8598?auto=format&fit=crop&q=80"
									alt="Comparison Preview"
									className="w-full h-32 object-cover rounded"
								/>
							</div>
						</div>

						{/* JSON File */}
						<div className="bg-gray-50 p-4 rounded-lg hover:bg-gray-100 transition-colors">
							<div className="flex items-center justify-between">
								<div className="flex items-center space-x-3">
									<FileJson className="h-6 w-6 text-green-500" />
									<span className="font-medium">
										JSON Analysis
									</span>
								</div>
								<a
									onClick={() => handleDownload("json")}
									className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 cursor-pointer"
									download={"output.json"}
									href={"output.json"}
								>
									<Download className="h-4 w-4" />
									<span>Download</span>
								</a>
							</div>
							<div className="mt-4 bg-gray-900 p-3 rounded">
								<pre className="text-xs text-green-400 overflow-x-auto">
									{JSON.stringify(
										{
											similarity_score: 0.89,
											matched_features: 127,
											timestamp: new Date().toISOString(),
										},
										null,
										2
									)}
								</pre>
							</div>
						</div>

						{/* Graph */}
						<div className="bg-gray-50 p-4 rounded-lg hover:bg-gray-100 transition-colors">
							<div className="flex items-center justify-between">
								<div className="flex items-center space-x-3">
									<BarChart3 className="h-6 w-6 text-purple-500" />
									<span className="font-medium">
										Analysis Graph
									</span>
								</div>
								<a
									onClick={() => handleDownload("graph")}
									className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 cursor-pointer"
									download={"analysis_graph.png"}
									href={"analysis_graph.png"}
								>
									<Download className="h-4 w-4" />
									<span>Download</span>
								</a>
							</div>
							<div className="mt-4 h-32 bg-white rounded border border-gray-200 flex items-center justify-center">
								<span className="text-gray-500">
									Graph Preview
								</span>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};

export default ReportModal;
