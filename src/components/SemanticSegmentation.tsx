import React, { useRef, useState } from 'react';

const SemanticSegmentation = () => {
	const [selectedImage, setSelectedImage] = useState<File | null>(null);
	const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string | null>(null);
	const fileInputRef = useRef<HTMLInputElement>(null);

	const handleUpload = () => {
		if (fileInputRef.current) fileInputRef.current.click();
	};

	const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files && e.target.files[0]) {
			setSelectedImage(e.target.files[0]);
		}
	};

	const handleExecute = async () => {
		if (!selectedImage) return;
		const formData = new FormData();
		formData.append('file', selectedImage);

		const res = await fetch(
			`https://detect.roboflow.com/semantic-segmentation-cav0a/1?api_key=YOUR_API_KEY`,
			{
				method: 'POST',
				body: formData,
			}
		);

		const blob = await res.blob();
		const url = URL.createObjectURL(blob);
		setAnnotatedImageUrl(url);
	};

	return (
		<div className="space-y-4">
			<input
				type="file"
				ref={fileInputRef}
				onChange={handleFileChange}
				className="hidden"
				accept="image/*"
			/>

			<button onClick={handleUpload} className="btn blue">Upload Image</button>
			<button onClick={handleExecute} className="btn green">Execute Analysis</button>

			{annotatedImageUrl && (
				<a
					href={annotatedImageUrl}
					download="annotated_image.png"
					className="btn purple"
				>
					Download Annotated Image
				</a>
			)}
		</div>
	);
};

export default SemanticSegmentation;
