
### **Instructions to Run the Code (with Roboflow Model)**

1. **Clone the Repository**

   * `git clone https://github.com/your-username/your-repo.git`
   * `cd your-repo`

2. **Install Python & Node.js (if not already installed)**

3. **Python Backend Setup**

   * Create virtual environment:

     * Windows: `venv\Scripts\activate`
     * macOS/Linux: `source venv/bin/activate`
   * Install dependencies: `pip install -r requirements.txt`
   * Create a `.env` file and add:

     ```
     ROBOFLOW_API_KEY=your_roboflow_api_key
     ```

4. **React Frontend Setup**

   * Navigate to frontend folder: `cd frontend`
   * Install packages: `npm install`
   * Run the frontend: `npm start`

5. **Run the Backend Server**

   * Go to backend folder (if separate) or root
   * Run: `python app.py`

6. **Execution Notes**

   * First-time run may take 1–2 minutes to load the Roboflow model
   * Internet connection is required
   * Large images may increase processing time


